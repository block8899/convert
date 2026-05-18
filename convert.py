import os
import sys
import subprocess
import glob
import shutil
import warnings
import inspect
import torch
import numpy as np
import tensorflow as tf

# 🛡️ 1. Vá numpy.load (chạy TRƯỚC import onnx2tf)
_orig_np_load = np.load
def _safe_np_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_np_load(*args, **kwargs)
np.load = _safe_np_load
np.lib.npyio.load = _safe_np_load

# 🛡️ 2. Import & bypass test data
import onnx2tf
try: onnx2tf.utils.common_functions.download_test_image_data = lambda: np.zeros((1,3,256,256), dtype=np.float32)
except: pass
try: onnx2tf.onnx2tf.download_test_image_data = lambda: np.zeros((1,3,256,256), dtype=np.float32)
except: pass

from basicsr.archs.rrdbnet_arch import RRDBNet
warnings.filterwarnings('ignore')

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"
INPUT_SHAPE = (1, 3, 256, 256)

# 1. Tải & Load Model
if not os.path.exists(PTH_FILE):
    print("⬇️ Downloading...")
    subprocess.run(["wget", "-q", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", "-O", PTH_FILE], check=True)
print("1️⃣ Loading architecture...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# 2. Export & Simplify ONNX
print("2️⃣ Exporting & Simplifying ONNX...")
torch.onnx.export(model, torch.randn(*INPUT_SHAPE), ONNX_FILE, export_params=True, opset_version=14, do_constant_folding=True,
                  input_names=["input"], output_names=["output"], verbose=False)
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True, capture_output=True)
print(f"✅ ONNX simplified: {os.path.getsize(ONNX_SIM_FILE)/1024/1024:.1f} MB")

# 3. ONNX -> SavedModel
print("3️⃣ Converting to TensorFlow SavedModel...")
if os.path.exists("saved_model"): shutil.rmtree("saved_model")
sig = inspect.signature(onnx2tf.convert)
params = sig.parameters
kwargs = {"input_onnx_file_path": ONNX_SIM_FILE, "output_folder_path": "saved_model",
          "overwrite_input_shape": ["input:1,3,256,256"], "verbosity": "warn"}
if "not_use_test_data" in params: kwargs["not_use_test_data"] = True
if "output_saved_model" in params: kwargs["output_saved_model"] = True

onnx2tf.convert(**kwargs)

# Tự động tìm thư mục chứa saved_model.pb (đề phòng onnx2tf tạo lồng folder)
sm_dir = "saved_model"
if not os.path.exists(os.path.join(sm_dir, "saved_model.pb")):
    for root, dirs, files in os.walk(sm_dir):
        if "saved_model.pb" in files:
            sm_dir = root
            break
print(f"✅ SavedModel ready at: {sm_dir}")

# 4. Quantize với TF Native Converter
print("4️⃣ Generating TFLite variants...")
def rep_data():
    for _ in range(30): yield [np.random.uniform(0, 1, INPUT_SHAPE).astype(np.float32)]

variants = {
    "float32": {"opt": [], "args": {}},
    "float16": {"opt": [tf.lite.Optimize.DEFAULT], "args": {"target_spec.supported_types": [tf.float16]}},
    "int8":    {"opt": [tf.lite.Optimize.DEFAULT], "args": {
                    "representative_dataset": rep_data,
                    "target_spec.supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                    "inference_input_type": tf.uint8,
                    "inference_output_type": tf.uint8
                }}
}

generated = []
for name, cfg in variants.items():
    try:
        c = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
        c.optimizations = cfg["opt"]
        for k, v in cfg["args"].items(): setattr(c, k.replace(".", "_"), v)
        tflite_bytes = c.convert()
        fname = f"RealESRGAN_x2plus_{name}.tflite"
        with open(fname, "wb") as f: f.write(tflite_bytes)
        generated.append(fname)
        print(f"✅ {fname} ({len(tflite_bytes)/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Skip {name}: {str(e)[:100]}...")

# 5. Kiểm tra kết quả
if not generated:
    print("❌ Không sinh được file .tflite nào! Kiểm tra log phía trên.")
    print("📁 Workspace contents:", os.listdir("."))
    sys.exit(1)

print(f"\n📦 Thành công {len(generated)} file: {generated}")
