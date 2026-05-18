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

# 🛡️ 1. Vá numpy.load cho phép pickle (chạy trước import onnx2tf)
_orig_np_load = np.load
def _safe_np_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_np_load(*args, **kwargs)
np.load = _safe_np_load
np.lib.npyio.load = _safe_np_load

# 🛡️ 2. Import onnx2tf & bypass test data
import onnx2tf
try:
    import onnx2tf.utils.common_functions as _cf
    _cf.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except: pass
try:
    import onnx2tf.onnx2tf as _main
    if hasattr(_main, 'download_test_image_data'):
        _main.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except: pass

from basicsr.archs.rrdbnet_arch import RRDBNet
warnings.filterwarnings('ignore')

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"
INPUT_SHAPE = (1, 3, 256, 256)

# 1. Tải & Load Model
if not os.path.exists(PTH_FILE):
    print("⬇️ Downloading RealESRGAN_x2plus.pth...")
    subprocess.run(["wget", "-q", "--show-progress", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", "-O", PTH_FILE], check=True)
    print("✅ Download complete.")

print("1️⃣ Loading architecture...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# 2. Export & Simplify ONNX
print("2️⃣ Exporting & Simplifying ONNX...")
torch.onnx.export(model, torch.randn(*INPUT_SHAPE), ONNX_FILE,
                  export_params=True, opset_version=14, do_constant_folding=True,
                  input_names=["input"], output_names=["output"], verbose=False)
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True, capture_output=True)
print(f"✅ ONNX simplified: {os.path.getsize(ONNX_SIM_FILE)/1024/1024:.1f} MB")

# 3. Chuyển ONNX -> TF SavedModel (DÙNG PYTHON API + DYNAMIC SIGNATURE)
print("3️⃣ Converting ONNX -> TensorFlow SavedModel...")
if os.path.exists("saved_model"): shutil.rmtree("saved_model")

# Tự động detect tham số tương thích với phiên bản onnx2tf đang cài
sig = inspect.signature(onnx2tf.convert)
params = sig.parameters
convert_kwargs = {
    "input_onnx_file_path": ONNX_SIM_FILE,
    "output_folder_path": "saved_model",
    "overwrite_input_shape": ["input:1,3,256,256"],
    "verbosity": "warn"
}
if "not_use_test_data" in params:
    convert_kwargs["not_use_test_data"] = True
if "output_saved_model" in params:
    convert_kwargs["output_saved_model"] = True

try:
    onnx2tf.convert(**convert_kwargs)
    print("✅ SavedModel generated.")
except Exception as e:
    print(f"❌ Failed to create SavedModel: {e}")
    sys.exit(1)

# 4. Quantize bằng TF Native Converter (API chuẩn)
print("4️⃣ Generating TFLite variants...")
def rep_data():
    for _ in range(50):
        yield [np.random.uniform(0, 1, INPUT_SHAPE).astype(np.float32)]

success_files = []
for name in ["float32", "float16", "int8"]:
    print(f"🔄 Processing {name.upper()}...")
    try:
        c = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        
        if name == "float32":
            c.optimizations = []
        elif name == "float16":
            c.optimizations = [tf.lite.Optimize.DEFAULT]
            c.target_spec.supported_types = [tf.float16]
        elif name == "int8":
            c.optimizations = [tf.lite.Optimize.DEFAULT]
            c.representative_dataset = rep_data
            c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            c.inference_input_type = tf.uint8
            c.inference_output_type = tf.uint8

        tflite_bytes = c.convert()
        out_path = f"RealESRGAN_x2plus_{name}.tflite"
        with open(out_path, "wb") as f: f.write(tflite_bytes)
        success_files.append((out_path, len(tflite_bytes)/1024/1024))
        print(f"✅ {out_path} ({len(tflite_bytes)/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Skip {name}: {str(e)[:120]}...")

# 5. Dọn dẹp
for d in ["saved_model", "tflite_out"]:
    if os.path.exists(d): shutil.rmtree(d)
print(f"\n📦 Thành công {len(success_files)} file: {[f[0] for f in success_files]}")
