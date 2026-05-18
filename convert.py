import os
import sys
import subprocess
import glob
import shutil
import warnings
import torch
import numpy as np
import tensorflow as tf

# 🛡️ Vá numpy.load (chạy trước import onnx2tf)
_orig_np_load = np.load
def _safe_np_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_np_load(*args, **kwargs)
np.load = _safe_np_load
np.lib.npyio.load = _safe_np_load

import onnx2tf
# Bypass test data load
try:
    import onnx2tf.utils.common_functions as _o2t_cf
    _o2t_cf.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except: pass
try:
    import onnx2tf.onnx2tf as _o2t_main
    if hasattr(_o2t_main, 'download_test_image_data'):
        _o2t_main.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except: pass

from basicsr.archs.rrdbnet_arch import RRDBNet
warnings.filterwarnings('ignore')

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"
INPUT_SHAPE = (1, 3, 256, 256)  # NCHW

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

# 3. Chuyển ONNX -> TF SavedModel (Dùng làm gốc để quantize)
print("3️⃣ Converting ONNX -> TensorFlow SavedModel...")
if os.path.exists("saved_model"): shutil.rmtree("saved_model")
onnx2tf.convert(
    input_onnx_file_path=ONNX_SIM_FILE,
    output_folder_path="saved_model",
    overwrite_input_shape=["input:1,3,256,256"],
    output_saved_model=True,  # ✅ Xuất SavedModel thay vì .tflite
    verbosity="warn"
)
print("✅ SavedModel generated.")

# 4. Quantize bằng TF Native Converter (Chuẩn & Ổn định nhất)
print("4️⃣ Generating TFLite variants (Float32 / Float16 / Int8)...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_data():
    for _ in range(50):
        yield [np.random.uniform(0, 1, INPUT_SHAPE).astype(np.float32)]

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

success_files = []
for name, cfg in variants.items():
    print(f"🔄 Processing {name.upper()}...")
    try:
        c = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        c.optimizations = cfg["opt"]
        for k, v in cfg["args"].items():
            setattr(c, k.replace(".", "_"), v)
            
        tflite_bytes = c.convert()
        out_path = f"RealESRGAN_x2plus_{name}.tflite"
        with open(out_path, "wb") as f: f.write(tflite_bytes)
        success_files.append((out_path, len(tflite_bytes)/1024/1024))
        print(f"✅ {out_path} ({len(tflite_bytes)/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Skip {name}: {str(e)[:150]}... (Int8 thường bị skip do op không hỗ trợ quantize)")

# 5. Dọn dẹp
for d in ["tflite_out", "saved_model"]:
    if os.path.exists(d): shutil.rmtree(d)
print(f"\n📦 Thành công {len(success_files)} file: {[f[0] for f in success_files]}")
