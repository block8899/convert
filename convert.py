import os
import sys
import subprocess
import glob
import shutil
import warnings
import torch
import numpy as np

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
    import onnx2tf.utils.common_functions as _o2t_cf
    _o2t_cf.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except Exception: pass
try:
    import onnx2tf.onnx2tf as _o2t_main
    if hasattr(_o2t_main, 'download_test_image_data'):
        _o2t_main.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except Exception: pass

from basicsr.archs.rrdbnet_arch import RRDBNet
warnings.filterwarnings('ignore')

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"
INPUT_SHAPE = (1, 3, 256, 256)  # NCHW

# 1. Tải model
if not os.path.exists(PTH_FILE):
    print("⬇️ Downloading RealESRGAN_x2plus.pth...")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    subprocess.run(["wget", "-q", "--show-progress", url, "-O", PTH_FILE], check=True)
    print("✅ Download complete.")

# 2. Load model
print("1️⃣ Loading architecture...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# 3. Export ONNX (static shape)
print("2️⃣ Exporting to ONNX...")
torch.onnx.export(model, torch.randn(*INPUT_SHAPE), ONNX_FILE,
                  export_params=True, opset_version=14, do_constant_folding=True,
                  input_names=["input"], output_names=["output"], verbose=False)
print(f"✅ ONNX: {os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB")

# 4. Simplify ONNX
print("3️⃣ Simplifying...")
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True, capture_output=True)
print("✅ Simplified.")

# 5. 🔄 Convert đa định dạng (Float32 / Float16 / Int8)
print("4️⃣ Converting formats...")
if os.path.exists("tflite_out"): shutil.rmtree("tflite_out")

# Dữ liệu calibration cho Int8 (20 mẫu ngẫu nhiên range [0,1])
def calib_data():
    for _ in range(20):
        yield [np.random.uniform(0, 1, INPUT_SHAPE).astype(np.float32)]

configs = [
    {"name": "float32", "args": {}},
    {"name": "float16", "args": {"output_float16": True}},
    {"name": "int8",    "args": {"output_integer_quantization": True, 
                                 "integer_quantization_type": "per-tensor",
                                 "representative_dataset": calib_data}}
]

success_files = []
for cfg in configs:
    out_dir = f"tflite_{cfg['name']}"
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    print(f"🔄 Processing {cfg['name'].upper()}...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=ONNX_SIM_FILE,
            output_folder_path=out_dir,
            overwrite_input_shape=["input:1,3,256,256"],
            verbosity="warn",
            **cfg["args"]
        )
        files = glob.glob(f"{out_dir}/*.tflite")
        if files:
            out_name = f"RealESRGAN_x2plus_{cfg['name']}.tflite"
            os.rename(files[0], out_name)
            size_mb = os.path.getsize(out_name) / 1024 / 1024
            success_files.append(out_name)
            print(f"✅ {out_name} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️ Không sinh được file {cfg['name']}")
    except Exception as e:
        print(f"❌ Skip {cfg['name']}: {e}")

# 6. Dọn dẹp
if os.path.exists("tflite_out"): shutil.rmtree("tflite_out")
print(f"\n📦 Tổng cộng {len(success_files)} file thành công: {success_files}")
