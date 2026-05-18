import os
import sys
import subprocess
import glob
import shutil
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"
TFLITE_FILE = "RealESRGAN_x2plus.tflite"

INPUT_SHAPE = (1, 3, 256, 256)  # NCHW

# 1. Tải model
if not os.path.exists(PTH_FILE):
    print("⬇️ Downloading RealESRGAN_x2plus.pth...")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    subprocess.run(["wget", "-q", "--show-progress", url, "-O", PTH_FILE], check=True)
    print("✅ Download complete.")

# 2. Load model
print("1️⃣ Loading RealESRGAN architecture...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# 3. Export ONNX (static shape)
print("2️⃣ Exporting to ONNX (static shape)...")
dummy_input = torch.randn(*INPUT_SHAPE)
torch.onnx.export(
    model, dummy_input, ONNX_FILE,
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=["input"], output_names=["output"], verbose=False
)
print(f"✅ ONNX exported: {os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB")

# 4. Simplify ONNX
print("3️⃣ Simplifying ONNX graph...")
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True)
print("✅ ONNX simplified.")

# 5. Convert to TFLite
print("4️⃣ Converting to TFLite... (wait 3-6 mins)")
if os.path.exists("tflite_out"):
    shutil.rmtree("tflite_out")

# ✅ FIX: -v "info" thay vì -v "1"
cmd = [
    sys.executable, "-m", "onnx2tf",
    "-i", ONNX_SIM_FILE,
    "-o", "tflite_out",
    "-ois", "input:1,3,256,256",
    "-v", "info"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
except subprocess.CalledProcessError as e:
    print("❌ onnx2tf FAILED!")
    print("🔴 STDERR:", e.stderr[-2000:] if e.stderr else "None")
    sys.exit(1)

# 6. Lấy file kết quả
tflite_files = glob.glob("tflite_out/*.tflite")
if tflite_files:
    target = next((f for f in tflite_files if "float32" in f), tflite_files[0])
    os.rename(target, TFLITE_FILE)
    print(f"🎉 SUCCESS! {TFLITE_FILE} ({os.path.getsize(TFLITE_FILE)/1024/1024:.1f} MB)")
else:
    print("❌ No .tflite file generated.")
    print("📁 tflite_out:", os.listdir("tflite_out") if os.path.exists("tflite_out") else "Not found")
    sys.exit(1)
