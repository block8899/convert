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

# 📌 Cố định input shape: batch=1, RGB, 256x256 (chia hết cho scale=2)
INPUT_SHAPE = (1, 3, 256, 256)

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

# 3. Export ONNX với STATIC SHAPE (KHÔNG dùng dynamic_axes)
print("2️⃣ Exporting to ONNX (static shape)...")
dummy_input = torch.randn(*INPUT_SHAPE)
torch.onnx.export(
    model, dummy_input, ONNX_FILE,
    export_params=True,
    opset_version=14,  # Opset 14 hỗ trợ Resize tốt nhất
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    # ❌ KHÔNG dùng dynamic_axes → ép shape tĩnh để onnx2tf không crash
    verbose=False
)
print(f"✅ ONNX exported: {os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB")

# 4. Simplify ONNX
print("3️⃣ Simplifying ONNX graph...")
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True)
print("✅ ONNX simplified.")

# 5. Convert to TFLite với cờ FIX SHAPE
print("4️⃣ Converting to TFLite... (wait 3-6 mins)")
if os.path.exists("tflite_out"):
    shutil.rmtree("tflite_out")

# ✅ Cờ quan trọng:
# -ois: cố định input shape dạng "name:dim1,dim2,..."
# -kt: giữ channel order NCHW → NHWC cho TFLite
# -onwdt: fix shape inference cho các op đặc biệt
cmd = [
    sys.executable, "-m", "onnx2tf",
    "-i", ONNX_SIM_FILE,
    "-o", "tflite_out",
    "-ois", "input:1,3,256,256",  # ✅ Fix input shape tĩnh
    "-kt",                          # ✅ Fix channel order
    "-onwdt"                        # ✅ Fix shape inference
]
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ onnx2tf failed with exit code {e.returncode}")
    sys.exit(1)

# 6. Lấy file kết quả
tflite_files = glob.glob("tflite_out/*.tflite")
if tflite_files:
    target = next((f for f in tflite_files if "float32" in f), tflite_files[0])
    os.rename(target, TFLITE_FILE)
    print(f"🎉 SUCCESS! {TFLITE_FILE} ({os.path.getsize(TFLITE_FILE)/1024/1024:.1f} MB)")
else:
    print("❌ No .tflite file generated. Check logs above.")
    sys.exit(1)
