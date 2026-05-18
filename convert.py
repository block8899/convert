import os
import sys
import subprocess
import glob
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"
TFLITE_FILE = "RealESRGAN_x2plus.tflite"

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

# 3. Export ONNX
print("2️⃣ Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model, dummy_input, ONNX_FILE,
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, 
                  "output": {0: "batch", 2: "height", 3: "width"}}
)
print(f"✅ ONNX exported: {os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB")

# 4. Simplify ONNX (bắt buộc để onnx2tf không crash)
print("3️⃣ Simplifying ONNX graph...")
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True)
print("✅ ONNX simplified.")

# 5. Convert to TFLite (STREAM LOG TRỰC TIẾP để thấy lỗi thật)
print("4️⃣ Converting to TFLite... (wait 2-5 mins)")
cmd = [sys.executable, "-m", "onnx2tf", "-i", ONNX_SIM_FILE, "-o", "tflite_out", "-otfl", "-v", "1"]
try:
    # Không capture_output, để log hiện thẳng ra GitHub Actions
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
