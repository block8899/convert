import os
import sys
import subprocess
import glob
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
TFLITE_FILE = "RealESRGAN_x2plus.tflite"

# ✅ Bước 1: Tải file từ link nếu chưa có
if not os.path.exists(PTH_FILE):
    print(f"⬇️ Đang tải {PTH_FILE} từ GitHub Releases...")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    subprocess.run(["wget", "-q", "--show-progress", url, "-O", PTH_FILE], check=True)
    print("✅ Tải xong!")

# ✅ Bước 2: Load model RealESRGAN
print("1️⃣ Load kiến trúc RealESRGAN...")
model = RRDBNet(
    num_in_ch=3, 
    num_out_ch=3, 
    num_feat=64, 
    num_block=23, 
    num_grow_ch=32, 
    scale=2
)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# ✅ Bước 3: Export sang ONNX (dùng input shape cố định để TFLite ổn định)
print("2️⃣ Export sang ONNX...")
dummy_input = torch.randn(1, 3, 256, 256)  # batch=1, RGB, 256x256
torch.onnx.export(
    model, dummy_input, ONNX_FILE,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print(f"✅ ONNX export xong: {os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB")

# ✅ Bước 4: Convert ONNX → TFLite
print("3️⃣ Convert ONNX → TFLite...")
subprocess.run(
    [sys.executable, "-m", "onnx2tf", "-i", ONNX_FILE, "-o", "tflite_out", "-otfl", "-oniwg"],
    check=True,
    capture_output=True,
    text=True
)

# Lấy file float32.tflite vừa sinh
tflite_candidates = glob.glob("tflite_out/*_float32.tflite")
if tflite_candidates:
    os.rename(tflite_candidates[0], TFLITE_FILE)
    size_mb = os.path.getsize(TFLITE_FILE) / 1024 / 1024
    print(f"🎉 THÀNH CÔNG! File: {TFLITE_FILE} | Kích thước: {size_mb:.2f} MB")
else:
    # Thử tìm file .tflite bất kỳ nếu không tìm thấy float32
    fallback = glob.glob("tflite_out/*.tflite")
    if fallback:
        os.rename(fallback[0], TFLITE_FILE)
        print(f"🎉 THÀNH CÔNG (fallback)! File: {TFLITE_FILE}")
    else:
        print("❌ Không tìm thấy file .tflite đầu ra. Kiểm tra log onnx2tf phía trên.")
        sys.exit(1)
