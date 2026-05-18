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
    scale=2  # Scale factor = 2
)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# ✅ Bước 3: Export sang ONNX
# 📌 Input shape phải chia hết cho 4 (do RRDBNet có 4 block downsample)
# Dùng 260x260 hoặc 512x512 để an toàn nhất
print("2️⃣ Export sang ONNX...")
dummy_input = torch.randn(1, 3, 260, 260)  # ✅ 260 chia hết cho 4
torch.onnx.export(
    model, dummy_input, ONNX_FILE,
    export_params=True,
    opset_version=14,  # Opset 14 ổn định nhất cho RealESRGAN
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, 
                  "output": {0: "batch", 2: "height", 3: "width"}},
    verbose=False
)
print(f"✅ ONNX export xong: {os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB")

# ✅ Bước 4: Convert ONNX → TFLite
print("3️⃣ Convert ONNX → TFLite...")
# 📌 Loại bỏ -oniwg, thêm --non_verbose để log rõ hơn
cmd = [
    sys.executable, "-m", "onnx2tf",
    "-i", ONNX_FILE,
    "-o", "tflite_out",
    "-otfl",  # Output TFLite float32
    "--non_verbose",  # Log ngắn gọn, dễ đọc lỗi
    "-onwdt",  # Output node width dimension type (fix shape mismatch)
    "-kat"  # Keep activation type (tránh lỗi precision)
]

try:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    print("📦 onnx2tf output:\n", result.stdout[-500:])  # In 500 ký tự cuối để xem log
except subprocess.CalledProcessError as e:
    print("❌ Lỗi onnx2tf chi tiết:")
    print("STDOUT:", e.stdout[-1000:] if e.stdout else "None")
    print("STDERR:", e.stderr[-1000:] if e.stderr else "None")
    sys.exit(1)

# Lấy file float32.tflite vừa sinh
tflite_candidates = glob.glob("tflite_out/*_float32.tflite")
if not tflite_candidates:
    tflite_candidates = glob.glob("tflite_out/*.tflite")  # Fallback tìm bất kỳ file .tflite

if tflite_candidates:
    os.rename(tflite_candidates[0], TFLITE_FILE)
    size_mb = os.path.getsize(TFLITE_FILE) / 1024 / 1024
    print(f"🎉 THÀNH CÔNG! File: {TFLITE_FILE} | Kích thước: {size_mb:.2f} MB")
else:
    print("❌ Không tìm thấy file .tflite đầu ra.")
    print("📁 Contents of tflite_out/:", os.listdir("tflite_out") if os.path.exists("tflite_out") else "Folder không tồn tại")
    sys.exit(1)
