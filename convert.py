import os
import sys
import subprocess
import torch
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
TFLITE_FILE = "RealESRGAN_x2plus.tflite"

if not os.path.exists(PTH_FILE):
    raise FileNotFoundError(f"❌ Không tìm thấy {PTH_FILE}. Hãy upload file vào repo.")

print("1️⃣ Load kiến trúc RealESRGAN & weights...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu")
# RealESRGAN lưu weight trong key 'params_ema' hoặc 'params'
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

print("2️⃣ Export sang ONNX...")
# Dùng shape cố định 1x3x256x256 để TFLite tương thích ổn định nhất
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model, dummy_input, ONNX_FILE,
    export_params=True, opset_version=14,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("✅ ONNX export thành công!")

print("3️⃣ Convert ONNX → TFLite (dùng onnx2tf)...")
subprocess.run(
    [sys.executable, "-m", "onnx2tf", "-i", ONNX_FILE, "-o", "tflite_out", "-otfl"],
    check=True
)

# Lấy file float32 vừa sinh ra
tflite_list = glob.glob("tflite_out/*_float32.tflite")
if tflite_list:
    os.rename(tflite_list[0], TFLITE_FILE)
    size_mb = os.path.getsize(TFLITE_FILE) / 1024 / 1024
    print(f"🎉 HOÀN TẤT! File: {TFLITE_FILE} | Size: {size_mb:.2f} MB")
else:
    print("❌ Lỗi convert. Kiểm tra log phía trên.")
