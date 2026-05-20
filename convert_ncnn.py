import torch
import pnnx
import os
import subprocess
import sys

from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"

if not os.path.exists(PTH_FILE):
    print("Downloading model...")
    subprocess.run([
        "wget", "-q", "--max-redirect=5",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "-O", PTH_FILE
    ], check=True)

print("1. Loading model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

print("2. Converting to NCNN via PNNX...")
dummy = torch.randn(1, 3, 256, 256)
pnnx.export(model, "esrgan", inputs=dummy)

print("3. Verifying...")
if os.path.exists("esrgan.param") and os.path.exists("esrgan.bin"):
    size_param = os.path.getsize("esrgan.param") / 1024
    size_bin = os.path.getsize("esrgan.bin") / 1024 / 1024
    print(f"  esrgan.param: {size_param:.1f} KB")
    print(f"  esrgan.bin: {size_bin:.1f} MB")
    print("NCNN OK!")
else:
    print("NCNN conversion FAILED!")
    sys.exit(1)
