import torch
import pnnx
import os
import subprocess
import sys
import gc

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
print("   (This may take several minutes for RRDBNet 23 blocks)")

# ★ Giảm bộ nhớ: disable gradient
torch.set_grad_enabled(False)

dummy = torch.randn(1, 3, 256, 256)

try:
    pnnx.export(model, "esrgan", inputs=dummy)
    print("3. PNNX export done!")
except Exception as e:
    print(f"PNNX failed: {e}")
    # ★ Fallback: thử export nhỏ hơn
    print("Trying with smaller input...")
    try:
        dummy_small = torch.randn(1, 3, 64, 64)
        pnnx.export(model, "esrgan", inputs=dummy_small)
        print("3. PNNX export done (small input)!")
    except Exception as e2:
        print(f"PNNX also failed with small input: {e2}")
        sys.exit(1)

# ★ Cleanup
del model, dummy
gc.collect()

print("4. Verifying...")
if os.path.exists("esrgan.param") and os.path.exists("esrgan.bin"):
    size_param = os.path.getsize("esrgan.param") / 1024
    size_bin = os.path.getsize("esrgan.bin") / 1024 / 1024
    print(f"  esrgan.param: {size_param:.1f} KB")
    print(f"  esrgan.bin: {size_bin:.1f} MB")

    # Đọc param file để verify
    with open("esrgan.param", "r") as f:
        lines = f.readlines()
        print(f"  Layers: {len(lines) - 2}")

    print("NCNN OK!")
else:
    print("NCNN conversion FAILED!")
    print(f"Files in current dir: {os.listdir('.')}")
    sys.exit(1)
