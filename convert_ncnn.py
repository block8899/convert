#!/usr/bin/env python3
"""
Convert RealESRGAN PyTorch model to NCNN format using PNNX (command-line tool)
"""
import os
import sys
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
PT_FILE = "realesrgan.pt"  # TorchScript intermediate

# Download model if not exists
if not os.path.exists(PTH_FILE):
    print("📥 Downloading model...")
    subprocess.run([
        "wget", "-q", "--max-redirect=5",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "-O", PTH_FILE
    ], check=True)

print("1️⃣ Loading PyTorch model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

print("2️⃣ Exporting to TorchScript (.pt)...")
dummy_input = torch.randn(1, 3, 256, 256)
try:
    traced = torch.jit.trace(model, dummy_input, strict=False)
    traced.save(PT_FILE)
    print(f"✓ Saved: {PT_FILE}")
except Exception as e:
    print(f"❌ TorchScript export failed: {e}")
    # Fallback: try script mode
    try:
        scripted = torch.jit.script(model)
        scripted.save(PT_FILE)
        print(f"✓ Saved via script mode: {PT_FILE}")
    except Exception as e2:
        print(f"❌ Both trace and script failed: {e2}")
        sys.exit(1)

print("3️⃣ Converting to NCNN via PNNX CLI...")
# PNNX is a CLI tool, not a Python module
pnnx_cmd = [
    "pnnx",
    PT_FILE,
    "inputshape=[1,3,256,256]",
    "fp16=1",
    "optlevel=2",
    f"moduleop={model.__class__.__module__}.{model.__class__.__name__}"
]
print(f"🔧 Running: {' '.join(pnnx_cmd)}")

result = subprocess.run(pnnx_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ PNNX failed:\n{result.stderr}")
    sys.exit(1)
else:
    print("✓ PNNX conversion completed")
    if result.stdout:
        print(f"📋 Output:\n{result.stdout[:500]}")  # Show first 500 chars

print("4️⃣ Verifying NCNN files...")
param_file = "realesrgan.ncnn.param"
bin_file = "realesrgan.ncnn.bin"

if os.path.exists(param_file) and os.path.exists(bin_file):
    size_param = os.path.getsize(param_file) / 1024
    size_bin = os.path.getsize(bin_file) / 1024 / 1024
    print(f"  ✓ {param_file}: {size_param:.1f} KB")
    print(f"  ✓ {bin_file}: {size_bin:.1f} MB")
    print("🎉 NCNN conversion SUCCESS!")
else:
    # Try alternative naming (pnnx may output without .ncnn prefix)
    alt_param = "realesrgan_pnnx.param"
    alt_bin = "realesrgan_pnnx.bin"
    if os.path.exists(alt_param) and os.path.exists(alt_bin):
        os.rename(alt_param, param_file)
        os.rename(alt_bin, bin_file)
        print(f"  ✓ Renamed alternative files to NCNN format")
    else:
        print("❌ NCNN files not found!")
        print("📁 Files in directory:", os.listdir("."))
        sys.exit(1)
