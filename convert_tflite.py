#!/usr/bin/env python3
"""
Convert RealESRGAN PyTorch model to TFLite format via ONNX → TensorFlow
"""
import os
import sys
import subprocess
import shutil
import warnings
import torch
import numpy as np
import tensorflow as tf

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"

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

print("2️⃣ Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 256, 256)
try:
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FILE,
        export_params=True,
        opset_version=13,  # Use 13 for better TFLite compatibility
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        verbose=False,
    )
    print(f"✓ Saved: {ONNX_FILE} ({os.path.getsize(ONNX_FILE)/1024/1024:.1f} MB)")
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    sys.exit(1)

print("3️⃣ Simplifying ONNX...")
try:
    subprocess.run(
        [sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE],
        check=True,
        capture_output=True,
        timeout=120
    )
    print(f"✓ Simplified: {ONNX_SIM_FILE} ({os.path.getsize(ONNX_SIM_FILE)/1024/1024:.1f} MB)")
except subprocess.TimeoutExpired:
    print("⚠️ ONNX simplify timeout, using original ONNX")
    ONNX_SIM_FILE = ONNX_FILE
except Exception as e:
    print(f"⚠️ ONNX simplify failed: {e}, using original ONNX")
    ONNX_SIM_FILE = ONNX_FILE

print("4️⃣ Converting ONNX → TensorFlow SavedModel...")
if os.path.exists("saved_model"):
    shutil.rmtree("saved_model")

try:
    import onnx2tf
    convert_kwargs = {
        "input_onnx_file_path": ONNX_SIM_FILE,
        "output_folder_path": "saved_model",
        "overwrite_input_shape": ["1,3,256,256"],
        "verbosity": "error",
        "not_use_onnxsim": True,  # Already simplified
    }
    
    # Handle version differences
    import inspect
    sig = inspect.signature(onnx2tf.convert)
    if "not_use_test_data" in sig.parameters:
        convert_kwargs["not_use_test_data"] = True
    
    onnx2tf.convert(**convert_kwargs)
    print("✓ TensorFlow SavedModel generated")
except Exception as e:
    print(f"❌ onnx2tf conversion failed: {e}")
    sys.exit(1)

# Find the actual SavedModel directory
sm_dir = "saved_model"
for root, dirs, files in os.walk(sm_dir):
    if "saved_model.pb" in files:
        sm_dir = root
        break

print("5️⃣ Converting SavedModel → TFLite...")
try:
    loaded = tf.saved_model.load(sm_dir)
    
    # Get concrete function with correct input signature
    if hasattr(loaded, 'signatures') and loaded.signatures:
        concrete_func = list(loaded.signatures.values())[0]
    else:
        # Fallback: create concrete function manually
        @tf.function
        def serve_fn(x):
            return loaded(x, training=False) if hasattr(loaded, '__call__') else loaded
        concrete_func = serve_fn.get_concrete_function(
            tf.TensorSpec(shape=[1, 256, 256, 3], dtype=tf.float32, name="input")
        )
    
    # Convert to FLOAT32 TFLite
    print("  🔄 FLOAT32...")
    converter_f32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_f32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_f32 = converter_f32.convert()
    fname_f32 = "RealESRGAN_x2plus_float32.tflite"
    with open(fname_f32, "wb") as f:
        f.write(tflite_f32)
    print(f"  ✓ {fname_f32} ({len(tflite_f32)/1024/1024:.2f} MB)")
    
    # Convert to FLOAT16 TFLite
    print("  🔄 FLOAT16...")
    converter_f16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter_f16.convert()
    fname_f16 = "RealESRGAN_x2plus_float16.tflite"
    with open(fname_f16, "wb") as f:
        f.write(tflite_f16)
    print(f"  ✓ {fname_f16} ({len(tflite_f16)/1024/1024:.2f} MB)")
    
except Exception as e:
    print(f"❌ TFLite conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("6️⃣ Verifying TFLite models...")
all_ok = True
for fname in [fname_f32, fname_f16]:
    if not os.path.exists(fname):
        print(f"❌ {fname} not found")
        all_ok = False
        continue
    try:
        interpreter = tf.lite.Interpreter(model_path=fname)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        
        # Test inference
        test_input = np.random.rand(*inp['shape']).astype(inp['dtype'])
        interpreter.set_tensor(inp['index'], test_input)
        interpreter.invoke()
        result = interpreter.get_tensor(out['index'])
        
        # Basic sanity check
        if np.all(result == 0) or np.any(np.isnan(result)):
            print(f"❌ {fname}: Invalid output")
            all_ok = False
        else:
            print(f"✓ {fname}: OK (output range: {result.min():.4f} ~ {result.max():.4f})")
    except Exception as e:
        print(f"❌ {fname} verification failed: {e}")
        all_ok = False

if all_ok:
    print("\n🎉 TFLite conversion SUCCESS!")
else:
    print("\n⚠️ Some TFLite models have issues")
    sys.exit(1)
