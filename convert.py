import os
import sys
import subprocess
import shutil
import warnings
import inspect
import torch
import numpy as np
import tensorflow as tf

# Vá numpy.load cho phép pickle
_orig_np_load = np.load
def _safe_np_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_np_load(*args, **kwargs)
np.load = _safe_np_load
np.lib.npyio.load = _safe_np_load

# Import & bypass test data load
import onnx2tf
try:
    onnx2tf.utils.common_functions.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except:
    pass
try:
    onnx2tf.onnx2tf.download_test_image_data = lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)
except:
    pass

from basicsr.archs.rrdbnet_arch import RRDBNet
warnings.filterwarnings('ignore')

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"

# 1. Download & Load Model
if not os.path.exists(PTH_FILE):
    print("Downloading model...")
    subprocess.run([
        "wget", "-q", "--max-redirect=5",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "-O", PTH_FILE
    ], check=True)

print("1. Loading architecture...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# 2. Export & Simplify ONNX
print("2. Exporting & Simplifying ONNX...")
torch.onnx.export(
    model,
    torch.randn(1, 3, 256, 256),
    ONNX_FILE,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    verbose=False,
)
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE], check=True, capture_output=True)
print(f"ONNX simplified: {os.path.getsize(ONNX_SIM_FILE) / 1024 / 1024:.1f} MB")

# 3. ONNX -> SavedModel
print("3. Converting to TensorFlow SavedModel...")
if os.path.exists("saved_model"):
    shutil.rmtree("saved_model")

sig = inspect.signature(onnx2tf.convert)
kwargs = {
    "input_onnx_file_path": ONNX_SIM_FILE,
    "output_folder_path": "saved_model",
    "overwrite_input_shape": ["input:1,3,256,256"],
    "verbosity": "warn",
}
if "not_use_test_data" in sig.parameters:
    kwargs["not_use_test_data"] = True
if "output_saved_model" in sig.parameters:
    kwargs["output_saved_model"] = True
onnx2tf.convert(**kwargs)

# Tìm thư mục chứa saved_model.pb
sm_dir = "saved_model"
for root, _, files in os.walk(sm_dir):
    if "saved_model.pb" in files:
        sm_dir = root
        break
print(f"SavedModel ready at: {sm_dir}")

# 4. Convert TFLite — Float32 + Float16
print("4. Generating TFLite variants...")
loaded = tf.saved_model.load(sm_dir)

if loaded.signatures:
    concrete_func = next(iter(loaded.signatures.values()))
    input_shape = concrete_func.structured_input_signature[1][0].shape.as_list()
else:
    input_shape = [1, 256, 256, 3]
    concrete_func = tf.function(loaded.__call__).get_concrete_function(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    )
print(f"Detected TF input shape: {input_shape}")

# Float32
print("Processing FLOAT32...")
converter_32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_32 = converter_32.convert()
fname_32 = "RealESRGAN_x2plus_float32.tflite"
with open(fname_32, "wb") as f:
    f.write(tflite_32)
print(f"{fname_32} ({len(tflite_32) / 1024 / 1024:.2f} MB)")

# Float16
print("Processing FLOAT16...")
converter_16 = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
converter_16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_16.target_spec.supported_types = [tf.float16]
tflite_16 = converter_16.convert()
fname_16 = "RealESRGAN_x2plus_float16.tflite"
with open(fname_16, "wb") as f:
    f.write(tflite_16)
print(f"{fname_16} ({len(tflite_16) / 1024 / 1024:.2f} MB)")

# 5. Verify tất cả file TFLite
print("\n5. Verifying TFLite files...")

tflite_files = [fname for fname in [fname_32, fname_16] if os.path.exists(fname)]
all_ok = True

for fname in tflite_files:
    size_mb = os.path.getsize(fname) / 1024 / 1024
    print(f"\n--- {fname} ({size_mb:.2f} MB) ---")

    try:
        interpreter = tf.lite.Interpreter(model_path=fname)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        in_shape = input_details[0]['shape']
        in_dtype = input_details[0]['dtype']
        out_shape = output_details[0]['shape']
        out_dtype = output_details[0]['dtype']

        print(f"  Input:  {list(in_shape)}, dtype={in_dtype}")
        print(f"  Output: {list(out_shape)}, dtype={out_dtype}")

        # Check ops
        ops_used = set()
        for detail in interpreter._get_ops_details():
            ops_used.add(detail['op_name'])

        flex_ops = [op for op in ops_used if op.startswith('FLEX_') or op.startswith('CUSTOM_')]
        if flex_ops:
            print(f"  WARNING: {len(flex_ops)} unsupported ops: {flex_ops}")
            print(f"  -> Mobile TFLite KHÔNG hỗ trợ file này!")
            all_ok = False
        else:
            print(f"  Ops: {len(ops_used)} ops, all supported on mobile")

        # Test inference
        dummy = np.random.rand(*in_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        out_min = float(output.min())
        out_max = float(output.max())
        print(f"  Output range: {out_min:.6f} ~ {out_max:.6f}")

        if out_max == 0 and out_min == 0:
            print(f"  FAIL: Output toan 0 - model loi!")
            all_ok = False
        elif out_min < -2.0 or out_max > 5.0:
            print(f"  WARNING: Output range bat thuong")
        else:
            print(f"  OK: Output co gia tri thuc")

    except Exception as e:
        print(f"  FAIL: Khong the load/verify: {e}")
        all_ok = False

# 6. Summary
print("\n" + "=" * 50)
if all_ok and tflite_files:
    print("ALL PASSED - Model san sang su dung!")
    print(f"Files generated: {tflite_files}")
else:
    print("CO LOI - Kiem tra log o tren!")
    if not all_ok:
        sys.exit(1)

print("\nDone!")
