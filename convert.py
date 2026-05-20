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
    model, torch.randn(1, 3, 256, 256), ONNX_FILE,
    export_params=True, opset_version=14,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"],
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

sm_dir = "saved_model"
for root, _, files in os.walk(sm_dir):
    if "saved_model.pb" in files:
        sm_dir = root
        break
print(f"SavedModel ready at: {sm_dir}")

# 4. Convert TFLite
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
converter_16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter_16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_16.target_spec.supported_types = [tf.float16]
tflite_16 = converter_16.convert()
fname_16 = "RealESRGAN_x2plus_float16.tflite"
with open(fname_16, "wb") as f:
    f.write(tflite_16)
print(f"{fname_16} ({len(tflite_16) / 1024 / 1024:.2f} MB)")

# 5. Convert ONNX -> NCNN
print("5. Converting ONNX to NCNN...")
param_file = "esrgan.param"
bin_file = "esrgan.bin"

try:
    result = subprocess.run(
        ["onnx2ncnn", ONNX_SIM_FILE, param_file, bin_file],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        size_param = os.path.getsize(param_file) / 1024
        size_bin = os.path.getsize(bin_file) / 1024 / 1024
        print(f"NCNN: {param_file} ({size_param:.1f} KB), {bin_file} ({size_bin:.1f} MB)")
    else:
        print(f"onnx2ncnn stderr: {result.stderr[:500]}")
        # onnx2ncnn có thể warning nhưng vẫn tạo file
        if os.path.exists(param_file) and os.path.exists(bin_file):
            size_param = os.path.getsize(param_file) / 1024
            size_bin = os.path.getsize(bin_file) / 1024 / 1024
            print(f"NCNN (with warnings): {param_file} ({size_param:.1f} KB), {bin_file} ({size_bin:.1f} MB)")
        else:
            print("NCNN conversion FAILED - no output files")
except FileNotFoundError:
    print("onnx2ncnn not found! Make sure NCNN tools are installed.")
except Exception as e:
    print(f"NCNN conversion error: {e}")

# 6. Verify TFLite
print("\n6. Verifying TFLite files...")
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
        out_shape = output_details[0]['shape']

        print(f"  Input:  {list(in_shape)}, dtype={input_details[0]['dtype']}")
        print(f"  Output: {list(out_shape)}, dtype={output_details[0]['dtype']}")

        ops_used = set()
        for detail in interpreter._get_ops_details():
            ops_used.add(detail['op_name'])

        print(f"  ALL OPS ({len(ops_used)}): {sorted(ops_used)}")

        flex_ops = [op for op in ops_used if op.startswith('FLEX_') or op.startswith('CUSTOM_')]
        if flex_ops:
            print(f"  UNSUPPORTED: {flex_ops}")
            all_ok = False
        else:
            print(f"  All ops supported on mobile")

        dummy = np.random.rand(*in_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        out_min = float(output.min())
        out_max = float(output.max())
        print(f"  Output range: {out_min:.6f} ~ {out_max:.6f}")

        if out_max == 0 and out_min == 0:
            print(f"  FAIL: Output toan 0!")
            all_ok = False
        else:
            print(f"  OK: Output co gia tri thuc")

    except Exception as e:
        print(f"  FAIL: {e}")
        all_ok = False

# 7. Verify NCNN files exist
print("\n7. Verifying NCNN files...")
ncnn_ok = False
if os.path.exists(param_file) and os.path.exists(bin_file):
    size_param = os.path.getsize(param_file) / 1024
    size_bin = os.path.getsize(bin_file) / 1024 / 1024
    print(f"  {param_file}: {size_param:.1f} KB")
    print(f"  {bin_file}: {size_bin:.1f} MB")

    # Đọc số layer từ param file
    with open(param_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            layer_count = lines[1].strip()
            print(f"  Layers: {layer_count}")

    ncnn_ok = True
    print(f"  NCNN model ready")
else:
    print(f"  NCNN files NOT FOUND")

# 8. Summary
print("\n" + "=" * 50)
print("SUMMARY:")
print(f"  TFLite float32: {'OK' if os.path.exists(fname_32) else 'MISSING'}")
print(f"  TFLite float16: {'OK' if os.path.exists(fname_16) else 'MISSING'}")
print(f"  NCNN:           {'OK' if ncnn_ok else 'MISSING'}")

if all_ok and ncnn_ok:
    print("\nALL PASSED!")
elif all_ok:
    print("\nTFLite OK, NCNN FAILED - check logs above")
elif ncnn_ok:
    print("\nNCNN OK, TFLite FAILED")
else:
    print("\nBOTH FAILED!")
    sys.exit(1)

print("\nDone!")
