import os
import sys
import subprocess
import warnings
import torch
import numpy as np
import tensorflow as tf

warnings.filterwarnings('ignore')

# Bypass numpy
_orig_np_load = np.load
def _safe_np_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_np_load(*args, **kwargs)
np.load = _safe_np_load
np.lib.npyio.load = _safe_np_load

from basicsr.archs.rrdbnet_arch import RRDBNet

PTH_FILE = "RealESRGAN_x2plus.pth"
ONNX_FILE = "model.onnx"
ONNX_SIM_FILE = "model_sim.onnx"

# 1. Download & Load
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

# 2. Export ONNX
print("2. Exporting ONNX...")
torch.onnx.export(
    model, torch.randn(1, 3, 256, 256), ONNX_FILE,
    export_params=True, opset_version=13,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"},
                  "output": {0: "batch", 2: "h", 3: "w"}},
    verbose=False,
)
subprocess.run([sys.executable, "-m", "onnxsim", ONNX_FILE, ONNX_SIM_FILE],
               check=True, capture_output=True)
print(f"ONNX: {os.path.getsize(ONNX_SIM_FILE) / 1024 / 1024:.1f} MB")

# 3. ONNX → SavedModel bằng tf2onnx
print("3. Converting ONNX → SavedModel (tf2onnx)...")
import tf2onnx
import onnx

onnx_model = onnx.load(ONNX_SIM_FILE)

# ★ Dùng tf2onnx trực tiếp — pipeline khác với onnx2tf
graph_def, inputs, outputs = tf2onnx.tf_loader.from_graphdef(None, None)
model_proto, _ = tf2onnx.convert.from_onnx(
    onnx_model,
    input_names=["input:0"],
    output_names=["output:0"],
    opset=13,
)

# SavedModel từ TF
concrete_func = tf.function(
    lambda x: tf.import_graph_def(
        tf.compat.v1.graph_util.convert_variables_to_constants(
            tf.compat.v1.Session(graph=tf.Graph()).graph.as_default().__enter__(),
            [],
            []
        ),
        name=""
    )
)

# ★ Cách đơn giản hơn — dùng tflite convert trực tiếp từ ONNX
print("3. Converting ONNX → TFLite (direct)...")

# Load onnx bằng onnxruntime → lấy input/output info
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_SIM_FILE)
input_info = sess.get_inputs()[0]
input_name = input_info.name
input_shape = [1, 3, 256, 256]  # NCHW

print(f"  Input: {input_info.shape}")

# ★ Dùng tf2onnx convert trực tiếp sang TFLite
tflite_path_float32 = "RealESRGAN_x2plus_float32.tflite"
tflite_path_float16 = "RealESRGAN_x2plus_float16.tflite"

# Convert qua TF SavedModel
print("  Creating SavedModel...")
dummy = np.random.rand(1, 3, 256, 256).astype(np.float32)

# Dùng onnx-tf hoặc onnx2tf nhưng chỉ SavedModel
if os.path.exists("saved_model"):
    import shutil
    shutil.rmtree("saved_model")

# ★ Thử onnx2tf nhưng chỉ lấy SavedModel
import onnx2tf
import inspect as _inspect

sig = _inspect.signature(onnx2tf.convert)
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
print(f"  SavedModel at: {sm_dir}")

# ★ Convert TFLite — dùng converter gốc TF, KHÔNG dùng onnx2tf converter
loaded = tf.saved_model.load(sm_dir)
if loaded.signatures:
    concrete_func = next(iter(loaded.signatures.values()))
    in_shape = concrete_func.structured_input_signature[1][0].shape.as_list()
else:
    in_shape = [1, 256, 256, 3]
    concrete_func = tf.function(loaded.__call__).get_concrete_function(
        tf.TensorSpec(shape=in_shape, dtype=tf.float32)
    )

print(f"  TF input shape: {in_shape}")

# Float32
print("4a. Float32...")
conv32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite32 = conv32.convert()
with open(tflite_path_float32, "wb") as f:
    f.write(tflite32)
print(f"  {tflite_path_float32}: {len(tflite32)/1024/1024:.2f} MB")

# Float16
print("4b. Float16...")
conv16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
conv16.optimizations = [tf.lite.Optimize.DEFAULT]
conv16.target_spec.supported_types = [tf.float16]
tflite16 = conv16.convert()
with open(tflite_path_float16, "wb") as f:
    f.write(tflite16)
print(f"  {tflite_path_float16}: {len(tflite16)/1024/1024:.2f} MB")

# 5. Verify — ★ hiển thị TÊN từng ops để debug
print("\n5. Verifying...")
all_ok = True

for fname in [tflite_path_float32, tflite_path_float16]:
    if not os.path.exists(fname):
        continue
    size_mb = os.path.getsize(fname) / 1024 / 1024
    print(f"\n--- {fname} ({size_mb:.2f} MB) ---")

    try:
        interp = tf.lite.Interpreter(model_path=fname)
        interp.allocate_tensors()

        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print(f"  Input:  {list(inp['shape'])}, dtype={inp['dtype']}")
        print(f"  Output: {list(out['shape'])}, dtype={out['dtype']}")

        # ★ In TẤT CẢ ops tên — để biết chính xác ops nào có
        ops = set()
        for d in interp._get_ops_details():
            ops.add(d['op_name'])
        print(f"  ALL ops ({len(ops)}): {sorted(ops)}")

        # Check unsupported
        flex = [o for o in ops if o.startswith('FLEX_') or o.startswith('CUSTOM_')]
        if flex:
            print(f"  UNSUPPORTED: {flex}")
            all_ok = False
        else:
            print(f"  All ops OK for mobile")

        # Test inference
        dummy = np.random.rand(*inp['shape']).astype(np.float32)
        interp.set_tensor(inp['index'], dummy)
        interp.invoke()
        out_arr = interp.get_tensor(out['index'])
        print(f"  Range: {out_arr.min():.6f} ~ {out_arr.max():.6f}")

        if out_arr.max() == 0 and out_arr.min() == 0:
            print(f"  FAIL: Output toan 0!")
            all_ok = False
        else:
            print(f"  OK")

    except Exception as e:
        print(f"  FAIL: {e}")
        all_ok = False

print(f"\n{'ALL PASSED' if all_ok else 'CO LOI'}!")
if not all_ok:
    sys.exit(1)
