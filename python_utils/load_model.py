# Import necessary libraries
import tensorrt as trt

# Set model and engine paths
onnx_model_path = "your_model.onnx"
engine_path = "your_model.trt"

# Create inference engine
logger = trt.Logger()
trt_engine = trt.init_inference_engine(None, logger)

# Read ONNX model directly
with open(onnx_model_path, "rb") as f:
    model_data = f.read()

# Convert ONNX model to TensorRT engine
config = trt.BuilderConfig()
config.max_workspace_size = 1<<20  # Set appropriate workspace size
profile = trt.BuilderProfile()
profile.int8 = True  # Enable INT8 for better performance (optional)
config.profile = profile
trt_engine.convert_onnx_model(model_data, config)

# Save the TRT engine
with open(engine_path, "wb") as f:
    f.write(trt_engine.serialize())

# ... Further code for loading and running the engine ...
