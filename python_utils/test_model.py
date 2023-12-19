import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# Function to load TensorRT engine from a '.engine' file
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# Function to allocate device memory and copy data to the device
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output

# Function to perform inference with TensorRT engine
def inference(engine, h_input, d_input, h_output, d_output):
    stream = cuda.Stream()

    # Set the input data (dummy data in this example)
    np.random.seed(123)
    input_data = np.random.rand(*h_input.shape).astype(np.float32)
    np.copyto(h_input, input_data.ravel())

    # Copy input data to the device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    with engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # Copy output data to the host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output

if __name__ == '__main__':
    # Path to the TensorRT engine file
    engine_file_path = 'your_model.engine'

    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    engine = load_engine(engine_file_path)

    # Allocate buffers for input and output
    h_input, d_input, h_output, d_output = allocate_buffers(engine)

    # Perform inference
    output_result = inference(engine, h_input, d_input, h_output, d_output)

    # Print the inference result
    print("Inference Result:")
    print(output_result)
