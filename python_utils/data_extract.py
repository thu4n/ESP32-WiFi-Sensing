import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import collections
import signal

#from hampel import hampel
import pandas as pd
#import tensorrt as trt
#import common

from read_stdin import readline, print_until_first_csi_line

packet_count = 0

# Deque definition
perm_amp = collections.deque(maxlen=200)
perm_phase = collections.deque(maxlen=200)

# Function to load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def predict_with_engine(engine, data):
    # Assuming your engine has a single input and a single output
    input_shape = (200, 54, 1)  # Adjust based on your input shape

    # Allocate device memory for inputs and outputs
    d_input, h_input, d_output, h_output = common.allocate_buffers(engine)

    # Copy input data to host
    np.copyto(h_input, data.ravel())

    # Copy input data to device
    cuda.memcpy_htod(d_input, h_input)

    # Execute the model
    with engine.create_execution_context() as context:
        context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])

    # Copy output data to host
    cuda.memcpy_dtoh(h_output, d_output)

    # Return the prediction
    return h_output

def process(res):
    # Parser
    all_data = res.split(',')
    csi_data = all_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")

    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]
    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    csi_size = len(csi_data)
    amplitudes = []
    phases = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            phase_calc = math.atan2(imaginary[j], real[j])
            amplitudes.append(amplitude_calc)
            phases.append(phase_calc)

        perm_phase.append(phases)
        perm_amp.append(amplitudes)

def filter(amplitude_df):
    amp_np = amplitude_df.to_numpy().T
    filtered_data = []
    savgol_filtered = []
    for i in range(0, 64):
  # lọc dữ liệu dùng hampel
        result = hampel(amp_np[i], window_size=3, n_sigma=5.0)
        filtered_data.append(result.filtered_data)
        savgol_filtered.append(savgol_filter(result.filtered_data,window_length=5, polyorder=3))
    displacement = np.transpose(savgol_filtered)
    return displacement

def predict(data):
    normalized_data = data / 255.0
    prediction = model.predict(normalized_data)
    print(prediction)

def main():
    engine_path = "your_model.trt"  # Replace with your engine path
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime()
        engine = runtime.deserialize_engine(f)
        context = engine.create_inference_context()
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)  # Adjust for your model input
        inputs, outputs, bindings = context.get_bindings()
        bindings[0].data = input_data
        context.execute(bindings)


    try:
        while True:
            line = readline()
            if "CSI_DATA" in line:
                process(line)
                if(len(perm_amp) >= 200):
                    print("Chunk", perm_amp[-1])
                    #deque_list = list(perm_amp)
                    perm_amp.clear
                    #amp_df = pd.Dataframe(deque_list)
                    #pred_data = filter(amp_df)
                    #predict(pred_data)

    except KeyboardInterrupt:
        pass  # Handle Ctrl+C to gracefully exit

if __name__ == "__main__":
    main()

