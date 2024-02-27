import serial
import math
import numpy as np
import pandas as pd

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import paho.mqtt.client as paho
from paho import mqtt

# Function to load TensorRT engine from a '.engine' file
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output

def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

# with this callback you can see if your publish was successful
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

def pub_mqtt(num):
    client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
    client.on_connect = on_connect

# enable TLS for secure connection
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
# set username and password
    client.username_pw_set("JetsonNano", "JetsonNano123")
# connect to HiveMQ Cloud on port 8883 (default for MQTT)
    client.connect("1904448cf01a4564947dae8e889f5fee.s2.eu.hivemq.cloud", 8883)
    client.loop_start()

    client.on_publish = on_publish
    print("MQTT called",num)
    client.publish("predict", payload==str(num), qos=1)

    client.loop_stop()
    client.disconnect()

def inference(engine, h_input, d_input, h_output, d_output, input_data):
    stream = cuda.Stream()
    input_data = np.array(input_data)
    input_data /= 255
    # Set the input data (dummy data in this example)
    np.random.seed(123)
    #input_data = np.random.rand(*h_input.shape).astype(np.float32)
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

def process_data(data):
    # Add your data processing logic here
    print(data)

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
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            amplitudes.append(amplitude_calc)
    df = pd.DataFrame(amplitudes)
    return df

def predict(df):
    columns_to_drop = [2,3,4,5,32,59,60,61,62]
    df.drop(df.columns[columns_to_drop], axis=1,inplace=True)
    #df = df.transpose().values.reshape((200, 55, 1))
    output_result = inference(engine, h_input, d_input, h_output, d_output,df)
    predicted_class = np.argmax(output_result)

    print("Inference Result:")
    print(predicted_class)
    pub_mqtt(predicted_class)

def main():
    serial_port = "/dev/ttyUSB0"  # Use the specified serial port
    baud_rate = 921600  # Set the baud rate to 921600
    count = 0
    # Configure the serial port
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1)
    dfs = []

    engine_file_path = '3act_cnn.engine'
    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    engine = load_engine(engine_file_path)
    h_input, d_input, h_output, d_output = allocate_buffers(engine)
    try:
        while True:
            try:
                data = ser.readline().decode("utf-8").strip()
                if "CSI_DATA" in data:
                    df = process(data)
                    df_transposed = df.transpose() 
                    #print(df_transposed.shape)
                    if df_transposed.shape[1] == 64:
                        # Append the DataFrame to the list
                        dfs.append(df_transposed)
                        count += 1
                    if(count == 200):
                        #print("Chunk", len(perm_amp))
                        result_df = pd.concat(dfs, axis=0)
                        #result_df = result_df.reset_index(drop=True)
                        print(result_df.shape)
                        dfs = []
                        count = 0
                        predict(result_df)
                        
            except Exception as e:
                print("Error:",{e})
                pass

    except KeyboardInterrupt:
        print("Exiting gracefully.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
