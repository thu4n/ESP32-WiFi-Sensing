import pandas as pd
df = pd.read_csv("D:\\Wifi_Sensing\\esp32-csi-tool\\datasets\\mtd02.csv")
df['date_time'] = pd.to_datetime(df['epoch'], unit='s')
df.to_csv("D:\\Wifi_Sensing\\esp32-csi-tool\\datasets\\mtd02_prep1.csv", date_format='%H:%M:%S')