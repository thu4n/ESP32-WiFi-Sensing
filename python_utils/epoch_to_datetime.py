import pandas as pd
df = pd.read_csv("D:\\Wifi_Sensing\\esp32-wifi-sensing\\datasets\\prep\\mtd02.csv")
df['date_time'] = pd.to_datetime(df['epoch'], unit='s')
df.to_csv("D:\\Wifi_Sensing\\esp32-wifi-sensing\\datasets\\prep\\mtd02_prep4.csv", date_format='%H:%M:%S')