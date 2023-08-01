import pandas as pd
activities = ["NA","LA","RA","LL","RL","JJ","SO"]
df = pd.read_csv("D:\\Wifi_Sensing\\esp32-wifi-sensing\\datasets\\prep\\mtd02.csv")
for i in df.index:
    print(" ")