import numpy as np
import matplotlib.pyplot as plt

# Load the CSI data
csi_data = np.loadtxt('datasets/tch-prep/tch-csi-10-running-mean.csv',skiprows=1, delimiter=',',usecols=list(range(1, 64)),)

# Create a heatmap of the CSI data
plt.figure(figsize=(10, 10))
plt.imshow(csi_data, cmap='hot')
plt.colorbar()
plt.xlabel('Subcarrier')
plt.ylabel('Time')
plt.title('CSI Heatmap')
plt.show()