import numpy as np
import matplotlib.pyplot as plt

def draw_heatmap(csi_data, x_label, y_label, title):
  """Draws a heatmap of the given CSI data.

  Args:
    csi_data: A numpy array containing the CSI data.
    x_label: The label for the x-axis of the heatmap.
    y_label: The label for the y-axis of the heatmap.
    title: The title of the heatmap.
  """

  # Create a heatmap object.
  heatmap = plt.pcolormesh(csi_data)

  # Set the x and y labels.
  plt.xlabel(x_label)
  plt.ylabel(y_label)

  # Set the title.
  plt.title(title)

  # Show the heatmap.
  plt.colorbar()
  plt.show()

# Load the CSI data into a numpy array.
csi_data = np.loadtxt("datasets/tch-prep/tch-pca-2.csv", delimiter=",")

# Draw the heatmap.
draw_heatmap(csi_data, "X-axis", "Y-axis", "CSI Data Heatmap")
