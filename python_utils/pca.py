import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data from the CSV file
data = pd.read_csv('datasets/tch-prep/tch-csi-10-running-mean.csv',skiprows=1, usecols=list(range(1, 64)), index_col=None)

# Reshape the data into a 2D matrix, with each row representing a subcarrier and each column representing a data point.
#data_matrix = data.to_numpy().reshape((64, -1))

# Standardize the data
standardized_data = (data - data.mean()) / data.std()

# Create a PCA object
pca = PCA()

# Fit the PCA object to the data
pca.fit(standardized_data)

# Transform the data using the PCA object
transformed_data = pca.transform(standardized_data)

# Analyze the results
# For example, you can plot the transformed data to see how the different principal components are related to each other.
# Plot the transformed data
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Plot of Transformed Data")
plt.show()

# Export the transformed data to a CSV file
pd.DataFrame(transformed_data).to_csv('datasets/tch-prep/tch-pca.csv', index=False)