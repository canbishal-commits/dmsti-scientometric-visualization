import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read researcher names
with open("researchers.txt", "r", encoding="utf-8") as file:
    researchers = [line.strip() for line in file.readlines()]

# Number of researchers
n = len(researchers)

# Create random collaboration matrix
np.random.seed(42)

matrix = np.random.randint(0, 6, size=(n, n))

# Make matrix symmetric
matrix = (matrix + matrix.T) // 2

# Set diagonal to 0
np.fill_diagonal(matrix, 0)

# Convert collaboration matrix to dissimilarity matrix
dissimilarity_matrix = 1 / (matrix + 1)

# Apply PCA
pca = PCA(n_components=2)

coordinates = pca.fit_transform(dissimilarity_matrix)

# Plot
plt.figure(figsize=(12, 8))

for i, researcher in enumerate(researchers):

    x = coordinates[i, 0]
    y = coordinates[i, 1]

    plt.scatter(x, y)

    plt.text(x, y, researcher, fontsize=8)

plt.title("DMSTI Researcher Visualization using PCA")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.grid(True)

plt.show()