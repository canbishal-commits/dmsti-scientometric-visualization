import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import requests
from bs4 import BeautifulSoup

# -----------------------------------------
# Read researcher names from text file
# -----------------------------------------

with open("researchers.txt", "r", encoding="utf-8") as file:
    researchers = [line.strip() for line in file.readlines()]

# Total number of researchers
n = len(researchers)

# Dictionary to store publications
researcher_publications = {}

# -----------------------------------------
# Download publication data from eLABa
# -----------------------------------------

for researcher in researchers:

    print(f"Downloading publications for: {researcher}")

    # Replace spaces with "+"
    researcher_url = researcher.replace(" ", "+")

    # Create eLABa URL
    url = f"https://elaba.mb.vu.lt/dmsti/?aut={researcher_url}"

    # Download webpage
    response = requests.get(url)

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract plain text
    text = soup.get_text()

    # Split text into lines
    lines = text.splitlines()

    # Store publications
    publications = []

    for line in lines:

        line = line.strip()

        # Filter publication-like lines
        if 80 < len(line) < 250 and "." in line:

            # Avoid duplicate entries
            if line not in publications:
                publications.append(line)

    # Save publications for researcher
    researcher_publications[researcher] = publications

    print(f"{researcher}: {len(publications)} publications found")

# -----------------------------------------
# Create collaboration matrix
# -----------------------------------------

matrix = np.zeros((n, n), dtype=int)

for i in range(n):

    for j in range(n):

        # Skip same researcher comparison
        if i == j:
            continue

        researcher1 = researchers[i]
        researcher2 = researchers[j]

        publications1 = researcher_publications[researcher1]
        publications2 = researcher_publications[researcher2]

        shared_count = 0

        # Compare publication titles
        for pub1 in publications1:

            for pub2 in publications2:

                pub1_clean = pub1.lower().strip()
                pub2_clean = pub2.lower().strip()

                # Compare first 60 characters
                if pub1_clean[:60] == pub2_clean[:60]:

                    shared_count += 1
                    break

        # Save shared publication count
        matrix[i][j] = shared_count

# -----------------------------------------
# Display collaboration matrix
# -----------------------------------------

print("\nCollaboration Matrix:\n")

df_matrix = pd.DataFrame(
    matrix,
    index=researchers,
    columns=researchers
)

print(df_matrix)

# Export matrix to CSV
df_matrix.to_csv("collaboration_matrix.csv")

print("\ncollaboration_matrix.csv saved successfully.")

# -----------------------------------------
# Convert to dissimilarity matrix
# -----------------------------------------

dissimilarity_matrix = np.max(matrix) - matrix

# Set diagonal values to zero
np.fill_diagonal(dissimilarity_matrix, 0)

# -----------------------------------------
# PCA Visualization
# -----------------------------------------

pca = PCA(n_components=2)

coordinates = pca.fit_transform(dissimilarity_matrix)

# Convert full names to initials
def get_initials(name):

    parts = name.split()

    initials = ""

    for part in parts:
        initials += part[0].upper()

    return initials

# Create PCA plot
plt.figure(figsize=(12, 8))

for i, researcher in enumerate(researchers):

    x = coordinates[i, 0]
    y = coordinates[i, 1]

    plt.scatter(x, y, s=60)

    plt.text(
        x + 0.02,
        y + 0.02,
        get_initials(researcher),
        fontsize=6
    )

plt.title("DMSTI Researcher Visualization using PCA")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.grid(True)

plt.savefig("pca_visualization.png")

plt.show()

# -----------------------------------------
# SMACOF / MDS Visualization
# -----------------------------------------

mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42,
    normalized_stress="auto",
    n_init=4
)

mds_coordinates = mds.fit_transform(dissimilarity_matrix)

# Create MDS plot
plt.figure(figsize=(12, 8))

for i, researcher in enumerate(researchers):

    x = mds_coordinates[i, 0]
    y = mds_coordinates[i, 1]

    plt.scatter(x, y, s=60)

    plt.text(
        x + 0.02,
        y + 0.02,
        get_initials(researcher),
        fontsize=6
    )

plt.title("DMSTI Researcher Visualization using SMACOF / MDS")

plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")

plt.grid(True)

plt.savefig("mds_visualization.png")

plt.show()