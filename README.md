# DMSTI Scientometric Visualization

This project visualizes collaboration between VU MIF DMSTI researchers using publication data from eLABa.

## Methods Used

- Python
- NumPy
- Pandas
- Matplotlib
- PCA (Principal Component Analysis)
- MDS / SMACOF visualization

## How It Works

1. Researcher names are read from `researchers.txt`
2. Publication data is downloaded from eLABa
3. A collaboration matrix is created based on shared publications
4. Publication titles are compared using partial string matching
5. The collaboration matrix is transformed into a dissimilarity matrix
6. Researchers are visualized on a 2D plane using:
   - PCA
   - SMACOF / MDS

Researchers with more shared publications appear closer together.

## Files

- `main.py` = main program
- `researchers.txt` = researcher names
- `collaboration_matrix.csv` = generated collaboration matrix
- `pca_visualization.png` = PCA visualization
- `mds_visualization.png` = MDS visualization

## Running the Program

Install required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn requests beautifulsoup4