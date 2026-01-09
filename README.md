
# 🌍 Global Development Clustering – Streamlit App

## Overview
This project clusters countries based on global development indicators such as
economic performance, health outcomes, demographic structure, and digital access.
It uses unsupervised learning techniques and provides an interactive Streamlit application
for exploration and visualization.

## Features
- Data cleaning and median imputation
- Feature scaling using StandardScaler
- KMeans clustering with adjustable number of clusters
- PCA-based 2D visualization of clusters
- Country-level cluster exploration

## Project Structure
```
global_clustering_app/
├── app.py
├── requirements.txt
├── World_development_mesurement.xlsx
└── README.md
```

## How to Run Locally
1. Place `World_development_mesurement.xlsx` in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment (Streamlit Cloud)
1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud
3. Create a new app and select `app.py`.
4. Deploy.

## Model Details
- Scaling: StandardScaler
- Clustering: KMeans
- Visualization: PCA (2 components)

## License
This project is for educational and demonstration purposes.
