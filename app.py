
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Global Development Clustering", layout="wide")

st.title("🌍 Global Development Clustering")
st.markdown(
    "This application clusters countries based on economic, health, demographic, "
    "and digital development indicators using unsupervised machine learning."
)

@st.cache_data
def load_data():
    return pd.read_excel("World_development_mesurement.xlsx")

df = load_data()
df_original = df.copy()

# Convert to numeric
for col in df.columns:
    if col != "Country":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop fully missing columns
df = df.dropna(axis=1, how="all")

# Median imputation
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

# Sidebar
st.sidebar.header("⚙️ Controls")
k = st.sidebar.slider("Number of Clusters (KMeans)", 2, 8, 4)

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_df)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["Cluster"]
pca_df["Country"] = df["Country"]

tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "🧠 Clustering Results", "📈 PCA Visualization"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_original.head(20))
    st.subheader("Key Statistics")
    st.dataframe(df[num_cols].describe().T)

with tab2:
    st.subheader("Cluster-wise Distribution")
    st.bar_chart(df["Cluster"].value_counts().sort_index())
    st.subheader("Cluster Profiles (Mean Values)")
    st.dataframe(df.groupby("Cluster")[num_cols].mean())
    st.subheader("Explore a Country")
    country = st.selectbox("Select Country", sorted(df["Country"].unique()))
    st.write("Assigned Cluster:", int(df[df["Country"] == country]["Cluster"].values[0]))

with tab3:
    st.subheader("PCA-based Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Country Clusters (PCA Projection)")
    st.pyplot(fig)

st.markdown("---")
st.markdown("Model: KMeans | Scaling: StandardScaler | Visualization: PCA")
