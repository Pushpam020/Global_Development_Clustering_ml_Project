import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Global Development Clustering",
    layout="wide"
)

st.title("🌍 Global Development Clustering")
st.markdown("""
This application clusters countries based on **economic, health, demographic, and digital development indicators**
using unsupervised machine learning techniques.
""")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("World_development_mesurement.xlsx")

df = load_data()
df_original = df.copy()

# -------------------------------------------------
# DATA PREPROCESSING
# -------------------------------------------------
# Convert to numeric
for col in df.columns:
    if col != "Country":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop fully missing columns
df = df.dropna(axis=1, how="all")

# Median imputation
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("⚙️ Clustering Controls")

k = st.sidebar.slider(
    "Number of Clusters (KMeans)",
    min_value=2,
    max_value=6,
    value=4
)

st.sidebar.caption("Recommended range: 3–5 clusters for interpretability")

# -------------------------------------------------
# SCALING
# -------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

# -------------------------------------------------
# KMEANS MODEL (FINAL MODEL)
# -------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_df)

# -------------------------------------------------
# PCA FOR VISUALIZATION
# -------------------------------------------------
pca_2 = PCA(n_components=2)
pca_data = pca_2.fit_transform(scaled_df)

explained_var = pca_2.explained_variance_ratio_

pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["Cluster"]
pca_df["Country"] = df["Country"]

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Overview",
    "🧠 Clustering Results",
    "📈 PCA Visualization"
])

# -------------------------------------------------
# TAB 1: DATASET OVERVIEW
# -------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_original.head(20))

    st.subheader("Statistical Summary")
    st.dataframe(df[num_cols].describe().T)

# -------------------------------------------------
# TAB 2: CLUSTERING RESULTS
# -------------------------------------------------
with tab2:
    st.subheader("Cluster-wise Distribution")
    st.bar_chart(df["Cluster"].value_counts().sort_index())

    st.subheader("Cluster Profiles (Mean Values)")
    cluster_profile = df.groupby("Cluster")[num_cols].mean()
    st.dataframe(cluster_profile)

    st.subheader("Cluster Interpretation")
    for cid in cluster_profile.index:
        st.markdown(
            f"""
            **Cluster {cid}**
            - Higher GDP & digital access → more developed
            - Lower birth & infant mortality → better health outcomes
            """
        )

    st.subheader("Explore a Country")
    country = st.selectbox(
        "Search Country",
        sorted(df["Country"].unique())
    )

    country_data = df[df["Country"] == country]
    st.write("**Assigned Cluster:**", int(country_data["Cluster"].values[0]))

    st.download_button(
        "⬇️ Download Clustered Data",
        data=df.to_csv(index=False),
        file_name="clustered_countries.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# TAB 3: PCA VISUALIZATION
# -------------------------------------------------
with tab3:
    st.subheader("PCA-based Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        ax=ax
    )
    ax.set_title("Country Clusters (PCA Projection)")
    st.pyplot(fig)

    st.caption(
        f"PC1 explains {explained_var[0]*100:.1f}% variance, "
        f"PC2 explains {explained_var[1]*100:.1f}% variance"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "📌 **Model:** KMeans | "
    "📊 **Scaling:** StandardScaler | "
    "📉 **Visualization:** PCA"
)
