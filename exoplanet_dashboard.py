# exoplanet_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Exoplanet Discovery Dashboard",
    layout="wide"
)

st.title("Exoplanet Discovery & Analysis Dashboard")
st.markdown("Analyze, cluster, and visualize exoplanets using NASA data (mock dataset).")

# ------------------- MOCK DATA -------------------
# Mock NASA exoplanet dataset
data = {
    "Planet Name": ["Kepler-22b", "Kepler-69c", "Kepler-452b", "HD 209458 b", "Proxima Centauri b"],
    "Radius (Earth radii)": [2.4, 1.7, 1.6, 13.8, 1.1],
    "Mass (Earth masses)": [36, 7, 5, 220, 1.3],
    "Orbital Period (days)": [289, 242, 385, 3.5, 11.2],
    "Distance (ly)": [600, 2700, 1400, 150, 4.24],
}

df = pd.DataFrame(data)

# ------------------- CLUSTERING -------------------
st.subheader("Exoplanet Clustering")
st.markdown("Cluster exoplanets based on radius, mass, and orbital period.")

X = df[["Radius (Earth radii)", "Mass (Earth masses)", "Orbital Period (days)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)

cluster_mapping = {
    "0": "Small Rocky",
    "1": "Gas Giant",
    "2": "Medium/Neptune-like"
}
df['Exoplanet Type (ML)'] = df['Cluster'].map(cluster_mapping)

st.dataframe(df)

# ------------------- VISUALIZATION -------------------
st.subheader("Exoplanet Visualizations")

fig1 = px.scatter_3d(
    df,
    x="Radius (Earth radii)",
    y="Mass (Earth masses)",
    z="Orbital Period (days)",
    color="Exoplanet Type (ML)",
    hover_name="Planet Name",
    size="Radius (Earth radii)",
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    df,
    x="Planet Name",
    y="Distance (ly)",
    color="Exoplanet Type (ML)",
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------- INFO -------------------
st.subheader("About")
st.markdown(
    """
    This dashboard uses **mock NASA exoplanet data** to demonstrate:
    - Clustering planets by physical characteristics
    - 3D scatter plot and bar chart visualizations
    - Machine learning-based classification
    """
)
