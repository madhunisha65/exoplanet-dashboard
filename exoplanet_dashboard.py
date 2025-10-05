# exoplanet_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import requests

st.set_page_config(page_title="Exoplanet Dashboard", layout="wide")

st.title("🌌 Exoplanet Discovery & Analysis Dashboard")
st.markdown("Analyze, cluster, and visualize exoplanets using NASA data.")

# ------------------- FETCH DATA -------------------
NASA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_bmassj,pl_radj,pl_orbper,st_teff,st_rad,st_mass+from+ps&format=csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(NASA_URL)
        st.success("✅ NASA data fetched successfully!")
    except:
        st.warning("⚠ Could not fetch NASA data. Using mock data.")
        # Mock dataset
        data = {
            "pl_name": ["Kepler-22b", "Kepler-69c", "Kepler-186f", "Kepler-62f", "Kepler-442b"],
            "pl_bmassj": [0.36, 0.41, 1.1, 0.12, 0.15],
            "pl_radj": [2.4, 1.7, 1.1, 1.4, 1.3],
            "pl_orbper": [289.9, 242.5, 130.0, 267.3, 112.3],
            "st_teff": [5518, 5638, 3788, 4925, 4402],
            "st_rad": [0.979, 0.93, 0.47, 0.64, 0.60],
            "st_mass": [0.97, 0.81, 0.54, 0.69, 0.61]
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# ------------------- SIDEBAR FILTERS -------------------
st.sidebar.header("Filters")

mass_range = st.sidebar.slider("Planet Mass (Jupiter Mass)", 
                               float(df['pl_bmassj'].min()), float(df['pl_bmassj'].max()), 
                               (float(df['pl_bmassj'].min()), float(df['pl_bmassj'].max())))

radius_range = st.sidebar.slider("Planet Radius (Jupiter Radius)", 
                                 float(df['pl_radj'].min()), float(df['pl_radj'].max()), 
                                 (float(df['pl_radj'].min()), float(df['pl_radj'].max())))

orbital_period_range = st.sidebar.slider("Orbital Period (days)", 
                                         float(df['pl_orbper'].min()), float(df['pl_orbper'].max()), 
                                         (float(df['pl_orbper'].min()), float(df['pl_orbper'].max())))

filtered_df = df[
    (df['pl_bmassj'] >= mass_range[0]) & (df['pl_bmassj'] <= mass_range[1]) &
    (df['pl_radj'] >= radius_range[0]) & (df['pl_radj'] <= radius_range[1]) &
    (df['pl_orbper'] >= orbital_period_range[0]) & (df['pl_orbper'] <= orbital_period_range[1])
]

st.subheader("Filtered Exoplanets")
st.dataframe(filtered_df)

# ------------------- KMEANS CLUSTERING -------------------
st.sidebar.header("Clustering Settings")
num_clusters = st.sidebar.slider("Number of clusters (KMeans)", 2, 5, 3)

features = ['pl_bmassj', 'pl_radj', 'pl_orbper']
X = filtered_df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_df['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)

# ------------------- VISUALIZATIONS -------------------
st.subheader("Exoplanet Cluster Scatter Plot")
fig_scatter = px.scatter(filtered_df, x='pl_radj', y='pl_bmassj', color='Cluster',
                         hover_data=['pl_name', 'pl_orbper'], 
                         labels={'pl_radj':'Radius (RJ)', 'pl_bmassj':'Mass (MJ)'}, 
                         title="Exoplanet Radius vs Mass")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Star Temperature Distribution")
fig_hist = px.histogram(filtered_df, x='st_teff', nbins=20, color='Cluster', 
                        labels={'st_teff':'Star Effective Temperature (K)'},
                        title="Star Temperatures by Cluster")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Planets", len(filtered_df))
col2.metric("Average Planet Mass (MJ)", round(filtered_df['pl_bmassj'].mean(), 2))
col3.metric("Average Planet Radius (RJ)", round(filtered_df['pl_radj'].mean(), 2))
