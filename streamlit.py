import streamlit as st

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import folium

from folium.plugins import HeatMap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="PatrolIQ Crime Analytics",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.title("🚓 PatrolIQ Dashboard")

@st.cache_data
def load_data():
    feature_df = pd.read_csv(r"C:\Users\SAKTHI\Desktop\myproject\PatrolIQ\data\feature_data\feature_dataset.csv")
    clean_df = pd.read_csv(r"C:\Users\SAKTHI\Desktop\myproject\PatrolIQ\data\eda_chicago_crime.csv")
    return feature_df, clean_df

feature_df, clean_df = load_data()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 2rem;'>
        <div style='font-size:2.5rem;'>🚓</div>       
        <div style='font-size:1.2rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.5px;'>EMIPredict AI</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>PatrolIQ Crime Analytics Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.sidebar.selectbox(
    "Navigation",
    [
        "Overview",
        "Crime Hotspots Map",
        "Temporal Crime Analysis",
        "Dimensionality Reduction (PCA)",
        "MLflow Model Tracking"
    ])  
    st.markdown("---")


def render_overview_sidebar():
    total_records = "500,000"
    num_features = "22"

    st.sidebar.markdown(f"""
    <div style='font-size:0.75rem; color:#475569; padding: 0.6rem;'>
    <b style='color:#94a3b8;'>Dataset</b><br>
    {total_records} records · {num_features} features<br><br>

    <b style='color:#94a3b8;'>Models</b><br>
    K-Means, DBSCAN, Hierarchical<br>
    PCA, t-SNE<br><br>

    <b style='color:#94a3b8;'>Tracking</b><br>
    MLflow Active
    </div>
    """, unsafe_allow_html=True)

if page == "Overview":

    render_overview_sidebar()

    st.title("PatrolIQ — Smart Safety Analytics Platform")
    st.caption("**Crime intelligence analyst at the Chicago Police Department -- 2000 - 2026**")


    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Dataset","500,000")
    col2.metric("Total Records",f"{len(clean_df):,}")
    col3.metric("ML Features",feature_df.shape[1])
    col4.metric("Clusters Identified","7")


    tabs = st.tabs([
        "🔍 Overview",
        "🎯 Objectives",
        "🧠 ML Methods",
        "📊 Dashboard"
    ])

    with tabs[0]:
        st.markdown("""
        ### 🔍 Overview
        PatrolIQ is an AI-driven crime intelligence platform designed to identify and analyze crime hotspots using advanced machine learning and data visualization techniques.

        **Key components:**
        - 📍 Geographic clustering with K-Means  
        - 📉 Dimensionality reduction using PCA  
        - ⏰ Temporal analysis of crime patterns  
        - 📊 Experiment tracking via MLflow  
        - 🌐 Interactive dashboard built with Streamlit  
        """)

    with tabs[1]:
        st.markdown("""
        ### 🎯 Objectives
        - Identify high-density crime hotspots  
        - Analyze time-based risk patterns  
        - Simplify high-dimensional data into interpretable insights  
        - Evaluate clustering performance using quantitative metrics  
        - Deliver a scalable, production-ready intelligence system  
        """)

    with tabs[2]:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("""
            **Clustering**
            | Algorithm | Use |
            |-----------|-----|
            | K-Means | Geographic hotspots |
            | DBSCAN  | Density + outliers |
            | Hierarchical | Zone hierarchy |
            """)
        with c2:
            st.markdown("""
            **Dimensionality reduction**
            | Method | Output |
            |--------|--------|
            | PCA    | Variance ranking |
            | t-SNE  | 2D cluster viz |
            """)
        with c3:
            st.markdown("""
            **Evaluation metrics**
            | Metric | Target |
            |--------|--------|
            | Silhouette | ≥ 0.5 |
            | Davies-Bouldin | Low |
            | KL divergence | Low |
            """)

    with tabs[3]:
        st.markdown("""
        ### 📊 Dashboard Sections
        Navigate through the application using the sidebar:

        1. **Geographic Hotspots** – Interactive crime cluster mapping  
        2. **Temporal Patterns** – Hourly and seasonal trend analysis  
        3. **PCA Visualization** – 2D representation of feature space  
        """)

# ------------------------------------------------------------
# HOTSPOT MAP (OPTIMIZED)
# ------------------------------------------------------------
if page == "Crime Hotspots Map":

    st.title("📍 Geographic Crime Hotspots")

    # -----------------------------
    # DATA CLEANING
    # -----------------------------
    map_df = clean_df.copy()

    map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
    map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')

    mask = (
        (map_df['latitude'] > 41.6) & (map_df['latitude'] < 42.1) &
        (map_df['longitude'] > -87.9) & (map_df['longitude'] < -87.5)
    )

    clean_map_df = map_df[mask].dropna(subset=['latitude', 'longitude'])

    clean_map_df = clean_map_df.sample(
        min(2000, len(clean_map_df)),
        random_state=42
    )

    if clean_map_df.empty:
        st.error("🚨 No valid Chicago coordinates found!")
    else:

        # -----------------------------
        # BASE MAP
        # -----------------------------
        m = folium.Map(
            location=[41.8781, -87.6298],
            zoom_start=11
        )

        # -----------------------------
        # 🔥 HEATMAP
        # -----------------------------
        heat_data = clean_map_df[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data, radius=12, blur=15).add_to(m)

        # -----------------------------
        # 🎯 CLUSTER POINTS
        # -----------------------------
        if "district" in clean_map_df.columns:

            colors = [
                "red", "blue", "green", "purple", "orange",
                "beige", "darkblue","darkgreen", "cadetblue",
                "white", "pink", "gray", "black",
            ]

            for _, row in clean_map_df.iterrows():

                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=2,
                    color=colors[int(row['district']) % len(colors)],
                    fill=True,
                    fill_opacity=0.6
                ).add_to(m)

        # -----------------------------
        # DISPLAY
        # -----------------------------
        st_folium(m, width=1200, height=600)

    # Statistics Bar

# ------------------------------------------------------------
# TEMPORAL ANALYSIS
# ------------------------------------------------------------
elif page == "Temporal Crime Analysis":

    st.title("⏰ Temporal Crime Patterns")
    st.caption("Analyzing how crime shifts across hours, days, and seasons.")
    
    df = clean_df.copy()
    
    # -----------------------------
    # ROW 1 (Hour + Day)
    # -----------------------------
    col1, col2 = st.columns(2)

    # 🔹 Crimes by Hour
    with col1:
        st.subheader("Crimes by Hour")

        crime_hour = df['hour'].value_counts().sort_index()

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=crime_hour.index, y=crime_hour.values, ax=ax1)

        ax1.set_title("Crimes by Hour")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Count")

        st.pyplot(fig1)

    # 🔹 Crimes by Day
    with col2:
        st.subheader("Crimes by Day of Week")

        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        crime_day = df['day_name'].value_counts().reindex(day_order)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=crime_day.index, y=crime_day.values, ax=ax2)

        ax2.set_title("Crimes by Day")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Count")

        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # -----------------------------
    # ROW 2 (Season + Weekend)
    # -----------------------------
    col3, col4 = st.columns(2)

    # 🔹 Crimes by Season
    with col3:
        st.subheader("Crimes by Season")

        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        crime_season = df["season"].value_counts().reindex(season_order)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=crime_season.index, y=crime_season.values, ax=ax3)

        ax3.set_title("Crimes by Season")
        ax3.set_xlabel("Season")
        ax3.set_ylabel("Count")

        st.pyplot(fig3)

    # 🔹 Weekend vs Weekday
    with col4:
        st.subheader("Weekend vs Weekday")

        weekend_map = {False: "Weekday", True: "Weekend"}
        weekend_counts = df["Weekend_crimes"].value_counts().sort_index()
        weekend_counts.index = weekend_counts.index.map(weekend_map)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=weekend_counts.index, y=weekend_counts.values, ax=ax4)

        ax4.set_title("Weekend vs Weekday Crimes")
        ax4.set_xlabel("Type")
        ax4.set_ylabel("Count")

        st.pyplot(fig4)

    # -----------------------------
    # OPTIONAL: SEASON DETAIL
    # -----------------------------
    st.subheader("📊 Seasonal Breakdown")

    fig5, ax5 = plt.subplots(figsize=(10, 4))

    sns.countplot(
        data=df,
        x='season',
        order=["Winter", "Spring", "Summer", "Autumn"],
        ax=ax5
    )

    ax5.set_title("Seasonal Crime Distribution")
    ax5.set_xlabel("Season")
    ax5.set_ylabel("Count")

    st.pyplot(fig5)

# ------------------------------------------------------------
# DIMENSIONALITY REDUCTION (PCA)
# ------------------------------------------------------------
elif page == "Dimensionality Reduction (PCA)":
    st.title("📉 Dimensionality Reduction Analysis")
    st.caption("Projecting high-dimensional crime features into 2D/3D space to identify hidden patterns.")

    # Sidebar Controls for PCA/t-SNE

    st.sidebar.subheader("Reduction Settings")
    method = st.sidebar.radio("Select Method", ["PCA (Fast)", "t-SNE (Deep Patterns)"])
    sample_size_pca = st.sidebar.slider("Sample Size", 1000, 10000, 5000)

    # 1. PRE-PROCESSING (Using your exact logic)
    # ---------------------------------------------------------
    drop_cols = ['id', 'case_number', 'date', 'block', 'iucr', 'primary_type', 
                 'description', 'location_description', 'updated_on', 'location']
    
    # Take sample and filter numeric features
    X_raw = feature_df.sample(sample_size_pca, random_state=42)
    X = X_raw.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    
    # PCA and t-SNE MUST be scaled to work correctly
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # 2. PCA LOGIC
    # ---------------------------------------------------------
    if method == "PCA (Fast)":


        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate Variance
        exp_var = pca.explained_variance_ratio_
        
        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("PC1 Variance", f"{exp_var[0]:.1%}")
        c2.metric("PC2 Variance", f"{exp_var[1]:.1%}")
        c3.metric("Total Explained", f"{np.sum(exp_var):.1%}")

        # Plot PCA
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        scatter = ax_pca.scatter(
            X_pca[:, 0], X_pca[:, 1], 
            c=X_raw['geo_cluster_kmeans'] if 'geo_cluster_kmeans' in X_raw.columns else None,
            cmap='viridis', s=10, alpha=0.6
        )
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        ax_pca.set_title("PCA: 2D Projection of Crime Features")
        if 'geo_cluster_kmeans' in X_raw.columns:
            plt.colorbar(scatter, label="K-Means Cluster")
        st.pyplot(fig_pca)


    # 3. t-SNE LOGIC
    # ---------------------------------------------------------
    else:
        st.info("t-SNE is calculating... this may take a moment for larger samples.")
        tsne = TSNE(
            n_components=2,
            perplexity=40,
            learning_rate='auto',
            random_state=42
        )
        X_tsne = tsne.fit_transform(X_scaled)

        fig_tsne, ax_tsne = plt.subplots(figsize=(10, 6))
        ax_tsne.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, alpha=0.6, color='#6366f1')
        ax_tsne.set_title("t-SNE: Non-linear Cluster Manifold")
        st.pyplot(fig_tsne)
        
        st.caption("t-SNE is better at finding 'islands' of similar crimes that PCA might miss.")



# ------------------------------------------------------------
# MLFLOW MODEL TRACKING (LIVE FETCH)
# ------------------------------------------------------------
elif page == "MLflow Model Tracking":

    import mlflow
    import pandas as pd

    st.title("🧪 MLflow Tracking")

    st.markdown("### ⚙️ MLflow Experiment Tracking")

    # -----------------------------
    # METRICS CARD
    # -----------------------------
    st.markdown("### 📊 Metrics Tracked")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        #### 🔹 Clustering Metrics
        - **Silhouette Score**
            - Higher is better (> 0.5)
        - **Davies-Bouldin Index**
            - Lower is better (< 1)
        - **Number of Clusters**
        """)

    with col4:
        st.markdown("""
        #### 🔹 Dimensionality Metrics
        - **PCA Variance**
            - Target: 70%+
        - **PC1 / PC2 Importance**
        - **t-SNE Sample Size**
        """)

    # -----------------------------
    # INSIGHTS CARD
    # -----------------------------
    st.markdown("### 🧠 Key Insights")

    st.success("""
    ✔ High Silhouette Score → Strong cluster separation  
    ✔ Low Davies-Bouldin → Well-separated clusters  
    ✔ PCA reduces noise before clustering  
    ✔ DBSCAN detects hotspots + outliers  
    """)    

    try:
        # ✅ Set tracking URI (local DB or server)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        # -----------------------------
        # LOAD EXPERIMENTS
        # -----------------------------
        experiments = mlflow.search_experiments()

        if not experiments:
            st.warning("No experiments found")
        else:
            exp_names = [exp.name for exp in experiments]

            selected_exp_name = st.selectbox("Select Experiment", exp_names)

            exp = mlflow.get_experiment_by_name(selected_exp_name)

            runs = mlflow.search_runs([exp.experiment_id])

            if runs.empty:
                st.warning("No runs in this experiment")
            else:
                # -----------------------------
                # 🔽 SELECT RUN ID
                # -----------------------------
                run_ids = runs["run_id"].tolist()
                selected_run = st.selectbox("Select Run ID", run_ids)

                run_data = runs[runs["run_id"] == selected_run].iloc[0]

                # -----------------------------
                # PARAMETERS
                # -----------------------------
                st.markdown("### ⚙️ Parameters")

                params = {
                    k.replace("params.", ""): v
                    for k, v in run_data.items()
                    if k.startswith("params.")
                }

                st.json(params)

                # -----------------------------
                # METRICS
                # -----------------------------
                st.markdown("### 📈 Metrics")

                metrics = {
                    k.replace("metrics.", ""): v
                    for k, v in run_data.items()
                    if k.startswith("metrics.")
                }

                st.json(metrics)

                # -----------------------------
                # 📊 HISTORY TABLE
                # -----------------------------
                st.markdown("### 📜 Recent Runs")

                cols = [
                    c for c in runs.columns
                    if "metrics." in c or "params." in c or c == "start_time"
                ]

                st.dataframe(runs[cols].head(10))

    except Exception as e:
        st.error(f"MLflow connection error: {str(e)}")

        st.markdown("#### Start MLflow server:")
        st.code("mlflow ui", language="bash")