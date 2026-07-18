# PatrolIQ---Smart-Safety-Analytics-Platform

PatrolIQ is an unsupervised machine learning project that analyzes Chicago crime data to discover geographic hotspots, temporal crime patterns, and the features that contribute most strongly to those patterns.

The project combines data acquisition, cleaning, exploratory data analysis, feature engineering, clustering, dimensionality reduction, and MLflow experiment tracking to support data-informed public safety analysis.

---

## Project Overview

Urban crime datasets contain large volumes of geographic, temporal, administrative, and incident-level information. Raw records alone are difficult to convert into operational insights.

PatrolIQ addresses this problem by:

- identifying geographic crime concentration zones;
- discovering time-based incident patterns;
- comparing multiple clustering algorithms;
- reducing high-dimensional data using PCA and t-SNE;
- tracking model parameters and metrics using MLflow;
- preparing analytical outputs for an interactive public safety dashboard.

The project is designed as a decision-support and exploratory analytics system rather than an individual-level crime prediction tool.

---
## Dataset

**Source:** Chicago Data Portal — Crimes, 2001 to Present  
**Access method:** Socrata CSV API  
**Time coverage in the sampling notebook:** 2001–2025  
**Original variables:** 22  
**Crime categories found in the cleaned data:** 33

### Important source columns

| Category | Columns |
|---|---|
| Incident identification | `id`, `case_number`, `iucr`, `fbi_code` |
| Crime classification | `primary_type`, `description`, `location_description` |
| Time | `date`, `year`, `updated_on` |
| Geography | `block`, `latitude`, `longitude`, `x_coordinate`, `y_coordinate`, `location` |
| Administrative areas | `beat`, `district`, `ward`, `community_area` |
| Status | `arrest`, `domestic` |

### Dataset sizes in the current notebooks

| Processing stage | Shape |
|---|---:|
| Stratified source sample used by cleaning notebook | 510,000 rows × 22 columns |
| Cleaned dataset | 479,440 rows × 22 columns |
| EDA dataset after basic feature engineering | 479,440 rows × 28 columns |
| Final feature dataset | 479,440 rows × 75 columns |
| Clustering sample | 50,000 rows |

Monthly stratified sampling is used to improve coverage across years and months instead of selecting only a single continuous block of records.

---
## Repository Structure

A recommended GitHub structure is shown below:

```text
PatrolIQ/
│
├── src/
│   ├── data.ipynb
│   ├── Data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── mlflow.db
│   ├── Clustering_Analysis.ipynb
│   └── PCA.ipynb
│
├── model
│   ├── clustering.py
│   ├── feature.py
│   ├── pca.py
├── mlruns
├── data
│
├── requirements.txt
└── README.md
```
---

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SciPy
- MLflow
- Jupyter Notebook
- Chicago Data Portal

---
