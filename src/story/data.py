"""Carga de los artefactos ya generados por el pipeline (data/, models/).

No reentrena nada: solo lee los CSV/joblib que produce `src/train.py`.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.evaluate import anomaly_profile, compare_models, crosstab_models, load_results

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

FEATURES_PATH = DATA_DIR / "customer_features.csv"
FEATURES_MODEL_PATH = DATA_DIR / "customer_features_model.csv"
PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
DBSCAN_TRIALS_PATH = MODELS_DIR / "dbscan_trials.csv"
ISOF_TRIALS_PATH = MODELS_DIR / "isof_trials.csv"
ISOF_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"


@st.cache_data
def load_customer_features():
    return pd.read_csv(FEATURES_PATH, index_col="CustomerID")


@st.cache_data
def load_features_model():
    return pd.read_csv(FEATURES_MODEL_PATH, index_col="CustomerID")


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    return load_results(str(FEATURES_MODEL_PATH), str(PREDICTIONS_PATH))


@st.cache_data
def load_best_params():
    import json

    if not BEST_PARAMS_PATH.exists():
        return None
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_trials(model_key: str):
    path = DBSCAN_TRIALS_PATH if model_key == "dbscan" else ISOF_TRIALS_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path).sort_values("number")
    df["best_so_far"] = df["value"].cummax()
    return df


@st.cache_resource
def load_isof_model():
    if not ISOF_MODEL_PATH.exists():
        return None
    return joblib.load(ISOF_MODEL_PATH)


@st.cache_data
def compute_pca(df_final: pd.DataFrame):
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(df_final)
    df_plot = pd.DataFrame(componentes, columns=["PC1", "PC2"], index=df_final.index)
    return df_plot, pca.explained_variance_ratio_.sum()


@st.cache_data
def get_compare_models(df_final: pd.DataFrame, df_resultados: pd.DataFrame):
    return compare_models(df_final, df_resultados)


@st.cache_data
def get_anomaly_profile(df_resultados: pd.DataFrame, anomaly_col: str):
    return anomaly_profile(df_resultados, anomaly_col)


@st.cache_data
def get_crosstab(df_resultados: pd.DataFrame):
    return crosstab_models(df_resultados)
