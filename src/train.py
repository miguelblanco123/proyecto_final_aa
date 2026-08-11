import json

import joblib
import numpy as np
import pandas as pd
from optuna import create_study, samplers
from optuna import logging as optuna_logging
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


def objective_dbscan(trial, df_final):
    params_dbsc = {
        "eps": trial.suggest_float("eps", low=0.1, high=3.0),
        "min_samples": trial.suggest_int("min_samples", low=2, high=50),
    }

    labels_trial = DBSCAN(**params_dbsc).fit_predict(df_final)

    n_clusters = len(set(labels_trial)) - (1 if -1 in labels_trial else 0)
    if n_clusters < 2:
        return -1

    return silhouette_score(df_final, labels_trial)


def objective_isoforest(trial, df_final):
    params_isof = {
        "n_estimators": trial.suggest_int("n_estimators", low=50, high=300),
        "max_samples": trial.suggest_float("max_samples", low=0.1, high=1.0),
        "contamination": trial.suggest_float("contamination", low=0.01, high=0.4),
        "max_features": trial.suggest_float("max_features", low=0.5, high=1.0),
    }

    labels_trial = IsolationForest(random_state=42, **params_isof).fit_predict(df_final)

    n_clusters = len(set(labels_trial))
    if n_clusters < 2:
        return -1

    return silhouette_score(df_final, labels_trial)


def tune_dbscan(df_final, n_trials=20, seed=42):
    sampler = samplers.TPESampler(seed=seed)
    study = create_study(direction="maximize", study_name="DBSC_study", sampler=sampler)

    optuna_logging.set_verbosity(optuna_logging.WARNING)
    study.optimize(lambda trial: objective_dbscan(trial, df_final), n_trials=n_trials)
    optuna_logging.set_verbosity(optuna_logging.INFO)

    return study


def tune_isolation_forest(df_final, n_trials=20, seed=42):
    sampler = samplers.TPESampler(seed=seed)
    study = create_study(direction="maximize", study_name="ISOF_study", sampler=sampler)

    optuna_logging.set_verbosity(optuna_logging.WARNING)
    study.optimize(lambda trial: objective_isoforest(trial, df_final), n_trials=n_trials)
    optuna_logging.set_verbosity(optuna_logging.INFO)

    return study


def train_dbscan(df_final, params):
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(df_final)
    return dbscan, labels


def train_isolation_forest(df_final, params):
    isoforest = IsolationForest(random_state=42, **params)
    labels = isoforest.fit_predict(df_final)
    return isoforest, labels


def save_artifacts(
    isoforest,
    best_params_dbscan,
    best_params_isof,
    df_predictions,
    study_dbsc=None,
    study_isof=None,
    models_dir="models",
    predictions_path="data/processed/model_predictions.csv",
):
    # DBSCAN es transductivo (fit_predict, sin predict independiente), asi que solo
    # persistimos sus etiquetas; Isolation Forest si expone predict() sobre datos nuevos
    joblib.dump(isoforest, f"{models_dir}/isolation_forest.joblib")

    with open(f"{models_dir}/best_params.json", "w") as f:
        json.dump({"dbscan": best_params_dbscan, "isolation_forest": best_params_isof}, f, indent=2)

    df_predictions.to_csv(predictions_path, encoding="utf-8")

    # Historial de trials de Optuna, para poder mostrar el progreso de la
    # optimizacion (estado de entrenamiento) sin tener que reentrenar
    if study_dbsc is not None:
        study_dbsc.trials_dataframe().to_csv(f"{models_dir}/dbscan_trials.csv", index=False)
    if study_isof is not None:
        study_isof.trials_dataframe().to_csv(f"{models_dir}/isof_trials.csv", index=False)


def main(
    input_path="data/processed/customer_features_model.csv",
    models_dir="models",
    predictions_path="data/processed/model_predictions.csv",
    n_trials=20,
):
    df_final = pd.read_csv(input_path, index_col="CustomerID")

    study_dbsc = tune_dbscan(df_final, n_trials=n_trials)
    dbscan, labels_dbscan = train_dbscan(df_final, study_dbsc.best_params)

    study_isof = tune_isolation_forest(df_final, n_trials=n_trials)
    isoforest, labels_isof = train_isolation_forest(df_final, study_isof.best_params)

    df_predictions = pd.DataFrame(index=df_final.index)
    df_predictions["label_dbscan"] = labels_dbscan
    df_predictions["anomalia_dbscan"] = np.where(labels_dbscan == -1, 1, 0)
    df_predictions["label_isof"] = labels_isof
    df_predictions["anomalia_isof"] = np.where(labels_isof == -1, 1, 0)

    save_artifacts(
        isoforest,
        study_dbsc.best_params,
        study_isof.best_params,
        df_predictions,
        study_dbsc=study_dbsc,
        study_isof=study_isof,
        models_dir=models_dir,
        predictions_path=predictions_path,
    )

    return df_predictions


if __name__ == "__main__":
    main()
