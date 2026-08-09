import pandas as pd
from sklearn.metrics import silhouette_score

COLS_PERFIL = [
    "Permanencia", "Compras", "Canasta_Prom", "Ticket_Prom",
    "Precio_Prom", "Precio_Max", "Productos Distintos", "Pct_Devoluciones",
]


def load_results(
    features_path="data/processed/customer_features_model.csv",
    predictions_path="data/processed/model_predictions.csv",
):
    df_final = pd.read_csv(features_path, index_col="CustomerID")
    df_pred = pd.read_csv(predictions_path, index_col="CustomerID")
    return df_final.join(df_pred)


def anomaly_profile(df_resultados, anomaly_col):
    perfil = df_resultados.groupby(anomaly_col)[COLS_PERFIL].mean().round(2)
    perfil.index = ["Normal (0)", "Anomalia (1)"]
    return perfil


def compare_models(df_final, df_resultados):
    sil_dbscan = silhouette_score(df_final, df_resultados["label_dbscan"])
    sil_isof = silhouette_score(df_final, df_resultados["label_isof"])

    return pd.DataFrame({
        "Silhouette Score": [sil_dbscan, sil_isof],
        "Anomalias Detectadas": [df_resultados["anomalia_dbscan"].sum(), df_resultados["anomalia_isof"].sum()],
        "Porcentaje Anomalias": [df_resultados["anomalia_dbscan"].mean() * 100, df_resultados["anomalia_isof"].mean() * 100],
    }, index=["DBSCAN", "Isolation Forest"]).round(2)


def crosstab_models(df_resultados):
    return pd.crosstab(
        df_resultados["anomalia_dbscan"].map({0: "Normal", 1: "Anomalia"}),
        df_resultados["anomalia_isof"].map({0: "Normal", 1: "Anomalia"}),
        rownames=["DBSCAN"],
        colnames=["Isolation Forest"],
    )


def main(
    features_path="data/processed/customer_features_model.csv",
    predictions_path="data/processed/model_predictions.csv",
):
    df_final = pd.read_csv(features_path, index_col="CustomerID")
    df_resultados = load_results(features_path, predictions_path)

    resumen = compare_models(df_final, df_resultados)
    print(resumen)
    print()
    print(crosstab_models(df_resultados))

    return resumen


if __name__ == "__main__":
    main()
