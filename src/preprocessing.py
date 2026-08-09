import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

COLS_LOG_NORMAL = [
    "Compras",
    "Canasta_Prom", "Ticket_Prom", "Precio_Prom", "Precio_Max",
    "Productos Distintos",
]
COLS_PW_NORMAL = ["Pct_Devoluciones"]
COLS_NUMERICAS = [
    "Permanencia", "Compras", "Canasta_Prom", "Ticket_Prom",
    "Precio_Prom", "Precio_Max", "Productos Distintos", "Pct_Devoluciones",
]


def transform_distributions(df):
    df = df.copy()
    transformers = dict()

    for col in COLS_LOG_NORMAL:
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
        transformer.fit(df[[col]].to_numpy())
        transformers[col] = transformer

    for col in COLS_PW_NORMAL:
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        transformer.fit(df[[col]].to_numpy())
        transformers[col] = transformer

    for col in [*COLS_LOG_NORMAL, *COLS_PW_NORMAL]:
        df[col] = transformers[col].transform(df[[col]].to_numpy())

    return df, transformers


def encode_categoricals(df):
    df = df.copy()
    df["Comprador Local"] = (df["Pais Principal"] == "United Kingdom").astype(int)
    df["Nomada"] = (df["Paises Distintos"] > 1).astype(int)
    return df


def scale_features(df):
    scaler = StandardScaler()
    df_num_escalado = pd.DataFrame(
        scaler.fit_transform(df[COLS_NUMERICAS]),
        columns=COLS_NUMERICAS,
        index=df.index,
    )
    df_cat_escalado = df[["Comprador Local", "Nomada"]]

    df_final = pd.concat([df_num_escalado, df_cat_escalado], axis=1)
    return df_final, scaler


def preprocess(df):
    df_transformed, transformers = transform_distributions(df)
    df_transformed = encode_categoricals(df_transformed)
    df_final, scaler = scale_features(df_transformed)
    return df_final, transformers, scaler


def main(
    input_path="data/processed/customer_features.csv",
    output_path="data/processed/customer_features_model.csv",
):
    df = pd.read_csv(input_path, index_col="CustomerID")
    df_final, _, _ = preprocess(df)
    df_final.to_csv(output_path, encoding="utf-8")
    return df_final


if __name__ == "__main__":
    main()
