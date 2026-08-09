import numpy as np
import pandas as pd


def load_online_retail_ii(path):
    df = pd.read_excel(path)
    df.rename(columns={
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }, inplace=True)
    return df


def load_online_retail(path):
    df = pd.read_excel(path)
    df.rename(columns={"InvoiceNo": "Invoice"}, inplace=True)
    return df


def merge_sources(online_retail_ii, online_retail):
    # Se omiten las ordenes de compra que ya estan en el primer dataset
    online_retail = online_retail[~online_retail["Invoice"].isin(online_retail_ii["Invoice"].unique())]

    df_raw = pd.concat([online_retail_ii, online_retail])
    df_raw.sort_values(by=["InvoiceDate", "Invoice", "StockCode"], inplace=True)

    cols_str = ["Invoice", "StockCode", "Description", "Country"]
    for col in cols_str:
        df_raw[col] = df_raw[col].astype(str)

    cols_float = ["Quantity", "UnitPrice", "CustomerID"]
    for col in cols_float:
        df_raw[col] = df_raw[col].astype(float)

    df_raw["InvoiceDate"] = pd.to_datetime(df_raw["InvoiceDate"])
    return df_raw


def clean(df_raw):
    df = df_raw.copy()

    df.dropna(subset="CustomerID", inplace=True)
    df.sort_values(by=["CustomerID", "InvoiceDate", "Invoice", "StockCode"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    return df


def load_data(raw_dir="data/raw"):
    online_retail_ii = load_online_retail_ii(f"{raw_dir}/online_retail_ii.xlsx")
    online_retail = load_online_retail(f"{raw_dir}/online_retail.xlsx")

    df_raw = merge_sources(online_retail_ii, online_retail)
    return clean(df_raw)


def build_customer_features(df):
    df_prod = df.copy()

    clientes_de_pruebas = df_prod[df_prod["StockCode"].str.casefold().str.contains("test")]["CustomerID"].drop_duplicates()
    df_prod = df_prod[~df_prod["CustomerID"].isin(clientes_de_pruebas)]

    df_prod_compras = df_prod[df_prod["Quantity"] >= 0]
    df_prod_devolucion = df_prod[df_prod["Quantity"] < 0]

    # -- Compras reales --
    df_customer = df_prod_compras.groupby(by=["CustomerID"]).agg({
        "Country": pd.Series.nunique,
        "InvoiceDate": ["min", "max"],
        "Sales": "sum",
        "Quantity": "sum",
        "Invoice": pd.Series.nunique,
        "UnitPrice": "max",
        "StockCode": pd.Series.nunique,
    })
    df_customer.columns = ["Paises Distintos", "Date_Min", "Date_Max", "Sales", "Quantity", "Compras", "Precio_Max", "Productos Distintos"]
    df_customer.reset_index(inplace=True)

    df_customer_top_country = df_prod_compras.groupby(by=["CustomerID", "Country"])["Invoice"].nunique().rename("Compras").reset_index().sort_values(by=["CustomerID", "Compras"], ascending=False)

    df_customer_top_country["rn"] = df_customer_top_country.groupby(by="CustomerID")["CustomerID"].cumcount()
    df_customer_top_country = df_customer_top_country[df_customer_top_country["rn"] == 0].drop(columns=["rn", "Compras"]).rename(columns={"Country": "Pais Principal"})

    df_customer = pd.merge(df_customer, df_customer_top_country, on="CustomerID", how="left")
    df_customer.set_index("CustomerID", inplace=True)

    df_customer["Permanencia"] = np.round((df_customer["Date_Max"] - df_customer["Date_Min"]).dt.total_seconds() / 60 / 60 / 24, 0) + 1

    df_customer["Canasta_Prom"] = df_customer["Quantity"] / df_customer["Compras"]
    df_customer["Ticket_Prom"] = df_customer["Sales"] / df_customer["Compras"]
    df_customer["Precio_Prom"] = df_customer["Sales"] / df_customer["Quantity"]

    # -- Devoluciones --
    df_customer_devolucion = df_prod_devolucion.groupby(by="CustomerID")["Invoice"].nunique().rename("Devoluciones").reset_index()

    # -- Union de datos --
    df_customer = pd.merge(df_customer, df_customer_devolucion, on="CustomerID", how="left")
    df_customer.set_index("CustomerID", inplace=True)
    df_customer.loc[df_customer["Devoluciones"].isna(), "Devoluciones"] = 0

    df_customer["Pct_Devoluciones"] = df_customer["Devoluciones"] / df_customer["Compras"]

    df_customer["Pais Principal"] = df_customer["Pais Principal"].astype("category")
    df_customer["Paises Distintos"] = df_customer["Paises Distintos"].astype("category")

    return df_customer[[
        "Pais Principal", "Paises Distintos",
        "Permanencia",
        "Compras",
        "Canasta_Prom", "Ticket_Prom", "Precio_Prom", "Precio_Max",
        "Productos Distintos",
        "Pct_Devoluciones",
    ]]


def main(raw_dir="data/raw", output_path="data/processed/customer_features.csv"):
    df = load_data(raw_dir=raw_dir)
    df_customer = build_customer_features(df)
    df_customer.to_csv(output_path, encoding="utf-8")
    return df_customer


if __name__ == "__main__":
    main()
