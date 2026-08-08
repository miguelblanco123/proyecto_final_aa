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
