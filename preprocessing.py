import pandas as pd
import numpy as np

def load_data(koi_path, confirmed_path=None, false_path=None):
    """
    Carga los datasets desde archivos CSV.
    Args:
        koi_path (str): ruta al archivo KOI cumulative.
        confirmed_path (str, opcional): ruta al archivo de planetas confirmados.
        false_path (str, opcional): ruta al archivo de falsos positivos.
    Returns:
        dict: contiene los DataFrames cargados.
    """
    data = {
        "koi": pd.read_csv(koi_path)
    }
    if confirmed_path:
        data["confirmed"] = pd.read_csv(confirmed_path)
    if false_path:
        data["false"] = pd.read_csv(false_path)
    return data


def clean_koi_data(df):
    """
    Limpia el dataset KOI cumulative.
    - Elimina columnas irrelevantes.
    - Maneja valores faltantes.
    - Convierte categorías a valores binarios (planeta real vs falso).
    """
    # Filtramos solo columnas relevantes
    cols = [
        "kepid", "koi_disposition", "koi_period", "koi_prad",
        "koi_srad", "koi_smass", "koi_teq", "koi_steff", "koi_slogg", "koi_smet"
    ]
    df = df[cols]

    # Renombrar columnas para que sean más claras
    df.rename(columns={
        "koi_disposition": "disposition",
        "koi_period": "orbital_period_days",
        "koi_prad": "planet_radius_re",
        "koi_srad": "star_radius_rs",
        "koi_smass": "star_mass_ms",
        "koi_teq": "equilibrium_temp_k",
        "koi_steff": "star_temp_k",
        "koi_slogg": "star_logg",
        "koi_smet": "star_metallicity"
    }, inplace=True)

    # Etiquetar target binario: 1 = planeta confirmado, 0 = falso positivo
    df = df[df["disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    df["label"] = np.where(df["disposition"] == "CONFIRMED", 1, 0)

    # Manejo de NaN: reemplazar por medianas
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def split_features_labels(df):
    """
    Separa las features (X) de las etiquetas (y).
    """
    X = df.drop(columns=["disposition", "label"])
    y = df["label"]
    return X, y


if __name__ == "__main__":
    # Ejemplo de uso
    data = load_data("../data/koi_cumulative.csv")
    koi_clean = clean_koi_data(data["koi"])
    X, y = split_features_labels(koi_clean)

    print("Shape de X:", X.shape)
    print("Distribución de clases:\n", y.value_counts())
