import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, cols_relevantes: list) -> pd.DataFrame:
    """
    Limpia y normaliza los datos para usarlos en la IA.
    """
    if df.empty:
        print("⚠️ El DataFrame está vacío, no se puede procesar.")
        return df

    # 1. Seleccionamos solo columnas relevantes (ejemplo: habitabilidad / clasificación)
       df = df[[c for c in cols_relevantes if c in df.columns]]


    # 2. Eliminar filas con valores faltantes
    df = df.dropna()

    # 3. Normalizar columnas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()

    print("✅ Datos preprocesados correctamente")
    return df
