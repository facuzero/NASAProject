import pandas as pd

BASE_PATH = "CSV"

def load_csv(file_path: str) -> pd.DataFrame:    
    #Carga un archivo CSV y devuelve un DataFrame de pandas.
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Archivo cargado correctamente: {file_path}")
        print(f"📊 Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        return df
    except Exception as e:
        print(f"❌ Error cargando el archivo {file_path}: {e}")
        return pd.DataFrame()
        
def load_kepler_data() -> dict:
    """
    Carga todos los datasets de la carpeta Kepler.
    """
    kepler_path = os.path.join(BASE_PATH, "kepler")

    datasets = {
        "certified_false_positive": load_csv(os.path.join(kepler_path, "Certifed_false_positive.csv")),
        "cumulative": load_csv(os.path.join(kepler_path, "cumulative.csv")),
        "false_positive": load_csv(os.path.join(kepler_path, "False_positive.csv")),
        "conf_names": load_csv(os.path.join(kepler_path, "kep_conf_names.csv")),
        "stellar": load_csv(os.path.join(kepler_path, "keplerstellar.csv"))
    }

    return datasets

def load_extra_data() -> dict:
    """
    Carga datasets adicionales fuera de Kepler.
    """
    datasets = {
        "exoplanetas_confirmados": load_csv(os.path.join(BASE_PATH, "exoplanetasConfirmados.csv")),
        "stellar_host": load_csv(os.path.join(BASE_PATH, "STELLARHOST.csv"))
    }

    return datasets