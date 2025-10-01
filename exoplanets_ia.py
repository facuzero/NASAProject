import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar datasets
kepler_False_positive_certified = pd.read_csv('CSV/kepler/Certifed_false_positivefpwg_2025.08.30_19.20.25.csv',skiprows=12,delimiter=',')
kepler_cumulative = pd.read_csv('CSV/kepler/cumulative_2025.08.30_19.18.46.csv',skiprows=52,sep='0')
kepler_False_positive = pd.read_csv('CSV/kepler/False_positives_q1_q17_dr25_koifpp_2025.08.30_19.19.44.csv',skiprows=38)
kepler_confirmed_names = pd.read_csv('CSV/kepler/kep_conf_names_2025.08.30_19.19.20.csv',skiprows=7)
kepler_stelar = pd.read_csv('CSV/kepler/keplerstellar.csv',low_memory=False)
exoplanetas_confirmados= pd.read_csv('CSV/exoplanetasConfirmados_PS_2025.08.30_18.58.25.csv',skiprows=96)
stellar_host= pd.read_csv('CSV/STELLARHOSTS_2025.08.30_18.59.57.csv',skiprows=49)

print(kepler_cumulative.columns.tolist())
# kepler_cumulative.columns= kepler_cumulative.columns.str.strip()
# kepler_cumulative.columns= kepler_cumulative.dropna(subset=["#"])



# plt.hist(kepler_cumulative["koi_period"], bins=50)
# plt.xlabel("Periodo orbital (días)")
# plt.ylabel("Número de planetas candidatos")
# plt.title("Distribución de periodos orbitales (Kepler)")
# plt.show()
