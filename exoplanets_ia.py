import pandas as pd
import matplotlib.pyplot as plt

    # 1. Cargar datasets

kepler_False_positive_certified = pd.read_csv('CSV/kepler/Certifed_false_positivefpwg_2025.08.30_19.20.25.csv',skiprows=13)
# print("KEPLER FALSE CERT",kepler_False_positive_certified.head())

kepler_cumulative = pd.read_csv('CSV/kepler/cumulative_2025.08.30_19.18.46.csv',skiprows=53)
# print(kepler_cumulative.head())

kepler_False_positive = pd.read_csv('CSV/kepler/False_positives_q1_q17_dr25_koifpp_2025.08.30_19.19.44.csv',skiprows=39)
# print(kepler_False_positive.head())

kepler_confirmed_names = pd.read_csv('CSV/kepler/kep_conf_names_2025.08.30_19.19.20.csv',skiprows=8)
# print(kepler_confirmed_names.head())

kepler_stelar = pd.read_csv('CSV/kepler/keplerstellar.csv',low_memory=False)
# print(kepler_stelar.head())

exoplanetas_confirmados= pd.read_csv('CSV/exoplanetasConfirmados_PS_2025.08.30_18.58.25.csv',skiprows=96)
# print(exoplanetas_confirmados.head())

stellar_host= pd.read_csv('CSV/STELLARHOSTS_2025.08.30_18.59.57.csv',skiprows=49)
# print(stellar_host.head())



