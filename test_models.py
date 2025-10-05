import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


kepler_data = pd.read_csv('CSV/kepler/cumulative_2025.10.03_11.06.40.csv',skiprows=144)
kepler_data_redNeuronal = pd.read_csv('CSV/kepler/cumulative_2025.10.03_11.06.40.csv',skiprows=144)
features = ['koi_period','koi_prad','koi_depth','koi_duration',
            'koi_steff','koi_slogg','koi_srad','koi_model_snr',
            'koi_insol','koi_teq','koi_impact']
X_new = kepler_data[features]

    #Carga del modelo de Random Forest
clf = joblib.load("rf_exoplanet_classifier_basic.joblib")

    #Prediccion
probs = clf.predict_proba(X_new)[:,1]
kepler_data["prob_planet"] = probs

#print(kepler_data.head())

model= load_model("exoplanet__nn.h5")

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new) 

# predicciones
probs = model.predict(X_new_scaled).flatten()
kepler_data_redNeuronal["prob_planet"] = probs

print(kepler_data_redNeuronal.head())