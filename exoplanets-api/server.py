import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os

# Inicializar Flask
app = Flask(__name__)

# Paths de los modelos
MODEL_PATH = os.path.join("models", "exoplanet_nn.h5")
SCALER_PATH = os.path.join("models", "scaler_kepler.joblib")

# Cargar modelo entrenado
model = load_model(MODEL_PATH)

# Cargar scaler (si existe, sino None)
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

# Columnas que el modelo espera (en el mismo orden del entrenamiento)
FEATURE_COLUMNS = [
    "koi_period",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad"
]


@app.route("/")
def home():
    return jsonify({"message": "Exoplanet Prediction API is running 🚀"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validar que se reciban todas las features
        if not all(col in data for col in FEATURE_COLUMNS):
            return jsonify({
                "error": f"Missing features. Expected {FEATURE_COLUMNS}"
            }), 400

        # Convertir a vector numpy en el orden correcto
        input_data = np.array([[data[col] for col in FEATURE_COLUMNS]])

        # Escalar si corresponde
        if scaler is not None:
            input_data = scaler.transform(input_data)

        # Predicción
        prediction = model.predict(input_data)
        probability = float(prediction[0][0])

        # Clasificación simple (puedes ajustarlo según tu problema)
        label = "Exoplanet Candidate" if probability >= 0.5 else "Not Exoplanet"

        return jsonify({
            "prediction": label,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Para desarrollo local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
