# 🌌 Exoplanet AI Classifier  

Proyecto desarrollado para el **NASA Space Apps Challenge 2025**, enfocado en la **clasificación de exoplanetas reales vs falsos positivos** y el análisis de su **potencial habitabilidad**.  

## 🚀 Objetivo  
1. Entrenar un modelo de **IA supervisada** que aprenda a distinguir entre **exoplanetas confirmados** y **falsos positivos** a partir de los datos de la misión Kepler, para luego poder identificar nuevos planetas.  
2. Analizar las características físicas de los planetas confirmados para calcular un **Índice de Habitabilidad** (Habitability Score).  

## 📊 Datos utilizados  
Los datasets fueron descargados desde el **NASA Exoplanet Archive**:  

- **KOI Table (Cumulative)** → Lista de candidatos (CONFIRMED, CANDIDATE, FALSE POSITIVE).  
- **Kepler False Positive Probabilities** → Probabilidad de que un objeto sea un falso positivo.  
- **Kepler Certified False Positives** → Catálogo oficial de descartados.  
- **Confirmed Exoplanets Table** → Exoplanetas confirmados con parámetros físicos (masa, radio, temperatura, flujo, etc.).  

Formato: `.csv`  

## 🛠️ Tecnologías utilizadas  

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)  
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)  
![Seaborn](https://img.shields.io/badge/Seaborn-Stats%20Plots-3776AB?logo=python&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)  
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)  
