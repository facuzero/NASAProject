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

- **Lenguaje**: Python
- **Librerías principales**:  
  - pandas  
  - numpy  
  - matplotlib / seaborn  
  - scikit-learn    
  - tensorflow / keras