import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from exoplanets_ia import kepler_cumulative

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV

kepler_cumulative["label"] = kepler_cumulative["koi_disposition"].map({
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0
})

# Features a usar
features = ["koi_period", "koi_prad", "koi_depth", "koi_duration", 
            "koi_steff", "koi_slogg", "koi_srad"]

X = kepler_cumulative[features].fillna(0)
y = kepler_cumulative["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))


df = kepler_cumulative.copy()  # trabaja sobre una copia
# normalizamos la columna por si hay mayúsculas/minúsculas y NaN
df['koi_disposition'] = df['koi_disposition'].astype(str).str.strip().str.upper()

# crear label: CONFIRMED o CANDIDATE -> 1, else 0
df['label'] = df['koi_disposition'].map(lambda x: 1 if x in ("CONFIRMED","CANDIDATE") else 0)

# ver balance de clases
print("Distribución de labels:\n", df['label'].value_counts())
print("Porcentajes:\n", df['label'].value_counts(normalize=True))


candidates = [
    'koi_period','koi_prad','koi_depth','koi_duration',
    'koi_steff','koi_slogg','koi_srad','koi_model_snr',
    'koi_insol','koi_teq','koi_impact'
]

features = [c for c in candidates if c in df.columns]
print("Features que usaremos:", features)


# transformar numéricos: imputar mediana y escalar
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, features)
], remainder='drop')  # 'drop' mantiene solo las features listadas


X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Distribución en train:\n", y_train.value_counts(normalize=True))
print("Distribución en test:\n", y_test.value_counts(normalize=True))


clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # ayuda cuando las clases están desbalanceadas
    ))
])

# Entrenar
clf.fit(X_train, y_train)


# Predicciones
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("Classification report:\n", classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de confusión')
plt.show()

# ROC AUC
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.title('ROC Curve')
plt.show()

# Precision-Recall
ap = average_precision_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# obtener importancias (las features están en el orden de `features`)
importances = clf.named_steps['model'].feature_importances_
fi = pd.Series(importances, index=features).sort_values(ascending=False)
print(fi)

# gráfico
fi.plot(kind='bar', figsize=(8,4))
plt.title('Feature importances (RandomForest)')
plt.show()


param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
}

grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)
print("Mejor score (cv):", grid.best_score_)
best_model = grid.best_estimator_


joblib.dump(clf, "rf_exoplanet_classifier_basic.joblib")
# o si usaste grid:
joblib.dump(best_model, "rf_exoplanet_classifier_grid.joblib")


# X_new = candidates_df[features]
# probs = clf.predict_proba(X_new)[:,1]  # prob de ser planeta
# candidates_df['prob_planet'] = probs
# candidates_df.sort_values('prob_planet', ascending=False, inplace=True)
# print(candidates_df[['kepoi_name','prob_planet']].head(20))
