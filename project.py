import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
data = pd.read_csv(
    "insurance_data.csv"
)  # Remplacez 'insurance_data.csv' par le nom du fichier CSV

# Afficher les premières lignes des données pour vérification
print("Aperçu des données :")
print(data.head())

# Vérifier les types de données et les valeurs manquantes
print("\nInformations sur les données :")
print(data.info())

# Séparation des caractéristiques (features) et de la cible (target)
X = data.drop("Prix de la prime d'assurance", axis=1)
y = data["Prix de la prime d'assurance"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Prétraitement des données
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Modèles de régression
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

# Entraînement et évaluation des modèles
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"y_true": y_test, "y_pred": y_pred, "MSE": mse, "R2": r2}

# Affichage des résultats
print("\nRésultats des modèles :")
for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.2f}")

# Visualisations pour le modèle XGBoost
xgboost_results = results["XGBoost"]
y_true = xgboost_results["y_true"]
y_pred = xgboost_results["y_pred"]

# Prédictions vs Valeurs Réelles
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
plt.xlabel("Valeurs Réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs Valeurs Réelles pour XGBoost")
plt.grid(True)
plt.show()

# Histogramme des Erreurs
plt.figure(figsize=(10, 6))
errors = y_pred - y_true
sns.histplot(errors, bins=30, kde=True, color="blue")
plt.xlabel("Erreurs de Prédiction")
plt.ylabel("Fréquence")
plt.title("Histogramme des Erreurs pour XGBoost")
plt.grid(True)
plt.show()

# Importance des Caractéristiques
xgboost_model = models["XGBoost"].named_steps["regressor"]
importances = xgboost_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = list(numerical_features) + list(
    categorical_features
)  # Assurez-vous que c'est correct

plt.figure(figsize=(10, 6))
plt.title("Importance des Caractéristiques pour XGBoost")
plt.bar(range(X_train.shape[1]), importances[indices], color="skyblue")
plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
plt.show()
