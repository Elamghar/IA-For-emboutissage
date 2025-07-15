# Synthèse des données
import pandas as pd
import numpy as np

np.random.seed(42)

# Réglages ABB
max_speed = 2.0  # m/s (ex. IRB 1600)
distance = 1.0   # m
alpha = 0.08
beta1 = 0.1
beta2 = 0.3

# Génération de poids entre 0.5 et 10 kg
poids = np.random.uniform(0.5, 10, 100)

# Vibration simulée : augmente avec le poids + bruit
vibration = 0.02 + alpha * poids + np.random.normal(0, 0.01, 100)

# Vitesse : pénalisée par poids et vibration
vitesse = max_speed - beta1 * poids - beta2 * vibration + np.random.normal(0, 0.05, 100)
vitesse = np.clip(vitesse, 0.1, None)

# Temps de déplacement : distance / vitesse + retard dû à vibration
temps = distance / vitesse + vibration * 0.5 + np.random.normal(0, 0.02, 100)

# Résultat dans DataFrame
df = pd.DataFrame({
    "poids_kg": np.round(poids, 2),
    "vibration_mps2": np.round(vibration, 3),
    "vitesse_mps": np.round(vitesse, 3),
    "temps_s": np.round(temps, 3)
})

print(df)
df.to_csv('abb_synthetic_data.csv', index=False)

# trainement des mdoels:
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ------------------------------------------------------------------
# 1) Charger le CSV généré (ou utiliser ton propre DataFrame)
# ------------------------------------------------------------------
df = pd.read_csv("abb_synthetic_data.csv")  

X = df[["poids_kg", "vibration_mps2"]]
y_speed = df["vitesse_mps"]
y_time  = df["temps_s"]

# ------------------------------------------------------------------
# 2) Split & Scaling
# ------------------------------------------------------------------
X_train, X_test, y_speed_train, y_speed_test = train_test_split(
    X, y_speed, test_size=0.2, random_state=42)

# Le même split pour le temps (pour garder les mêmes indices)
_, _, y_time_train, y_time_test = train_test_split(
    X, y_time, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3) Entraîner les deux modèles
# ------------------------------------------------------------------
model_speed = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_time  = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

model_speed.fit(X_train_scaled, y_speed_train)
model_time.fit(X_train_scaled, y_time_train)

# ------------------------------------------------------------------
# 4) Sauvegarder scaler + modèles
# ------------------------------------------------------------------
# a) Le scaler (optionnel mais conseillé)
joblib.dump(scaler, "scaler_abbr_2f.joblib")

# b) Les deux modèles
joblib.dump(model_speed, "xgb_speed_model.joblib")
joblib.dump(model_time,  "xgb_time_model.joblib")

print("✅ Modèles et scaler sauvegardés !")

# Test et import de models
import joblib
import numpy as np

# Rechargement
scaler       = joblib.load("scaler_abbr_2f.joblib")
model_speed  = joblib.load("xgb_speed_model.joblib")
model_time   = joblib.load("xgb_time_model.joblib")

# Exemple de prédiction
new_sample = np.array([[5.2, 0.45]])  # [poids_kg, vibration_mps2]
X_scaled   = scaler.transform(new_sample)

pred_speed = model_speed.predict(X_scaled)
pred_time  = model_time.predict(X_scaled)

print(f"Vitesse prédite : {pred_speed[0]:.3f} m/s")
print(f"Temps   prédit : {pred_time[0]:.3f} s")


