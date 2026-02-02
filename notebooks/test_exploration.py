import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 1. FIX DES CHEMINS (Indispensable avant d'importer nos propres fichiers)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Maintenant on peut importer nos modules
from src.data_generation import BondDataGenerator
from src.models import OASImputer

# 2. CONFIGURATION VISUELLE
plt.style.use('seaborn-v0_8') 

# 3. GÉNÉRATION DES DONNÉES
config_path = os.path.join(project_root, 'config', 'settings.yaml')
gen = BondDataGenerator(config_path)
df_full, metadata = gen.generate()
df_incomplete = gen.apply_illiquidity(df_full)

print(f"Nombre d'obligations : {df_full.shape[0]}")
print(f"Nombre de jours : {df_full.shape[1]}")

# --- TES ANCIENS GRAPHIQUES (On les garde pour la démo) ---

# 4. GRAPHIQUE 1 : Comparaison Réel vs Observé
bond_to_plot = df_full.index[0]
plt.figure(figsize=(12, 4))
plt.plot(df_full.columns, df_full.loc[bond_to_plot], label='Vraie Fair Value (Théorique)', alpha=0.4, color='blue')
plt.scatter(df_incomplete.columns, df_incomplete.loc[bond_to_plot], label='Prix observés (Marché)', color='red', s=20)
plt.title(f"Visualisation de l'illiquidité pour {bond_to_plot}")
plt.legend()
plt.show()

# --- MODÉLISATION ET RÉSULTATS ---

# 7. LANCEMENT DE L'IMPUTATION (ML)
# On initialise notre modèle avec plusieurs facteurs latents définis dans settings.yaml
n_factors_config = gen.config['model']['n_factors']
imputer = OASImputer(n_factors=n_factors_config)

# Méthode 1 : Naïve (on prend le dernier prix connu)
df_naive = imputer.naive_fill(df_incomplete)

# Méthode 2 : ML (Matrix Factorization / SVD)
df_ml = imputer.matrix_factorization_fill(df_incomplete)

# 8. CALCUL DE L'ERREUR (RMSE) - Version Pro
mask = df_incomplete.isnull()

# On extrait les valeurs réelles et prédites UNIQUEMENT là où il y avait des trous
y_true = df_full.values[mask.values]
y_naive = df_naive.values[mask.values]
y_ml = df_ml.values[mask.values]

# On calcule le RMSE sur ces vecteurs "plats"
rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))
rmse_ml = np.sqrt(mean_squared_error(y_true, y_ml))

print("\n" + "="*30)
print("RÉSULTATS DES MODÈLES")
print(f"Erreur Modèle Naïf : {rmse_naive:.2f} bps")
print(f"Erreur Modèle ML (SVD) : {rmse_ml:.2f} bps")
print("="*30)

# 9. LE GRAPHIQUE FINAL (La preuve par l'image)
plt.figure(figsize=(12, 6))
plt.plot(df_full.columns, df_full.loc[bond_to_plot], label='Vraie Valeur (Cachée)', alpha=0.3, color='blue', lw=3)
plt.scatter(df_incomplete.columns, df_incomplete.loc[bond_to_plot], color='red', label='Données dispo', s=30)
plt.plot(df_ml.columns, df_ml.loc[bond_to_plot], label='Prédiction ML (SVD)', color='green', linestyle='--')

plt.title(f"Reconstruction du spread pour {bond_to_plot}")
plt.xlabel("Date")
plt.ylabel("OAS Spread (bps)")
plt.legend()
plt.show()

# 10. ANALYSE APPROFONDIE DES ERREURS
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(df_full, df_incomplete, metadata)

# Comparaison des erreurs par secteur
errors_sector = evaluator.error_by_sector(df_ml)

plt.figure(figsize=(10, 6))
errors_sector.plot(kind='barh', color='skyblue')
plt.title("Précision du modèle par Secteur (RMSE en bps)")
plt.xlabel("Erreur (plus petit = meilleur)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

print("\n--- ERREUR PAR SECTEUR ---")
print(errors_sector)

# Comparaison des erreurs par Rating
errors_rating = evaluator.error_by_rating(df_ml)
print("\n--- ERREUR PAR RATING ---")
print(errors_rating)

# 11. STRATÉGIE DE TRADING (RICH/CHEAP)
from src.trading_strategy import TradingStrategy

strategy = TradingStrategy(threshold=10) # Seuil de 10 bps
signals, diffs = strategy.generate_signals(df_incomplete, df_ml)

# On regarde les opportunités au dernier jour de la simulation
last_date = df_full.columns[-1]
top_buy, top_sell = strategy.get_top_opportunities(diffs, last_date)

print(f"\n--- OPPORTUNITÉS AU {last_date.date()} ---")
print("\nTOP ACHAT (Marché trop pessimiste) :")
print(top_buy)
print("\nTOP VENTE (Marché trop optimiste) :")
print(top_sell)