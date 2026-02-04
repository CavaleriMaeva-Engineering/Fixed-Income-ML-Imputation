# Fixed-Income ML Imputation : Spreads High Yield & Analyse de Valeur Relative

**Auteur :** Maéva Cavaleri (Télécom SudParis)  

---
## Présentation du Projet

Le marché des obligations **High Yield** (notées en dessous de BBB-) est structurellement marqué par une forte **illiquidité**. Contrairement aux actions, ces titres ne s'échangent pas quotidiennement, ce qui génère des "trous" (données manquantes) dans les séries temporelles de spreads **OAS (Option-Adjusted Spread)**.

Ce projet propose une solution basée sur l'Apprentissage Automatique pour :
1. **Simuler un univers obligataire réaliste** avec des composantes de marché, de secteur et de risque idiosyncratique.
2. **Imputer les données manquantes** à l'aide d'un algorithme de **Matrix Factorization (SVD Itérative)** capable de capturer les facteurs latents du marché.
3. **Identifier des opportunités d'arbitrage** via une stratégie de **Relative Value (Rich/Cheap)** en comparant le prix de marché observé à la "Fair Value" reconstruite par l'IA.

---

## Stack Technique

*   **Langage :** Python 3.x
*   **Data Science :** `pandas`, `numpy`, `scikit-learn`
*   **Visualisation :** `matplotlib`, `seaborn`
*   **Configuration :** `pyyaml`

---

## Structure du Projet

```text
├── config/
│   └── settings.yaml          # Paramètres de simulation et du modèle
├── src/
│   ├── data_generation.py     # Moteur de simulation de spreads OAS
│   ├── models.py              # Imputers (Naïf et Matrix Factorization)
│   ├── evaluation.py          # Métriques de performance (RMSE par secteur/rating)
│   └── trading_strategy.py    # Générateur de signaux Rich/Cheap
├── notebooks/
│   └── analysis.ipynb         # Démonstration complète et visualisations
├── main.py                    # Script d'exécution principal
└── README.md
```

---

## Méthodologie

### 1. Modèle de Génération des Données
Les spreads sont simulés selon un modèle factoriel :
$$Spread_{i,t} = Base + \beta_i \cdot Mkt_t + Sector_{s,t} + \epsilon_{i,t}$$
L'illiquidité est introduite en supprimant aléatoirement un pourcentage défini de données (par défaut **75% de données manquantes**).

### 2. Algorithme d'Imputation : SVD Itérative
Plutôt qu'un simple remplissage moyen, le modèle utilise une **Décomposition en Valeurs Singulières (SVD)** :
*   Il réduit la dimension de la matrice des spreads pour identifier les thèmes cachés (facteurs latents).
*   Il reconstruit la matrice complète à partir de ces facteurs.
*   Le processus est itéré pour affiner la précision de la Fair Value.

### 3. Stratégie Relative Value
*   **Signal d'Achat (CHEAP) :** OAS Marché > Fair Value + Seuil.
*   **Signal de Vente (RICH) :** OAS Marché < Fair Value - Seuil.

---

## Installation et Utilisation

### Installation
```bash
git clone https://github.com/CavaleriMaeva-Engineering/Fixed-Income-ML-Imputation.git
cd Fixed-Income-ML-Imputation
pip install -r requirements.txt
```

### Exécution du projet
Pour lancer la simulation complète et voir les résultats :
```bash
python main.py
```

---

## Résultats & Visualisations

Le projet génère des analyses graphiques permettant de juger de la qualité de la reconstruction :
*   **Visualisation de l'illiquidité :** Heatmaps des données manquantes.
*   **Reconstruction de courbe :** Comparaison entre la Vraie Valeur (cachée), les données observées et la prédiction du modèle.
*   **Analyse d'erreur :** RMSE décomposé par secteur industriel et par notation de crédit (Rating).

---

## Configuration
Vous pouvez modifier les paramètres du marché ou de l'IA dans le fichier `config/settings.yaml` :
```yaml
simulation:
  n_bonds: 300
  n_days: 500
illiquidity:
  missing_rate: 0.75  # 75% de trous
model:
  n_factors: 15       # Nombre de facteurs latents pour la SVD
```
