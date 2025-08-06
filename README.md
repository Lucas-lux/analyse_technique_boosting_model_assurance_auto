# 🚗 Test Technique Assurance Auto - Analyse Prédictive des Primes

## 📋 Vue d'ensemble du projet

Ce projet présente une analyse complète de données d'assurance automobile visant à prédire les primes commerciales. L'étude comprend une exploration approfondie des données, la construction de modèles prédictifs et une analyse de dérive (drift) pour évaluer la robustesse des modèles dans le temps.

### 🎯 Objectifs principaux

1. **Prédiction des primes d'assurance** : Développer des modèles capables d'estimer les primes commerciales basées sur les caractéristiques des conducteurs et des véhicules
2. **Analyse exploratoire** : Comprendre les facteurs influençant les primes d'assurance
3. **Détection de dérive** : Évaluer la stabilité des modèles face aux changements temporels dans les données
4. **Comparaison de modèles** : Tester différentes approches (GLM vs LightGBM) pour optimiser les performances

## 📊 Structure du projet

```
test_technique_assurance_auto_lucas_bonere/
├── config/
│   └── parameters.py          # Configuration centralisée
├── data/
│   └── raw/                   # Données brutes
│       ├── X_train.csv        # Données d'entraînement
│       ├── X_test.csv         # Données de test
│       └── X_drift.csv        # Données de dérive
├── models/                    # Modèles entraînés
│   ├── glm_model.pkl
│   ├── lgbm_model.pkl
│   └── model_performance.json
├── notebooks/                 # Analyses Jupyter
│   ├── 01_exploration_initiale.ipynb
│   ├── 02_analyse_univariee.ipynb
│   ├── 03_analyse_bivariee.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_drift_analysis.ipynb
├── result/
│   └── submission.csv         # Prédictions finales
└── src/                       # Code source
    ├── cleaning.py           # Pipeline de nettoyage
    └── model.py              # Fonctions de modélisation
```

## 🔍 Analyse des données

### Variables disponibles

**Variables démographiques :**
- `AgeConducteur` : Âge du conducteur
- `SexeConducteur` : Genre du conducteur
- `StatutMatrimonial` : Statut marital

**Variables d'assurance :**
- `BonusMalus` : Coefficient bonus/malus
- `FrequencePaiement` : Fréquence de paiement
- `CodeProfession` : Code professionnel

**Variables véhicule :**
- `AgeVehicule` : Âge du véhicule
- `ClasseVehicule` : Classe du véhicule
- `PuissanceVehicule` : Puissance du véhicule
- `CarburantVehicule` : Type de carburant
- `UsageVehicule` : Usage du véhicule

**Variables contextuelles :**
- `Garage` : Type de garage
- `Region` : Région géographique

**Variable cible :**
- `PrimeCommerciale` : Prime d'assurance à prédire

### Nettoyage des données

Le pipeline de nettoyage (`src/cleaning.py`) applique les transformations suivantes :

1. **Suppression de colonnes** : Variables constantes ou trop manquantes (>50%)
2. **Gestion des corrélations** : Suppression de variables trop corrélées (>90%)
3. **Variables supprimées** : `PolicyId`, `CodeProfession`, `StatutMatrimonial`

## 🤖 Modélisation

### Approche adoptée

Deux modèles complémentaires ont été développés :

#### 1. Modèle GLM (Tweedie/Gamma)
- **Distribution** : Tweedie avec power=2 (distribution Gamma)
- **Fonction de lien** : Log-link
- **Régularisation** : L1/L2 avec alpha optimisé par validation croisée
- **Avantages** : Interprétabilité, robustesse statistique

#### 2. Modèle LightGBM
- **Algorithme** : Gradient Boosting optimisé
- **Encodage** : Label encoding automatique des variables catégorielles
- **Hyperparamètres** : Optimisés par GridSearchCV
- **Avantages** : Performance élevée, gestion automatique des interactions

### Métriques d'évaluation

- **RMSE** : Root Mean Square Error
- **MAE** : Mean Absolute Error  
- **R²** : Coefficient de détermination
- **MAPE** : Mean Absolute Percentage Error

### Performances initiales

| Métrique | GLM | LightGBM |
|----------|-----|----------|
| RMSE     | 108.91 | 107.91 |
| MAE      | 73.40 | 73.40 |
| R²       | 0.75 | 0.75 |
| MAPE     | 17.98% | 17.98% |

## 📈 Analyse de dérive (Drift Analysis)

### Détection de dérive

L'analyse de dérive révèle des changements significatifs dans les données :

#### Variables avec dérive détectée (5/11 - 45%)

**Variables numériques (100% touchées) :**
- `AgeConducteur` : Distribution des âges modifiée
- `BonusMalus` : Profil de risque des conducteurs évolué
- `AgeVehicule` : Parc automobile rajeuni

**Variables catégorielles (25% touchées) :**
- `FrequencePaiement` : Habitudes de paiement modifiées
- `ClasseVehicule` : Types de véhicules assurés différents

#### Variables stables (6/11 - 55%)
- Variables démographiques et géographiques
- Caractéristiques techniques des véhicules
- Conditions d'usage et de stationnement

### Impact sur les performances

**Dégradation majeure des métriques :**

| Métrique | Initial | Drift | Dégradation |
|----------|---------|-------|-------------|
| RMSE     | 107.91€ | 266.62€ | +147% |
| MAE      | 73.40€  | 214.09€ | +192% |
| R²       | 0.75    | -0.35  | -147% |
| MAPE     | 17.98%  | 43.90% | +144% |

### Changements de corrélations

**Corrélations significativement altérées :**
- `AgeVehicule` : -0.549 → -0.175 (affaiblissement de 68%)
- `BonusMalus` : 0.360 → 0.115 (baisse de 68%)

## 🔧 Utilisation

### Prérequis

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn joblib
```

### Exécution des analyses

1. **Exploration initiale** :
```bash
jupyter notebook notebooks/01_exploration_initiale.ipynb
```

2. **Analyse univariée** :
```bash
jupyter notebook notebooks/02_analyse_univariee.ipynb
```

3. **Analyse bivariée** :
```bash
jupyter notebook notebooks/03_analyse_bivariee.ipynb
```

4. **Entraînement des modèles** :
```bash
jupyter notebook notebooks/04_model_training.ipynb
```

5. **Analyse de dérive** :
```bash
jupyter notebook notebooks/05_drift_analysis.ipynb
```

### Génération des prédictions

```python
from src.model import predict_test
import joblib

# Charger les modèles
glm_model = joblib.load('models/glm_model.pkl')
lgbm_model = joblib.load('models/lgbm_model.pkl')

# Générer les prédictions
predict_test({'glm': glm_model, 'lgbm': lgbm_model})
```

## 📊 Résultats et insights

### Facteurs clés influençant les primes

1. **Âge du véhicule** : Corrélation négative forte (-0.55)
2. **Bonus/Malus** : Corrélation positive modérée (0.36)
3. **Âge du conducteur** : Corrélation négative faible (-0.21)

### Patterns de dérive identifiés

1. **Évolution temporelle naturelle** : Changement des habitudes de consommation
2. **Changement de clientèle** : Nouveaux segments de marché
3. **Contexte externe** : Facteurs macro-économiques ou réglementaires
4. **Biais d'échantillonnage** : Différences dans les canaux d'acquisition

### Recommandations

1. **Surveillance continue** : Mise en place d'un monitoring de dérive
2. **Réentraînement périodique** : Adaptation des modèles aux nouvelles données
3. **Feature engineering** : Création de variables plus robustes au temps
4. **Ensemble methods** : Combinaison de modèles pour améliorer la robustesse

## 🎯 Conclusion

Ce projet démontre l'importance de la détection de dérive dans les modèles de machine learning appliqués à l'assurance. Les résultats montrent que même des modèles performants initialement peuvent subir une dégradation significative face aux changements temporels dans les données.

L'analyse révèle que les variables comportementales et temporelles sont plus sensibles à la dérive que les caractéristiques structurelles, suggérant une évolution naturelle du marché de l'assurance plutôt qu'un biais méthodologique.

**Auteur** : Lucas Bonere  
**Date** : 2025 
**Technologies** : Python, Scikit-learn, LightGBM, Pandas, Matplotlib, Seaborn 