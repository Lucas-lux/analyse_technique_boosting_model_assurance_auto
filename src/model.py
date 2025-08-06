# src/models.py

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config.parameters import DATA_CONFIG, MODEL_CONFIG
from src.cleaning import clean_dataset 

def get_preprocessor(X):
    """Crée un preprocesseur pour les variables numériques et catégorielles."""
    # Identifier les variables numériques et catégorielles
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Créer les transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    return preprocessor, numeric_features, categorical_features

def load_data():
    """Charge les données préparées pour l'entraînement."""
    df = pd.read_csv(os.path.join("..", DATA_CONFIG['raw_data_path'], DATA_CONFIG['train_file']))
    df  = clean_dataset(df)   
    y = df.pop(DATA_CONFIG['target_column'])
    X = df.drop(columns=[], errors="ignore")
    return train_test_split(
        X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state']
    )

def load_test_data():
    """Charge le jeu de test et retourne X_test et PolicyId."""
    df = pd.read_csv(os.path.join("..", DATA_CONFIG['raw_data_path'], DATA_CONFIG['test_file']))
    
    ids = df[DATA_CONFIG['id_column']]
    df = clean_dataset(df)   
    X_test = df.drop(columns=[], errors="ignore")
    return X_test, ids

def load_drift_data():
    """Charge le jeu de test et retourne X_test et PolicyId."""
    df = pd.read_csv(os.path.join("..", DATA_CONFIG['raw_data_path'], DATA_CONFIG['drift_file']))
    ids = df[DATA_CONFIG['id_column']]
    df = clean_dataset(df)   
    X_drift = df.drop(columns=[], errors="ignore")
    return X_drift, ids

def train_glm(X_train, y_train):
    """Entraîne et valide un GLM Tweedie (Gamma)."""
    # Créer le preprocesseur pour les données
    preprocessor, numeric_features, categorical_features = get_preprocessor(X_train)
    
    print(f"Variables numériques: {numeric_features}")
    print(f"Variables catégorielles: {categorical_features}")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('glm', TweedieRegressor(
            power=MODEL_CONFIG['glm_power'],
            link=MODEL_CONFIG['glm_link'],
            alpha=MODEL_CONFIG['glm_alphas'],
            max_iter=MODEL_CONFIG['glm_max_iter']
        ))
    ])
    param_grid = {'glm__alpha': MODEL_CONFIG['glm_alphas']}
    search = GridSearchCV(
        pipeline, param_grid,
        cv=MODEL_CONFIG['cv_folds'],
        scoring=MODEL_CONFIG['scoring'],
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def train_lightgbm(X_train, y_train):
    """Entraîne et valide un LightGBM."""
    # LightGBM peut gérer automatiquement les variables catégorielles
    # Il faut juste encoder les variables string en catégories
    X_train_lgb = X_train.copy()
    # Détecter les colonnes object
    cat_cols = X_train_lgb.select_dtypes(include=['object']).columns.tolist()
    print(f"Variables catégorielles identifiées : {cat_cols}")
    
    # Créer et stocker les encodeurs pour réutilisation
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train_lgb[col] = le.fit_transform(X_train_lgb[col])
        label_encoders[col] = le  # Stocker pour réutilisation
    
    print(f"Variables catégorielles pour LightGBM: {cat_cols}")
    
    lgbm = lgb.LGBMRegressor(
        random_state=DATA_CONFIG['random_state'],
        verbose=1
    )
    param_grid = {
        'learning_rate': MODEL_CONFIG['lgbm_learning_rates'],
        'num_leaves': MODEL_CONFIG['lgbm_num_leaves'],
        'n_estimators': MODEL_CONFIG['lgbm_n_estimators']
    }
    search = GridSearchCV(
        lgbm, param_grid,
        cv=MODEL_CONFIG['cv_folds'],
        scoring=MODEL_CONFIG['scoring'],
        verbose=1
    )
    search.fit(X_train_lgb, y_train)
    return search.best_estimator_

def prepare_validation_lightgbm(X_val, X_train):
    """Encode X_val de la même manière que X_train pour LightGBM."""
    X_val_lgb = X_val.copy()
    categorical_features = X_val.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Encodage des variables catégorielles pour validation: {categorical_features}")
    
    # Encoder chaque variable catégorielle en utilisant les valeurs de X_train comme référence
    for col in categorical_features:
        le = LabelEncoder()
        # Ajuster l'encodeur sur les valeurs combinées de train et validation
        combined_values = pd.concat([X_train[col], X_val[col]]).unique()
        le.fit(combined_values)
        X_val_lgb[col] = le.transform(X_val[col])
        
    return X_val_lgb

def evaluate_model(model, X_val, y_val):
    
    """Calcule RMSE, MAE et R² sur le jeu de validation."""
    y_pred = model.predict(X_val)
    return {
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
        'MAE': mean_absolute_error(y_val, y_pred),
        'R2': r2_score(y_val, y_pred), 
        'MAPE': mean_absolute_percentage_error(y_val, y_pred)*100
    }

def predict_test(models, output_path='models/submission.csv'):
    """
    Génère les prédictions pour le jeu de test.
    Args:
        models (dict): {'glm': glm_model, 'lgbm': lgbm_model}
        output_path (str): chemin du fichier CSV à sauvegarder
    """
    X_test, ids = load_test_data()
    print(f"Données de test chargées: {X_test.shape}")
    print(f"Types de données dans X_test:")
    print(X_test.dtypes.value_counts())

    # Moyenne des prédictions
    pred_glm  = models['glm'].predict(X_test)

    X_test_lgbm = prepare_validation_lightgbm(X_test, X_test)
    pred_lgbm = models['lgbm'].predict(X_test_lgbm)

    preds     = 0.5 * pred_glm + 0.5 * pred_lgbm

    print(f"\n=== STATISTIQUES DES PRÉDICTIONS ===")
    print(f"GLM - Min: {pred_glm.min():.2f}, Max: {pred_glm.max():.2f}, Moyenne: {pred_glm.mean():.2f}")
    print(f"LightGBM - Min: {pred_lgbm.min():.2f}, Max: {pred_lgbm.max():.2f}, Moyenne: {pred_lgbm.mean():.2f}")
    print(f"Ensemble - Min: {preds.min():.2f}, Max: {preds.max():.2f}, Moyenne: {preds.mean():.2f}")

    submission = pd.DataFrame({
        DATA_CONFIG['id_column']:           ids,
        f"{DATA_CONFIG['target_column']}Pred": preds
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Soumission enregistrée dans {output_path}")


