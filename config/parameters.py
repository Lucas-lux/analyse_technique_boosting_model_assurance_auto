"""
Configuration parameters for Direct Assurance technical test
Centralizes all project parameters to avoid hard-coding
"""

# Data parameters
DATA_CONFIG = {
    'raw_data_path': 'data/raw/',
    'train_file': 'X_train.csv',
    'test_file': 'X_test.csv',
    'drift_file': 'X_drift.csv',
    'target_column': 'PrimeCommerciale',
    'id_column': 'PolicyId',
    'random_state': 42
}

# EDA parameters
EDA_CONFIG = {
    'figsize': (12, 8),
    'correlation_threshold': 0.8,
    'outlier_threshold': 3,      
    'missing_threshold': 0.05,  
    'plot_style': 'whitegrid'
}

# Model parameters
MODEL_CONFIG = {
    # Train/validation split
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'scoring': 'neg_root_mean_squared_error',

    # GLM (Tweedie/Gamma) parameters
    'glm_power': 2,             # Gamma distribution
    'glm_link': 'log',          # log-link
    'glm_alphas': [0.01, 0.1, 1.0, 5.0],  
    'glm_max_iter': 1000,

    # LightGBM parameters
    'lgbm_learning_rates': [0.01, 0.05, 0.1],
    'lgbm_num_leaves': [31, 63, 127],
    'lgbm_n_estimators': [100, 300, 500],
    'lgbm_max_depth': [-1, 6, 10], 
}

# Insurance specific parameters
INSURANCE_CONFIG = {
    'bonus_malus_range': (50, 350),
    'age_categories': {
        'jeune': (18, 25),
        'adulte': (26, 60),
        'senior': (61, 100)
    },
    'prime_categories': {
        'low': (0, 500),
        'medium': (500, 1500),
        'high': (1500, float('inf'))
    }
}

CLEANING_CONFIG = {
    "drop_cols": [
        "PolicyId",          
        "CodeProfession", 
        "StatutMatrimonial"
    ],
    "correlation_threshold": 0.90,
    "missing_ratio_limit": 0.50
}

# Drift detection parameters
DRIFT_CONFIG = {
    'drift_threshold': 0.1,
    'statistical_tests': ['ks_test', 'chi2_test'],
    'psi_threshold': 0.2
}
