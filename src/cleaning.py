# src/cleaning.py
import pandas as pd
import numpy as np
from config.parameters import CLEANING_CONFIG

def drop_static_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime colonnes constantes ou trop manquantes."""
    to_drop = []

    # constantes
    to_drop += [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # trop de NaN
    ratio = df.isna().mean()
    to_drop += ratio[ratio > CLEANING_CONFIG["missing_ratio_limit"]].index.tolist()

    return df.drop(columns=to_drop), to_drop

def drop_high_corr(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Supprime une variable parmi chaque paire trop corrélée."""
    thr = threshold or CLEANING_CONFIG["correlation_threshold"]
    corr = df.select_dtypes(include='number').corr().abs()
    upper = corr.where(
        pd.DataFrame(np.triu(np.ones(corr.shape), k=1).astype(bool), 
                     index=corr.index, columns=corr.columns)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return df.drop(columns=to_drop), to_drop

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de nettoyage."""
    # 1. suppressions déclarées en config
    df = df.drop(columns=CLEANING_CONFIG["drop_cols"], errors="ignore")

    # 2. constantes & NaN
    df, dropped_1 = drop_static_and_missing(df)

    # 3. corrélation
    df, dropped_2 = drop_high_corr(df)

    dropped = CLEANING_CONFIG["drop_cols"] + dropped_1 + dropped_2
    print(f"Colonnes supprimées : {sorted(set(dropped))}")

    return df
