# robustness.py
# ฟังก์ชันเกี่ยวกับ Robustness & Generalization

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from ProjectP import ensure_super_features_file, get_feature_target_columns

def run_time_series_cv():
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        print(f'[TimeSeriesCV] Fold {fold+1} AUC: {auc:.4f}')
    print(f'[TimeSeriesCV] Mean AUC: {sum(aucs)/len(aucs):.4f}')

def run_walk_forward_analysis():
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    n_splits = 5
    split_size = len(X) // n_splits
    aucs = []
    for i in range(n_splits-1):
        train_end = split_size * (i+1)
        val_start = train_end
        val_end = split_size * (i+2)
        X_train, X_val = X.iloc[:train_end], X.iloc[val_start:val_end]
        y_train, y_val = y.iloc[:train_end], y.iloc[val_start:val_end]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        print(f'[WFV] Window {i+1} AUC: {auc:.4f}')
    print(f'[WFV] Mean AUC: {sum(aucs)/len(aucs):.4f}')

def test_on_unseen_data():
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    split = int(len(df) * 0.8)
    X_train, X_test = df[feature_cols].iloc[:split], df[feature_cols].iloc[split:]
    y_train, y_test = df[target_col].iloc[:split], df[target_col].iloc[split:]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    print(f'[Unseen Test] Out-of-sample AUC: {auc:.4f}')
