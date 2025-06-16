# modeling.py
# ฟังก์ชันเกี่ยวกับ Model & Ensemble

def run_stacking_ensemble():
    """Stacking/Blending หลายโมเดล (meta-model)"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from ProjectP import ensure_super_features_file, get_feature_target_columns
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_pred)
    print(f'[Stacking] Stacking ensemble AUC: {auc:.4f}')

def run_automl():
    """AutoML integration (FLAML)"""
    import pandas as pd
    try:
        from flaml import AutoML
        from sklearn.metrics import roc_auc_score
        from ProjectP import ensure_super_features_file, get_feature_target_columns
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        feature_cols, target_col = get_feature_target_columns(df)
        X = df[feature_cols]
        y = df[target_col]
        automl = AutoML()
        automl.fit(X, y, task='classification', time_budget=60)
        y_pred = automl.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_pred)
        print(f'[AutoML] FLAML AUC: {auc:.4f}')
    except ImportError:
        print('[AutoML] ไม่พบ flaml ข้ามขั้นตอนนี้')

def run_deep_learning():
    """Deep Learning (LSTM) สำหรับ time series"""
    try:
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        from ProjectP import ensure_super_features_file, get_feature_target_columns
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        feature_cols, target_col = get_feature_target_columns(df)
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        X = torch.tensor(X)
        y = torch.tensor(y)
        class LSTMModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, 16, batch_first=True)
                self.fc = nn.Linear(16, 1)
            def forward(self, x):
                x, _ = self.lstm(x)
                x = self.fc(x[:,-1,:])
                return torch.sigmoid(x)
        model = LSTMModel(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()
        X_seq = X.unsqueeze(1).repeat(1,5,1)  # dummy sequence
        for epoch in range(3):
            optimizer.zero_grad()
            out = model(X_seq)
            loss = loss_fn(out.squeeze(), y)
            loss.backward()
            optimizer.step()
        print('[DL] LSTM training (demo) เสร็จสิ้น')
    except ImportError:
        print('[DL] ไม่พบ torch ข้ามขั้นตอนนี้')

def estimate_model_uncertainty():
    """Model uncertainty estimation (quantile regression)"""
    try:
        import pandas as pd
        import lightgbm as lgb
        from ProjectP import ensure_super_features_file, get_feature_target_columns
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        feature_cols, target_col = get_feature_target_columns(df)
        X = df[feature_cols]
        y = df[target_col]
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.9)
        model.fit(X, y)
        preds = model.predict(X)
        print(f'[Uncertainty] Quantile regression (90th percentile) ตัวอย่าง: {preds[:5]}')
    except ImportError:
        print('[Uncertainty] ไม่พบ lightgbm ข้ามขั้นตอนนี้')
