import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# NOTE: ต้องแน่ใจว่า ensure_super_features_file, get_feature_target_columns ถูก import หรือ define stub
try:
    from .preprocess import ensure_super_features_file, get_feature_target_columns
except ImportError:
    def ensure_super_features_file():
        raise NotImplementedError('ensure_super_features_file must be implemented or imported!')
    def get_feature_target_columns(df):
        raise NotImplementedError('get_feature_target_columns must be implemented or imported!')

def check_data_leakage(df, target_col):
    suspicious = []
    # ตรวจเฉพาะ feature ที่ไม่ใช่ target จริง และไม่ใช่ 'target' (dummy)
    for col in df.columns:
        if col == target_col or col == 'target':
            continue
        try:
            corr = abs(np.corrcoef(df[col], df[target_col])[0,1]) if df[col].dtype != 'O' else 0
        except Exception:
            corr = 0
        if corr > 0.95:
            suspicious.append(col)
    if suspicious:
        print(f"[Leakage] พบ feature ที่อาจรั่วข้อมูลอนาคต: {suspicious}")
        return True
    return False

# ฟังก์ชันใหม่: split ข้อมูล train/val/test (stratified, reproducible)
def split_train_val_test(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ปรับ train_and_validate_model ให้รองรับ test set และคืนผล test prediction/metrics
def train_validate_test_model():
    """Train, validate, and test model. Return metrics and predictions for all sets."""
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    print(f"[DEBUG][train] shape X: {X.shape}, y: {y.shape}, target unique: {np.unique(y)}")
    if len(np.unique(y)) == 1:
        print(f"[STOP][train] Target มีค่าเดียว: {np.unique(y)} หยุด pipeline")
        sys.exit(1)
    assert not check_data_leakage(df, target_col), "[STOP] พบ feature leakage ในข้อมูล!"
    # Balance class: oversample minority class (RandomOverSampler) + class_weight
    from sklearn.utils.class_weight import compute_class_weight
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print(f"[Balance] Oversampled: {np.bincount(y_res)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
    class_weight_dict = {k: v for k, v in zip(np.unique(y_res), class_weights)}
    # Split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X_res, y_res, test_size=0.15, val_size=0.15, random_state=42)
    # AutoML: ลองหลายโมเดล/parameter (ใช้ validation set)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.model_selection import ParameterGrid
    model_grid = [
        ('RandomForest', RandomForestClassifier, {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
        ('LogisticRegression', LogisticRegression, {'C': [0.1, 1.0], 'max_iter': [200]}),
        ('XGBoost', XGBClassifier, {'n_estimators': [100], 'max_depth': [5], 'use_label_encoder': [False], 'eval_metric': ['logloss']})
    ]
    best_score = -float('inf')
    best_model = None
    best_params = None
    for name, Model, param_grid in model_grid:
        for params in ParameterGrid(param_grid):
            print(f"[AutoML] ลองโมเดล {name} params={params}")
            model = Model(**params)
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_pred_val = model.predict_proba(X_val)[:,1]
            else:
                y_pred_val = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred_val)
            print(f"[AutoML] {name} val AUC: {auc:.4f}")
            if auc > best_score:
                best_score = auc
                best_model = Model(**params)
                best_params = params
    # Retrain best model on train+val, test on test set
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    best_model.fit(X_trainval, y_trainval)
    if hasattr(best_model, 'predict_proba'):
        y_pred_test_proba = best_model.predict_proba(X_test)[:,1]
        y_pred_test = (y_pred_test_proba > 0.5).astype(int)
    else:
        y_pred_test = best_model.predict(X_test)
        y_pred_test_proba = y_pred_test
    test_auc = roc_auc_score(y_test, y_pred_test_proba)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)
    test_cm = confusion_matrix(y_test, y_pred_test)
    # Save test predictions/metrics (เพิ่ม index เดิม)
    test_pred_df = pd.DataFrame({
        'row': y_test.index if hasattr(y_test, 'index') else np.arange(len(y_test)),
        'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_test_proba
    })
    test_pred_df.to_csv('output_default/test_predictions.csv', index=False)
    test_pred_df.to_parquet('output_default/test_predictions.parquet')
    import json
    with open('output_default/test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'auc': test_auc, 'report': test_report, 'confusion_matrix': test_cm.tolist()}, f, ensure_ascii=False, indent=2)
    # Plot confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(4,4))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('output_default/test_confusion_matrix.png')
    # Plot pred_proba histogram
    plt.figure(figsize=(6,3))
    sns.histplot(y_pred_test_proba, bins=50, kde=True)
    plt.title('Test Prediction Probability Distribution (pred_proba)')
    plt.xlabel('pred_proba')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('output_default/test_pred_proba_hist.png')
    print('[plot] บันทึกกราฟ test set ที่ output_default/test_confusion_matrix.png, test_pred_proba_hist.png')
    return {
        'val_auc': best_score,
        'test_auc': test_auc,
        'test_report': test_report,
        'test_cm': test_cm,
        'test_pred_df': test_pred_df
    }
def train_and_validate_model() -> float:
    """Train and validate model (AUC, leakage, noise, overfit) with GPU support."""
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    print(f"[DEBUG][train] shape X: {X.shape}, y: {y.shape}, target unique: {np.unique(y)}")
    if len(np.unique(y)) == 1:
        print(f"[STOP][train] Target มีค่าเดียว: {np.unique(y)} หยุด pipeline")
        sys.exit(1)
    # ตรวจสอบ feature leakage (เทพ): assertion/checkpoint
    assert not check_data_leakage(df, target_col), "[STOP] พบ feature leakage ในข้อมูล!"
    # Balance class: oversample minority class (RandomOverSampler) + class_weight
    from sklearn.utils.class_weight import compute_class_weight
    from imblearn.over_sampling import RandomOverSampler
    # Oversample minority class
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print(f"[Balance] Oversampled: {np.bincount(y_res)}")
    # Compute class_weight for model
    class_weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
    class_weight_dict = {k: v for k, v in zip(np.unique(y_res), class_weights)}
    use_gpu = False
    aucs = []
    # ใช้ X_res, y_res แทน X, y ในการ train/test
    X = X_res
    y = y_res
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.metrics import roc_auc_score as cu_roc_auc_score
        use_gpu = True
        print("[GPU] ใช้ cuML RandomForestClassifier (GPU)")
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X_res):
            X_train, X_val = X_res[train_idx], X_res[val_idx]
            y_train, y_val = y_res[train_idx], y_res[val_idx]
            print(f"[DEBUG][cuML] train y unique: {np.unique(y_train)}, val y unique: {np.unique(y_val)}")
            model = cuRF(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict_proba(X_val.values)[:,1]
            print(f"[DEBUG][cuML] y_pred (proba) ตัวอย่าง: {y_pred[:5]}")
            print(f"[DEBUG][cuML] y_pred unique: {np.unique(y_pred)}")
            auc = cu_roc_auc_score(y_val.values, y_pred)
            aucs.append(float(auc))
    except ImportError:
        try:
            import xgboost as xgb
            use_gpu = True
            print("[GPU] ใช้ XGBoost (GPU)")
            tscv = TimeSeriesSplit(n_splits=5)
            for train_idx, val_idx in tscv.split(X_res):
                X_train, X_val = X_res[train_idx], X_res[val_idx]
                y_train, y_val = y_res[train_idx], y_res[val_idx]
                print(f"[DEBUG][XGB] train y unique: {np.unique(y_train)}, val y unique: {np.unique(y_val)}")
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                params = {'device': 'cuda', 'max_depth': 5, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42}
                booster = xgb.train(params, dtrain, num_boost_round=100)
                y_pred = booster.predict(dval)
                print(f"[DEBUG][XGB] y_pred (proba) ตัวอย่าง: {y_pred[:5]}")
                print(f"[DEBUG][XGB] y_pred unique: {np.unique(y_pred)}")
                auc = roc_auc_score(y_val, y_pred)
                aucs.append(auc)
        except ImportError:
            try:
                from catboost import CatBoostClassifier
                use_gpu = True
                print("[GPU] ใช้ CatBoost (GPU)")
                tscv = TimeSeriesSplit(n_splits=5)
                for train_idx, val_idx in tscv.split(X_res):
                    X_train, X_val = X_res[train_idx], X_res[val_idx]
                    y_train, y_val = y_res[train_idx], y_res[val_idx]
                    print(f"[DEBUG][CatBoost] train y unique: {np.unique(y_train)}, val y unique: {np.unique(y_val)}")
                    model = CatBoostClassifier(iterations=100, depth=5, task_type="GPU", devices='0', verbose=0, random_seed=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_val)[:,1]
                    print(f"[DEBUG][CatBoost] y_pred (proba) ตัวอย่าง: {y_pred[:5]}")
                    print(f"[DEBUG][CatBoost] y_pred unique: {np.unique(y_pred)}")
                    auc = roc_auc_score(y_val, y_pred)
                    aucs.append(auc)
            except ImportError:
                print("[CPU] ไม่พบ cuML/XGBoost/CatBoost GPU จะใช้ RandomForest (CPU)")
                tscv = TimeSeriesSplit(n_splits=5)
                for train_idx, val_idx in tscv.split(X_res):
                    X_train, X_val = X_res[train_idx], X_res[val_idx]
                    y_train, y_val = y_res[train_idx], y_res[val_idx]
                    print(f"[DEBUG][RF] train y unique: {np.unique(y_train)}, val y unique: {np.unique(y_val)}")
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_val)[:,1]
                    print(f"[DEBUG][RF] y_pred (proba) ตัวอย่าง: {y_pred[:5]}")
                    print(f"[DEBUG][RF] y_pred unique: {np.unique(y_pred)}")
                    auc = roc_auc_score(y_val, y_pred)
                    aucs.append(auc)
    mean_auc = np.mean(aucs)
    print(f"[AUC] Cross-validated AUC: {mean_auc:.4f}")
    if mean_auc < 0.65:
        print("[STOP] AUC ต่ำกว่า 0.65 หยุด pipeline เพื่อป้องกัน overfitting/noise/data leak")
        sys.exit(1)
    print(f"[PASS] AUC = {mean_auc:.4f} ผ่านเกณฑ์ 0.65+ สามารถเข้าสู่ขั้นตอนถัดไปได้แบบเทพ (GPU: {use_gpu})")
    # หลัง cross-validation: บันทึก prediction/proba ของ fold สุดท้าย (หรือจะรวมทุก fold ก็ได้)
    # ตัวอย่างนี้จะบันทึกของ fold สุดท้าย
    from .predict import save_final_predictions
    save_final_predictions('output_default', y_val, (y_pred > 0.5).astype(int), y_pred)
    # วิเคราะห์ distribution ของ pred_proba หลังเทรน (auto-plot histogram)
    import matplotlib.pyplot as plt
    import seaborn as sns
    if 'y_pred' in locals():
        plt.figure(figsize=(6,3))
        sns.histplot(y_pred, bins=50, kde=True)
        plt.title('Prediction Probability Distribution (pred_proba)')
        plt.xlabel('pred_proba')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig('output_default/pred_proba_hist_auto.png')
        print('[plot] บันทึกกราฟ pred_proba histogram ที่ output_default/pred_proba_hist_auto.png')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.model_selection import ParameterGrid
    model_grid = [
        ('RandomForest', RandomForestClassifier, {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
        ('LogisticRegression', LogisticRegression, {'C': [0.1, 1.0], 'max_iter': [200]}),
        ('XGBoost', XGBClassifier, {'n_estimators': [100], 'max_depth': [5], 'use_label_encoder': [False], 'eval_metric': ['logloss']})
    ]
    best_score = -float('inf')
    best_model = None
    best_params = None
    best_pred = None
    best_y_val = None
    for name, Model, param_grid in model_grid:
        for params in ParameterGrid(param_grid):
            print(f"[AutoML] ลองโมเดล {name} params={params}")
            tscv = TimeSeriesSplit(n_splits=3)
            aucs = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = Model(**params)
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_val)[:,1]
                else:
                    y_pred = model.predict(X_val)
                auc = roc_auc_score(y_val, y_pred)
                aucs.append(auc)
            mean_auc = np.mean(aucs)
            print(f"[AutoML] {name} mean AUC: {mean_auc:.4f}")
            if mean_auc > best_score:
                best_score = mean_auc
                best_model = Model(**params)
                best_params = params
                # retrain on all data
                best_model.fit(X, y)
                if hasattr(best_model, 'predict_proba'):
                    best_pred = best_model.predict_proba(X)[:,1]
                else:
                    best_pred = best_model.predict(X)
                best_y_val = y
    print(f"[AutoML] Best model: {best_model} params: {best_params} AUC: {best_score:.4f}")
    # Save prediction of best model
    from .predict import save_final_predictions
    save_final_predictions('output_default', best_y_val, (best_pred > 0.5).astype(int), best_pred)
    # Plot pred_proba histogram
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,3))
    sns.histplot(best_pred, bins=50, kde=True)
    plt.title('Prediction Probability Distribution (pred_proba)')
    plt.xlabel('pred_proba')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('output_default/pred_proba_hist_auto.png')
    print('[plot] บันทึกกราฟ pred_proba histogram ที่ output_default/pred_proba_hist_auto.png')
    return best_score

def walk_forward_validation_evaluate(df, feature_cols, target_col, n_splits=5, test_size=0.15, random_state=42):
    """Walk-Forward Validation (WFV) สำหรับ time series/เทรดจริง"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    n_total = len(df)
    window_size = int((1 - test_size) * n_total / n_splits)
    test_window = int(test_size * n_total / n_splits)
    results = []
    for i in range(n_splits):
        train_start = i * window_size
        train_end = train_start + window_size
        test_start = train_end
        test_end = test_start + test_window
        if test_end > n_total:
            break
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx][target_col]
        X_test, y_test = df.iloc[test_idx][feature_cols], df.iloc[test_idx][target_col]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        results.append({
            'window': i+1,
            'train_range': (int(train_start), int(train_end)),
            'test_range': (int(test_start), int(test_end)),
            'auc': float(auc),
            'report': report,
            'confusion_matrix': cm.tolist(),
            'y_true': y_test.values.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        })
    # Export summary
    aucs = [r['auc'] for r in results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    with open('output_default/wfv_results.json', 'w', encoding='utf-8') as f:
        import json
        json.dump({'results': results, 'mean_auc': float(mean_auc), 'std_auc': float(std_auc)}, f, ensure_ascii=False, indent=2)
    # Plot AUC per window
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(aucs)+1), aucs, marker='o')
    plt.title('WFV AUC per Window')
    plt.xlabel('Window')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_default/wfv_auc_per_window.png')
    print('[WFV] บันทึกผลลัพธ์ WFV ที่ output_default/wfv_results.json, wfv_auc_per_window.png')
    return results, mean_auc, std_auc

def walk_forward_validation_custom_split(df, feature_cols, target_col, split_points, model_fn=None, random_state=42):
    """WFV แบบ custom split: split_points = [(train_start, train_end, test_start, test_end), ...]"""
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(split_points):
        X_train, y_train = df.iloc[train_start:train_end][feature_cols], df.iloc[train_start:train_end][target_col]
        X_test, y_test = df.iloc[test_start:test_end][feature_cols], df.iloc[test_start:test_end][target_col]
        model = model_fn() if model_fn else __import__('sklearn.ensemble').ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        results.append({
            'window': i+1,
            'train_range': (int(train_start), int(train_end)),
            'test_range': (int(test_start), int(test_end)),
            'auc': float(auc),
            'report': report,
            'confusion_matrix': cm.tolist(),
            'y_true': y_test.values.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        })
    # Advanced plot: AUC, error, prediction dist
    aucs = [r['auc'] for r in results]
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(aucs)+1), aucs, marker='o', label='AUC')
    plt.title('Custom WFV AUC per Window')
    plt.xlabel('Window')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_default/wfv_custom_auc_per_window.png')
    # Error rate per window
    error_rates = [np.mean(np.array(r['y_true']) != np.array(r['y_pred'])) for r in results]
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(error_rates)+1), error_rates, marker='x', color='red', label='Error Rate')
    plt.title('Custom WFV Error Rate per Window')
    plt.xlabel('Window')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_default/wfv_custom_error_per_window.png')
    # Prediction distribution
    all_pred_proba = np.concatenate([r['y_pred_proba'] for r in results])
    plt.figure(figsize=(6,3))
    sns.histplot(all_pred_proba, bins=50, kde=True)
    plt.title('Custom WFV Prediction Probability Distribution')
    plt.xlabel('pred_proba')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('output_default/wfv_custom_pred_proba_hist.png')
    # Export
    import json
    with open('output_default/wfv_custom_results.json', 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, ensure_ascii=False, indent=2)
    print('[WFV] บันทึกผลลัพธ์ WFV custom split ที่ output_default/wfv_custom_results.json, wfv_custom_auc_per_window.png, wfv_custom_error_per_window.png, wfv_custom_pred_proba_hist.png')
    return results

def generate_wfv_report(wfv_results, output_path='output_default/wfv_report.txt'):
    """สร้างรายงาน WFV summary/export เป็น text"""
    import numpy as np
    aucs = [r['auc'] for r in wfv_results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('=== Walk-Forward Validation (WFV) Report ===\n')
        f.write(f'Number of windows: {len(wfv_results)}\n')
        f.write(f'Mean AUC: {mean_auc:.4f}\n')
        f.write(f'Std AUC: {std_auc:.4f}\n')
        for r in wfv_results:
            f.write(f"\nWindow {r['window']} | Train: {r['train_range']} | Test: {r['test_range']}\n")
            f.write(f"AUC: {r['auc']:.4f}\n")
            f.write(f"Confusion Matrix: {r['confusion_matrix']}\n")
    print(f'[WFV] สร้างรายงาน WFV ที่ {output_path}')
