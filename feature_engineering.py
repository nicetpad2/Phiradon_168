# feature_engineering.py
# ฟังก์ชันเกี่ยวกับ Feature Engineering & Selection

def run_auto_feature_generation():
    """Auto Feature Generation (featuretools)"""
    import pandas as pd
    try:
        import featuretools as ft
        from ProjectP import ensure_super_features_file
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        es = ft.EntitySet(id='data')
        es = es.add_dataframe(dataframe_name='main', dataframe=df, index=None)
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='main', max_depth=2, verbose=True)
        feature_matrix.to_parquet('output_default/ft_auto_features.parquet')
        print('[Featuretools] สร้าง auto features และบันทึกที่ output_default/ft_auto_features.parquet')
    except ImportError:
        print('[Featuretools] ไม่พบ featuretools ข้ามขั้นตอนนี้')

def run_feature_interaction():
    """Feature interaction/combination (pairwise, polynomial, ratio)"""
    import pandas as pd
    import numpy as np
    from ProjectP import ensure_super_features_file, get_feature_target_columns
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, _ = get_feature_target_columns(df)
    # ตัวอย่าง: สร้าง pairwise product และ ratio ของ top 3 features
    top3 = feature_cols[:3]
    for i in range(len(top3)):
        for j in range(i+1, len(top3)):
            f1, f2 = top3[i], top3[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
            df[f'{f1}_div_{f2}'] = df[f1] / (df[f2]+1e-6)
    df.to_parquet('output_default/feature_interaction.parquet')
    print('[Feature Interaction] สร้าง pairwise product/ratio และบันทึกที่ output_default/feature_interaction.parquet')

def run_rfe_with_shap():
    """Recursive Feature Elimination (RFE) ร่วมกับ SHAP"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    try:
        import shap
        from ProjectP import ensure_super_features_file, get_feature_target_columns
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        feature_cols, target_col = get_feature_target_columns(df)
        X = df[feature_cols]
        y = df[target_col]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rfe = RFE(model, n_features_to_select=10)
        rfe.fit(X, y)
        selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
        print(f'[RFE+SHAP] Top 10 features by RFE: {selected}')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig('output_default/rfe_shap_summary.png')
        print('[RFE+SHAP] บันทึก SHAP summary plot ที่ output_default/rfe_shap_summary.png')
    except ImportError:
        print('[RFE+SHAP] ไม่พบ shap ข้ามขั้นตอนนี้')

def remove_multicollinearity():
    """Auto-detect & remove multicollinearity"""
    import pandas as pd
    import numpy as np
    from ProjectP import ensure_super_features_file, get_feature_target_columns
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, _ = get_feature_target_columns(df)
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f'[Multicollinearity] Features to drop (corr > 0.95): {to_drop}')
