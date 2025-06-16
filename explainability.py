# explainability.py
# ฟังก์ชันเกี่ยวกับ Explainability & Monitoring

def run_drift_detection():
    """Drift detection (concept/data drift)"""
    import pandas as pd
    try:
        from river.drift import ADWIN
        from ProjectP import ensure_super_features_file, get_feature_target_columns
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        feature_cols, _ = get_feature_target_columns(df)
        drift = ADWIN()
        drift_points = []
        for i, row in enumerate(df[feature_cols].values):
            drift.update(row[0])  # ใช้ feature แรกเป็นตัวอย่าง
            if drift.change_detected:
                drift_points.append(i)
        print(f'[Drift] พบ drift ที่ index: {drift_points}')
    except ImportError:
        print('[Drift] ไม่พบ river ข้ามขั้นตอนนี้')

def run_auto_report():
    """Auto-report generation (PDF/HTML)"""
    try:
        import pandas as pd
        from jinja2 import Template
        from ProjectP import ensure_super_features_file
        fe_super_path = ensure_super_features_file()
        df = pd.read_parquet(fe_super_path)
        html = Template('<h1>Auto Report</h1><p>Rows: {{rows}}</p>').render(rows=len(df))
        with open('output_default/auto_report.html','w',encoding='utf-8') as f:
            f.write(html)
        print('[Report] สร้าง auto_report.html สำเร็จ')
    except ImportError:
        print('[Report] ไม่พบ jinja2 ข้ามขั้นตอนนี้')

def run_backtest_visualization():
    """Backtest visualization (heatmap, rolling metrics, drawdown curve)"""
    import pandas as pd
    import matplotlib.pyplot as plt
    from ProjectP import ensure_super_features_file
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    if 'return_1' not in df:
        print('[Viz] ไม่พบ return_1 ในข้อมูล')
        return
    eq_curve = (1 + df['return_1']).cumprod()
    roll_max = eq_curve.cummax()
    drawdown = (eq_curve - roll_max) / roll_max
    plt.figure(figsize=(10,4))
    plt.subplot(2,1,1)
    plt.plot(eq_curve, label='Equity Curve')
    plt.title('Equity Curve')
    plt.subplot(2,1,2)
    plt.plot(drawdown, label='Drawdown')
    plt.title('Drawdown')
    plt.tight_layout()
    plt.savefig('output_default/backtest_viz.png')
    print('[Viz] บันทึกกราฟ backtest_viz.png')
