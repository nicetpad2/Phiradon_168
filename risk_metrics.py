# risk_metrics.py
# ฟังก์ชันเกี่ยวกับ Risk & Performance Metrics

def calc_risk_metrics():
    """วัด risk-adjusted return (Sharpe, Sortino, Max Drawdown)"""
    import pandas as pd
    import numpy as np
    from ProjectP import ensure_super_features_file
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    if 'return_1' not in df:
        print('[Risk] ไม่พบ return_1 ในข้อมูล')
        return
    returns = df['return_1']
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252*24*12)  # สมมติเป็น M1
    sortino = returns.mean() / (returns[returns<0].std() + 1e-8) * np.sqrt(252*24*12)
    eq_curve = (1 + returns).cumprod()
    roll_max = eq_curve.cummax()
    drawdown = (eq_curve - roll_max) / roll_max
    max_dd = drawdown.min()
    print(f'[Risk] Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Max Drawdown: {max_dd:.2%}')

def simulate_transaction_cost():
    """Transaction cost & slippage simulation"""
    import pandas as pd
    from ProjectP import ensure_super_features_file
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    if 'return_1' not in df:
        print('[TCost] ไม่พบ return_1 ในข้อมูล')
        return
    cost_per_trade = 0.0002  # 2 pip
    n_trades = len(df)
    gross_return = (1 + df['return_1']).prod() - 1
    net_return = (1 + df['return_1'] - cost_per_trade).prod() - 1
    print(f'[TCost] Gross return: {gross_return:.4f} | Net after cost: {net_return:.4f} | Total cost: {(gross_return-net_return):.4f}')

def run_position_sizing():
    """Position sizing & money management"""
    import pandas as pd
    from ProjectP import ensure_super_features_file
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    if 'return_1' not in df:
        print('[Position] ไม่พบ return_1 ในข้อมูล')
        return
    # ตัวอย่าง: Kelly formula
    win_rate = (df['return_1'] > 0).mean()
    win_loss = df['return_1'][df['return_1'] > 0].mean() / abs(df['return_1'][df['return_1'] <= 0].mean() + 1e-8)
    kelly = win_rate - (1 - win_rate) / win_loss if win_loss > 0 else 0
    print(f'[Position] Kelly position size: {kelly:.2%}')
