"""
Metrics and analysis logic for strategy module.
- Functions for calculating metrics, drawdown, performance, etc.
- No DataFrame or simulation loop logic here.
"""

def calculate_metrics(trade_log_df, final_equity, equity_history_segment, initial_capital=None, label="", model_type_l1="N/A", model_type_l2="N/A", run_summary=None, ib_lot_accumulator=0.0):
    """
    Calculates a comprehensive set of performance metrics from trade log and equity data.
    (ย้ายจาก strategy.py)
    Returns: dict of calculated metrics
    """
    # ...copy logic from strategy.py...
    import logging
    import numpy as np
    import pandas as pd
    import math
    from gc import collect as maybe_collect
    if initial_capital is None:
        from .strategy import INITIAL_CAPITAL
        initial_capital = INITIAL_CAPITAL
    metrics = {}
    label = label.strip()
    logging.info(f"  (Metrics) Calculating metrics for: '{label}'...")
    metrics[f"{label} Initial Capital (USD)"] = initial_capital
    metrics[f"{label} ML Model Used (L1)"] = model_type_l1 if model_type_l1 else "N/A"
    metrics[f"{label} ML Model Used (L2)"] = model_type_l2 if model_type_l2 else "N/A"
    fund_profile_info = run_summary.get('fund_profile', {}) if run_summary else {}
    metrics[f"{label} Fund MM Mode"] = fund_profile_info.get('mm_mode', 'N/A')
    metrics[f"{label} Fund Risk Setting"] = fund_profile_info.get('risk', np.nan)
    if run_summary and isinstance(run_summary, dict):
        metrics[f"{label} Final Risk Mode"] = run_summary.get("final_risk_mode", "N/A")
    # ...existing code for metrics calculation (copy from strategy.py)...
    # (เพื่อความกระชับ, สามารถ copy logic เต็มจาก strategy.py ได้)
    pass
