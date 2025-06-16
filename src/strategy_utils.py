"""
Utility/stateless functions for strategy module.
- Use for helpers: e.g. get_adaptive_tsl_step, dynamic_tp2_multiplier, etc.
- No stateful/order logic here.
"""

def dynamic_tp2_multiplier(current_atr, avg_atr, base=None):
    """Calculates a dynamic TP multiplier based on current vs average ATR."""
    if base is None:
        from .strategy import BASE_TP_MULTIPLIER
        base = BASE_TP_MULTIPLIER
    import pandas as pd
    import numpy as np
    current_atr_num = pd.to_numeric(current_atr, errors='coerce')
    avg_atr_num = pd.to_numeric(avg_atr, errors='coerce')
    if pd.isna(current_atr_num) or pd.isna(avg_atr_num) or np.isinf(current_atr_num) or np.isinf(avg_atr_num) or avg_atr_num < 1e-9:
        return base
    try:
        from .strategy import ADAPTIVE_TSL_HIGH_VOL_RATIO
        ratio = current_atr_num / avg_atr_num
        high_vol_ratio = ADAPTIVE_TSL_HIGH_VOL_RATIO
        high_vol_adjust = 0.6
        mid_vol_ratio = 1.2
        mid_vol_adjust = 0.3
        if ratio >= high_vol_ratio:
            return base + high_vol_adjust
        elif ratio >= mid_vol_ratio:
            return base + mid_vol_adjust
        else:
            return base
    except Exception:
        return base

def get_adaptive_tsl_step(current_atr, avg_atr, default_step=None):
    """Determines the TSL step size (in R units) based on volatility."""
    from .strategy import ADAPTIVE_TSL_DEFAULT_STEP_R, ADAPTIVE_TSL_HIGH_VOL_RATIO, ADAPTIVE_TSL_HIGH_VOL_STEP_R, ADAPTIVE_TSL_LOW_VOL_RATIO, ADAPTIVE_TSL_LOW_VOL_STEP_R
    import pandas as pd
    import numpy as np
    if default_step is None:
        default_step = ADAPTIVE_TSL_DEFAULT_STEP_R
    high_vol_ratio = ADAPTIVE_TSL_HIGH_VOL_RATIO
    high_vol_step = ADAPTIVE_TSL_HIGH_VOL_STEP_R
    low_vol_ratio = ADAPTIVE_TSL_LOW_VOL_RATIO
    low_vol_step = ADAPTIVE_TSL_LOW_VOL_STEP_R
    current_atr_num = pd.to_numeric(current_atr, errors='coerce')
    avg_atr_num = pd.to_numeric(avg_atr, errors='coerce')
    if pd.isna(current_atr_num) or pd.isna(avg_atr_num) or np.isinf(current_atr_num) or np.isinf(avg_atr_num) or avg_atr_num < 1e-9:
        return default_step
    try:
        ratio = current_atr_num / avg_atr_num
        if ratio > high_vol_ratio:
            return high_vol_step
        elif ratio < low_vol_ratio:
            return low_vol_step
        else:
            return default_step
    except Exception:
        return default_step

# TODO: Move more helpers from strategy.py
