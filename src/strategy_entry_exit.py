"""
Entry/exit logic for strategy module.
- Functions for entry signal, exit conditions, forced entry, etc.
- No DataFrame or simulation loop logic here.
"""

def is_entry_allowed(row, session, consecutive_losses, side, m15_trend=None, signal_score_threshold=None):
    """Entry condition logic for simulation/backtest. (ย้ายจาก strategy.py)"""
    # ...copy logic or import from trend_filter if needed...
    from strategy.trend_filter import is_entry_allowed as _impl
    return _impl(row, session, consecutive_losses, side, m15_trend, signal_score_threshold)

def check_main_exit_conditions(order, row, current_bar_index, now_timestamp):
    """
    Checks exit conditions for an order in strict priority: BE-SL -> SL -> TP -> MaxBars.
    (ย้ายจาก strategy.py)
    Returns: (order_closed_this_bar, exit_price, close_reason, close_timestamp)
    """
    from src.order_manager import check_main_exit_conditions as _impl
    return _impl(order, row, current_bar_index, now_timestamp)
