"""Simple dashboard generator using pandas HTML export."""

from __future__ import annotations

import os
import logging
from typing import Any

import plotly.graph_objects as go

import pandas as pd

logger = logging.getLogger(__name__)


def generate_dashboard(results: Any, output_filepath: str) -> str:
    """Generate an interactive HTML dashboard.

    Parameters
    ----------
    results : Any
        DataFrame or path to CSV file containing metrics.
    output_filepath : str
        Destination HTML filepath.
    """
    # [Patch v6.6.7] implement interactive chart generation
    if isinstance(results, str):
        if os.path.exists(results):
            from src.utils.data_utils import safe_read_csv

            results = safe_read_csv(results)
        else:
            logger.error("Results path %s not found", results)
            results = pd.DataFrame()
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    numeric_cols = results.select_dtypes(include="number").columns
    fig = go.Figure()
    if "fold" in results.columns and "test_pnl" in numeric_cols:
        fig.add_bar(x=results["fold"], y=results["test_pnl"], name="Test PnL")
        if "train_pnl" in numeric_cols:
            fig.add_bar(x=results["fold"], y=results["train_pnl"], name="Train PnL")
    elif len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig.add_scatter(x=list(range(len(results))), y=results[col], name=col)
    fig.update_layout(title="Metrics Summary", height=500, width=700)

    table_html = results.to_html(index=False)
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as fh:
        fh.write(
            f"<html><head><meta charset='utf-8'></head><body>{fig_html}<hr>{table_html}</body></html>"
        )

    logger.info("[Patch v6.6.7] Dashboard saved to %s", output_filepath)
    # [เทพ] เพิ่ม plot distribution/imbalance/feature leakage
    output_dir = os.path.dirname(output_filepath)
    if isinstance(results, pd.DataFrame) and not results.empty:
        # ลองหาไฟล์ prediction/parquet ที่ output_dir
        import glob
        import pandas as pd
        pred_files = glob.glob(os.path.join(output_dir, '*_strategy_result.parquet'))
        if pred_files:
            pred_file = max(pred_files, key=os.path.getmtime)
            try:
                df_pred = pd.read_parquet(pred_file)
                dist_path = plot_target_distribution(df_pred, output_dir)
                leakage_path = plot_feature_leakage_analysis(df_pred, output_dir)
            except Exception as e:
                logger.warning(f'[Dashboard] ไม่สามารถ plot distribution/leakage: {e}')
    return output_filepath

# [Patch v7.0.0] Underwater plot utility

def plot_underwater_curve(equity: pd.Series) -> go.Figure:
    """Return Plotly Figure of underwater drawdown curve."""
    if equity is None or equity.empty:
        raise ValueError("equity series is empty")
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    running_max = equity.cummax()
    underwater = (equity - running_max) / running_max
    fig = go.Figure()
    fig.add_scatter(x=equity.index, y=underwater, name="Underwater")
    fig.update_layout(title="Underwater Plot", height=400, width=700)
    return fig

def plot_target_distribution(df, output_dir):
    import matplotlib.pyplot as plt
    import os
    if 'target' in df.columns:
        counts = df['target'].value_counts().sort_index()
        plt.figure()
        counts.plot(kind='bar')
        plt.title('Target Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        out_path = os.path.join(output_dir, 'target_distribution_dashboard.png')
        plt.savefig(out_path)
        plt.close()
        logger.info(f'[Dist] บันทึกกราฟ distribution ที่ {out_path}')
        return out_path
    return None

def plot_feature_leakage_analysis(df, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    if 'target' not in df.columns:
        return None
    corrs = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    corrs.head(20).plot(kind='bar')
    plt.title('Top 20 Feature Correlations with Target')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'feature_leakage_analysis_dashboard.png')
    plt.savefig(out_path)
    plt.close()
    logger.info(f'[Leakage] บันทึกกราฟ correlation ที่ {out_path}')
    suspicious = corrs[corrs > 0.95]
    if not suspicious.empty:
        logger.warning(f'[Leakage] พบ feature ที่อาจรั่วข้อมูลอนาคต: {list(suspicious.index)}')
    return out_path

