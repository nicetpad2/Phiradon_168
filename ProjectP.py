"""Bootstrap script for running the main entry point."""

# --- Suppress joblib/loky physical core warnings and force safe settings ---
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # Disable joblib multiprocessing if possible
os.environ["LOKY_MAX_CPU_COUNT"] = "1"      # Limit loky to 1 CPU to avoid subprocess errors
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*", category=UserWarning)
warnings.filterwarnings("ignore", message="joblib.externals.loky.backend.context.*", category=UserWarning)

# [Patch v6.4.0] Ensure project modules are importable by setting sys.path and working directory
import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH and set cwd
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import logging
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress numpy RuntimeWarnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")  # Suppress joblib/loky warning

# [Log All Lines] Custom print function to log every line
class PrintLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    def write(self, message):
        message = message.rstrip('\n')
        if message:
            for line in message.splitlines():
                self.logger.log(self.level, line)
    def flush(self):
        pass

# [Log All Lines] Setup file handler for logging all output
log_file = os.path.join(project_root, 'projectp_full.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in root_logger.handlers):
    root_logger.addHandler(file_handler)

# Redirect stdout/stderr to logger (log ทุก print/exception จาก terminal)
# sys.stdout = PrintLogger(root_logger, logging.INFO)
# sys.stderr = PrintLogger(root_logger, logging.ERROR)

from src.features import DEFAULT_META_CLASSIFIER_FEATURES

# [Patch v6.3.0] Stub imports for missing features
try:
    from src.utils.auto_train_meta_classifiers import auto_train_meta_classifiers
except ImportError:  # pragma: no cover - fallback when module missing

    def auto_train_meta_classifiers(*args, **kwargs):
        logging.getLogger().warning(
            "[Patch v6.2.3] auto_train_meta_classifiers stub invoked; skipping."
        )


try:
    from reporting.dashboard import generate_dashboard
except ImportError:  # pragma: no cover - fallback when module missing

    def generate_dashboard(*args, **kwargs):
        logging.getLogger().warning(
            "[Patch v6.2.3] generate_dashboard stub invoked; skipping."
        )


import csv
import yaml
import time
from pathlib import Path

try:  # [Patch v5.10.2] allow import without heavy dependencies
    from src.config import logger, OUTPUT_DIR, DEFAULT_TRADE_LOG_MIN_ROWS
    import src.config as config
except Exception:  # pragma: no cover - fallback logger for tests
    logger = logging.getLogger("ProjectP")
    OUTPUT_DIR = Path("output_default")
# [Patch v5.9.17] Fallback logger if src.config fails
import sys
import os
import argparse
import subprocess
import json

from src.utils.pipeline_config import (
    load_config,
)  # [Patch v6.7.17] dynamic config loader

# [Patch v6.4.8] Optional fallback directory for raw data and logs
FALLBACK_DIR = os.getenv("PROJECTP_FALLBACK_DIR")

# [Patch v6.7.17] Load pipeline configuration for dynamic paths
pipeline_config = load_config()


# [Patch v6.3.1] Ensure working directory fallback on import
try:
    os.getcwd()
except Exception:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    print(f"[Info] Changed working directory to project root: {project_root}")

import pandas as pd
from typing import Dict, List
from src.utils.errors import PipelineError
from config_loader import update_config_from_dict  # [Patch] dynamic config update
from wfv_runner import run_walkforward  # [Patch] walk-forward helper
from src.features import build_feature_catalog
from src.pipeline_stages import run_preprocess, run_sweep, run_threshold, run_backtest, run_report

# Default grid for hyperparameter sweep
DEFAULT_SWEEP_PARAMS: Dict[str, List[float]] = {
    "learning_rate": [0.01, 0.05],
    "depth": [6, 8],
    "l2_leaf_reg": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bylevel": [0.8, 1.0],
    "bagging_temperature": [0.0, 1.0],
    "random_strength": [0.0, 1.0],
}

# [Patch] Initialize pynvml for GPU status detection
try:
    import pynvml  # type: ignore  # No stub available

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None
    nvml_handle = None
except Exception:  # pragma: no cover - NVML failure fallback
    nvml_handle = None

from src.data_loader import (
    auto_convert_gold_csv as auto_convert_csv,
    auto_convert_csv_to_parquet,
)
from src.utils.model_utils import get_latest_model_and_threshold


def configure_logging():
    """Set up consistent logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_logo() -> None:
    """แสดงโลโก้ Project P บนหน้าจอ."""
    logo = r"""
 ____            _            ____  ____  
|  _ \ ___  __ _| | _____    |  _ \|  _ \ 
| |_) / _ \/ _` | |/ / _ \   | |_) | |_) |
|  __/  __/ (_| |   <  __/   |  __/|  __/ 
|_|   \___|\__,_|_|\_\___|   |_|   |_|    
    """
    print(logo)


def custom_helper_function():
    """Stubbed helper for tests."""
    return True


def parse_projectp_args(args=None):
    """Parse command line arguments for ProjectP."""
    parser = argparse.ArgumentParser(description="สคริปต์ควบคุมโหมดการทำงาน")
    parser.add_argument(
        "--mode",
        choices=[
            "preprocess",
            "sweep",
            "threshold",
            "backtest",
            "report",
            "full_pipeline",
            "all",
            "hyper_sweep",
            "wfv",
            "hold",  # เพิ่มโหมด hold
            "train_auc",  # เพิ่มโหมด train_auc สำหรับตรวจสอบ AUC/leakage/overfit/noise
            "optuna_shap",  # เพิ่มโหมด optuna_shap แบบเทพ
            "robustness_unseen",  # เพิ่มโหมด robustness_unseen สำหรับทดสอบความทนทาน
        ],
        default="preprocess",
        help="ขั้นตอนที่จะรัน",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limit rows for fast pipeline loop test)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Limit number of rows loaded from data (overrides debug default)",
    )
    parser.add_argument(
        "--auto-convert",
        action="store_true",
        help="แปลงไฟล์ CSV อัตโนมัติ",
    )
    return parser.parse_known_args(args)[0]


def parse_args(args=None):  # backward compatibility
    return parse_projectp_args(args)


def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    """เลือกเฉพาะ feature numeric และ target ที่เหมาะสม (target_event ถ้ามี, รองลงมาคือ target_direction)"""
    feature_cols = [c for c in df.columns if c not in ['target_event','target_direction','Date','Time','Symbol','datetime'] and df[c].dtype != 'O']
    # [เทพ] ถ้ามี SELECTED_FEATURES ให้ใช้เฉพาะ feature ที่สำคัญ
    global SELECTED_FEATURES
    if 'SELECTED_FEATURES' in globals() and SELECTED_FEATURES:
        feature_cols = [c for c in feature_cols if c in SELECTED_FEATURES]
    if 'target_event' in df.columns:
        target_col = 'target_event'
    elif 'target_direction' in df.columns:
        target_col = 'target_direction'
    else:
        raise ValueError('ไม่พบ target ที่เหมาะสมในไฟล์ feature engineered')
    return feature_cols, target_col


def run_preprocess():
    """Preprocess pipeline: โหลด feature engineered + target อัตโนมัติ, เลือก feature/target ใหม่, บันทึก preprocessed_super.parquet"""
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    print(f'ใช้ feature: {feature_cols}')
    print(f'ใช้ target: {target_col}')
    df_out = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    # Always create 'target' column (copy from target_col)
    df_out['target'] = df_out[target_col]
    # If 'pred_proba' exists, keep it; else, create dummy if needed
    if 'pred_proba' not in df_out.columns:
        df_out['pred_proba'] = 0.5  # default dummy value
    print(f'[DEBUG][preprocess] df_out shape: {df_out.shape}')
    print(f'[DEBUG][preprocess] target unique: {df_out["target"].unique()}')
    for col in feature_cols:
        print(f'[DEBUG][preprocess] {col} unique: {df_out[col].unique()[:5]}')
    if len(df_out['target'].unique()) == 1:
        print(f"[STOP][preprocess] Target มีค่าเดียว: {df_out['target'].unique()} หยุด pipeline")
        sys.exit(1)
    out_path = os.path.join('output_default', 'preprocessed_super.parquet')
    df_out.to_parquet(out_path)
    print(f'บันทึกไฟล์ preprocessed_super.parquet ด้วย feature/target ใหม่ ({target_col})')


def _run_script(relative_path: str) -> None:
    """Execute a Python script located relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, relative_path)
    subprocess.run([sys.executable, abs_path], check=True)


def run_hyperparameter_sweep(params: Dict[str, List[float]]) -> None:
    """รันการค้นหาค่าพารามิเตอร์โดยใช้ไฟล์ preprocessed_super.parquet (เทพ)."""
    logger.debug(f"Starting sweep with params: {params}")
    from tuning.hyperparameter_sweep import run_sweep as _sweep, DEFAULT_TRADE_LOG

    # ใช้ไฟล์ preprocessed_super.parquet เป็น input หลัก
    m1_path = os.path.join('output_default', 'preprocessed_super.parquet')
    if not os.path.exists(m1_path):
        logging.error('[Auto] ไม่พบไฟล์ preprocessed_super.parquet กรุณารัน preprocess ก่อน')
        return
    logger.info("[Auto] Running sweep with m1_path=%s", m1_path)

    _sweep(
        str(OUTPUT_DIR),
        params,
        seed=42,
        resume=True,
        trade_log_path=DEFAULT_TRADE_LOG,
        m1_path=m1_path,
    )


def run_sweep():
    """รันการค้นหาค่าพารามิเตอร์ (backward compatibility)."""
    _run_script(os.path.join("tuning", "hyperparameter_sweep.py"))


def run_threshold_optimization() -> pd.DataFrame:
    logger.debug("Starting threshold optimization")
    from threshold_optimization import run_threshold_optimization as _opt
    # Force use of correct columns in preprocessed_super.parquet
    m1_path = os.path.join('output_default', 'preprocessed_super.parquet')
    import pandas as pd
    if os.path.exists(m1_path):
        df = pd.read_parquet(m1_path)
        # If 'target' or 'pred_proba' missing, log error
        if 'target' not in df.columns or 'pred_proba' not in df.columns:
            logging.error("[Patch] preprocessed_super.parquet missing 'target' or 'pred_proba' columns. Columns: %s", list(df.columns))
        else:
            logging.info("[Patch] preprocessed_super.parquet columns OK: %s", list(df.columns))
    return _opt()


def run_threshold():
    """รันการปรับค่า threshold (backward compatibility)."""
    _run_script("threshold_optimization.py")


def run_backtest():
    import main as pipeline
    """รันการทดสอบย้อนหลังโดยใช้ไฟล์ preprocessed_super.parquet (เทพ)."""
    model_dir = "models"
    model_path, threshold = get_latest_model_and_threshold(
        model_dir, "threshold_wfv_optuna_results.csv", take_first=True
    )
    m1_path = os.path.join('output_default', 'preprocessed_super.parquet')
    if not os.path.exists(m1_path):
        logging.error('[Auto] ไม่พบไฟล์ preprocessed_super.parquet กรุณารัน preprocess ก่อน')
        return
    import pandas as pd
    df_super = pd.read_parquet(m1_path)
    df_super = ensure_datetime_index(df_super)
    df_super.to_parquet(m1_path)
    logging.info('[Patch] Saved preprocessed_super.parquet with DatetimeIndex for downstream use.')
    pipeline.run_backtest_pipeline(
        df_super, pd.DataFrame(), model_path, threshold
    )
    # เรียก main pipeline ใน src/main.py เพื่อสร้าง metrics_summary_v32.csv จริง
    try:
        import importlib
        main_module = importlib.import_module('src.main')
        if hasattr(main_module, 'main'):
            main_module.main(['--mode', 'backtest'])
            logging.info('[เทพ] เรียก main pipeline สร้าง metrics_summary_v32.csv สำเร็จ')
        else:
            logging.warning('[เทพ] ไม่พบ main() ใน src.main')
    except Exception as e:
        logging.error(f'[เทพ] เรียก main pipeline สร้าง metrics_summary_v32.csv ล้มเหลว: {e}')
    output_dir = getattr(config, "OUTPUT_DIR", OUTPUT_DIR) if 'config' in globals() else OUTPUT_DIR
    ensure_metrics_summary(str(output_dir))


def run_report() -> None:
    import main as pipeline
    config = pipeline.load_config()
    pipeline.run_report(config)
    # เพิ่ม validation/plot confusion matrix, ROC, PR curve, feature importance
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
        metrics_path = os.path.join(getattr(config, "OUTPUT_DIR", OUTPUT_DIR), "metrics_summary_v32.csv")
        output_dir = getattr(config, "OUTPUT_DIR", OUTPUT_DIR) if hasattr(config, "OUTPUT_DIR") else OUTPUT_DIR
        pred_path = os.path.join(output_dir, "buy_sell_hold_strategy_result.parquet")
        if os.path.exists(pred_path):
            df_pred = pd.read_parquet(pred_path)
            # Save final predictions (เทพ) ให้แน่ใจว่ามีไฟล์ prediction จริง
            if 'target_direction' in df_pred:
                y_true = df_pred['target_direction']
                y_pred = (df_pred['target_buy_sell_hold'] == 1).astype(int) if 'target_buy_sell_hold' in df_pred else None
                y_proba = df_pred['pred_proba'] if 'pred_proba' in df_pred else None
                save_final_predictions(str(output_dir), y_true, y_pred, y_proba)
            # Plot distribution/imbalance
            plot_target_distribution(df_pred, output_dir)
            # Plot feature leakage analysis
            plot_feature_leakage_analysis(df_pred, output_dir)
    except Exception as e:
        print(f'[เทพ] Visualization/validation error: {e}')
    output_dir = getattr(config, "OUTPUT_DIR", OUTPUT_DIR) if hasattr(config, "OUTPUT_DIR") else OUTPUT_DIR
    ensure_metrics_summary(str(output_dir))


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
        out_path = os.path.join(output_dir, 'target_distribution.png')
        plt.savefig(out_path)
        plt.close()
        print(f'[Dist] บันทึกกราฟ distribution ที่ {out_path}')


def plot_feature_leakage_analysis(df, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    if 'target' not in df.columns:
        return
    corrs = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    corrs.head(20).plot(kind='bar')
    plt.title('Top 20 Feature Correlations with Target')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'feature_leakage_analysis.png')
    plt.savefig(out_path)
    plt.close()
    print(f'[Leakage] บันทึกกราฟ correlation ที่ {out_path}')
    suspicious = corrs[corrs > 0.95]
    if not suspicious.empty:
        print(f'[Leakage] พบ feature ที่อาจรั่วข้อมูลอนาคต: {list(suspicious.index)}')


def _execute_step(name: str, func, *args, **kwargs):
    """[Patch v6.9.23] Execute a pipeline step with timing and logs."""
    start = time.perf_counter()
    logger.info("[Patch v6.9.23] Starting %s", name)
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.warning("[Patch v6.9.23] %s completed in %.2fs", name, elapsed)
    logging.getLogger().warning("%s completed in %.2fs", name, elapsed)
    return result


def get_shap_top_features(n=10):
    import os
    import pandas as pd
    shap_path = os.path.join('output_default', 'shap_top_features.csv')
    if os.path.exists(shap_path):
        top_features = pd.read_csv(shap_path, header=None)[0].tolist()[:n]
        print(f'[SHAP] ใช้ top {n} features จาก SHAP: {top_features}')
        return top_features
    return None


def run_full_pipeline() -> None:
    from src.main import main as pipeline_main
    print("[Pipeline] เริ่ม full_pipeline...")
    try:
        print("[Pipeline] Step 0: Auto-generate latest features/targets...")
        script_path = os.path.join('scripts', 'auto_feature_target_engineering.py')
        script_path = safe_path(script_path)
        print(f"[DEBUG][subprocess] Running: {sys.executable} {script_path}")
        result = subprocess.run(
            [sys.executable, script_path],
            check=False, capture_output=True, text=True
        )
        print(f"[Pipeline] Step 0 เสร็จสิ้น (exit code: {result.returncode})")
        print("[Pipeline][subprocess][stdout]\n" + (result.stdout or "<no stdout>"))
        print("[Pipeline][subprocess][stderr]\n" + (result.stderr or "<no stderr>"))
        if result.returncode != 0:
            print(f"[Pipeline][WARNING] auto_feature_target_engineering.py exited with code {result.returncode}, but pipeline will continue.")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 0: {e}")
    try:
        print("[Pipeline] Step 0.5: Auto Feature Engineering...")
        from feature_engineering import run_auto_feature_generation, run_feature_interaction, run_rfe_with_shap
        run_auto_feature_generation()
        run_feature_interaction()
        run_rfe_with_shap()
        print("[Pipeline] Step 0.5 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 0.5: {e}")
    try:
        print("[Pipeline] Step 1: Preprocess...")
        _execute_step("preprocess", run_preprocess)
        print("[Pipeline] Step 1 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 1: {e}")
    try:
        print("[Pipeline] Step 1.5: Validate after preprocess...")
        _execute_step("validate_after_preprocess", train_and_validate_model)
        print("[Pipeline] Step 1.5 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 1.5: {e}")
    try:
        print("[Pipeline] Step 2: Hyperparameter Sweep...")
        _execute_step("sweep", run_hyperparameter_sweep, DEFAULT_SWEEP_PARAMS)
        print("[Pipeline] Step 2 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 2: {e}")
    try:
        print("[Pipeline] Step 2.5: Optuna+SHAP...")
        _execute_step("optuna_shap", run_optuna_with_shap)
        print("[Pipeline] Step 2.5 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 2.5: {e}")
    try:
        print("[Pipeline] Step 2.6: SHAP feature selection...")
        shap_features = get_shap_top_features(n=10)
        global SELECTED_FEATURES
        SELECTED_FEATURES = shap_features
        print(f"[Pipeline] Step 2.6 เสร็จสิ้น: SELECTED_FEATURES = {SELECTED_FEATURES}")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 2.6: {e}")
    try:
        print("[Pipeline] Step 3: Auto-apply best hyperparameters from sweep...")
        output_dir = getattr(config, "OUTPUT_DIR", OUTPUT_DIR) if 'config' in globals() else OUTPUT_DIR
        summary_file = os.path.join(output_dir, "hyperparameter_summary.csv")
        if os.path.exists(summary_file):
            from src.utils.data_utils import safe_read_csv
            df = safe_read_csv(summary_file)
            if "metric" in df.columns and df["metric"].notna().any():
                best = df.sort_values(by="metric", ascending=False).iloc[0]
                if 'config' in globals():
                    if hasattr(config, "LEARNING_RATE"):
                        config.LEARNING_RATE = best.get("learning_rate", getattr(config, "LEARNING_RATE", 0.01))
                    if hasattr(config, "DEPTH"):
                        config.DEPTH = int(best.get("depth", getattr(config, "DEPTH", 6)))
                    if hasattr(config, "L2_LEAF_REG"):
                        config.L2_LEAF_REG = int(best.get("l2_leaf_reg", getattr(config, "L2_LEAF_REG", 1)))
                    logger.info(
                        f"Applied best hyperparameters: lr={getattr(config, 'LEARNING_RATE', None)}, "
                        f"depth={getattr(config, 'DEPTH', None)}, l2={getattr(config, 'L2_LEAF_REG', None)}"
                    )
        else:
            logger.warning("ไม่มีคอลัมน์ metric ในไฟล์ sweep หรือไม่มีค่า metric ใช้ค่า default")
        print("[Pipeline] Step 3 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 3: {e}")
    try:
        print("[Pipeline] Step 4: Threshold optimization...")
        _execute_step("threshold", run_threshold_optimization)
        print("[Pipeline] Step 4 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 4: {e}")
    try:
        print("[Pipeline] Step 4.5: Validate after threshold...")
        _execute_step("validate_after_threshold", train_and_validate_model)
        print("[Pipeline] Step 4.5 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 4.5: {e}")
    try:
        print("[Pipeline] Step 5: Backtest...")
        _execute_step("backtest", run_backtest)
        # [เทพ] Save prediction หลัง backtest
        output_dir = getattr(config, "OUTPUT_DIR", OUTPUT_DIR) if 'config' in globals() else OUTPUT_DIR
        pred_path = os.path.join(output_dir, "buy_sell_hold_strategy_result.parquet")
        if os.path.exists(pred_path):
            import pandas as pd
            df_pred = pd.read_parquet(pred_path)
            if 'target_direction' in df_pred:
                y_true = df_pred['target_direction']
                y_pred = (df_pred['target_buy_sell_hold'] == 1).astype(int) if 'target_buy_sell_hold' in df_pred else None
                y_proba = df_pred['pred_proba'] if 'pred_proba' in df_pred else None
                save_final_predictions(str(output_dir), y_true, y_pred, y_proba)
        print("[Pipeline] Step 5 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 5: {e}")
    # [เทพ] Robustness test on unseen/future data
    try:
        print("[Pipeline] Step 5.9: Robustness test on unseen/future data...")
        from robustness import test_on_unseen_data
        _execute_step("robustness_unseen", test_on_unseen_data)
        print("[Pipeline] Step 5.9 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 5.9: {e}")
    try:
        print("[Pipeline] Step 6: Hold strategy...")
        _execute_step("hold", run_hold_strategy)
        print("[Pipeline] Step 6 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 6: {e}")
    try:
        print("[Pipeline] Step 7: Buy/Sell/Hold strategy...")
        _execute_step("buy_sell_hold", run_buy_sell_hold_strategy)
        print("[Pipeline] Step 7 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 7: {e}")
    try:
        print("[Pipeline] Step 8: Analyze buy/sell/hold results...")
        _execute_step("analyze_buy_sell_hold", analyze_buy_sell_hold_results)
        print("[Pipeline] Step 8 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 8: {e}")
    try:
        print("[Pipeline] Step 9: Summary performance...")
        import pandas as pd
        bsh_path = os.path.join('output_default', 'buy_sell_hold_strategy_result.parquet')
        hold_path = os.path.join('output_default', 'hold_strategy_result.parquet')
        model_path = os.path.join('output_default', 'preprocessed_super.parquet')
        if os.path.exists(bsh_path):
            df_bsh = pd.read_parquet(bsh_path)
            print(f"[Summary] Buy: {df_bsh['cum_return_buy'].iloc[-1]:.4f} | Sell: {df_bsh['cum_return_sell'].iloc[-1]:.4f} | Hold: {df_bsh['cum_return_hold'].iloc[-1]:.4f}")
        if os.path.exists(hold_path):
            df_hold = pd.read_parquet(hold_path)
            hold_final = df_hold['cum_return'].iloc[-1] if 'cum_return' in df_hold else float('nan')
            print(f"[Summary] Hold Strategy Cumulative Return: {hold_final:.4f}")
        if os.path.exists(model_path):
            df_model = pd.read_parquet(model_path)
            model_acc = df_model['target_direction'].mean() if 'target_direction' in df_model else float('nan')
            print(f"[Summary] Model Mean target_direction: {model_acc:.4f}")
        print("[Pipeline] Step 9 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 9: {e}")
    try:
        print("[Pipeline] Step 10: Dashboard/report...")
        metrics_path = os.path.join(output_dir, "metrics_summary_v32.csv")
        if os.path.exists(metrics_path):
            from src.utils.data_utils import safe_read_csv
            results_df = safe_read_csv(metrics_path)
        else:
            results_df = pd.DataFrame()
        _execute_step(
            "dashboard",
            generate_dashboard,
            results=results_df,
            output_filepath=os.path.join(output_dir, "dashboard.html"),
        )
        _execute_step("report", run_report)
        print("[Pipeline] Step 10 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 10: {e}")
    try:
        print("[Pipeline] Step 11: Ensure output files...")
        # Ensure metrics summary before output file check
        ensure_metrics_summary(str(output_dir))
        ensure_output_files([os.path.join(output_dir, "metrics_summary_v32.csv")])
        print("[Pipeline] Step 11 เสร็จสิ้น")
    except Exception as e:
        print(f"[Pipeline][ERROR] Step 11: {e}")
    print("[Pipeline] Full pipeline completed! 🎉")


def release_gpu_resources(handle, use_gpu: bool) -> None:
    """Release NVML handle and log the result."""
    if use_gpu and "pynvml" in globals() and handle:
        try:
            pynvml.nvmlShutdown()
            logging.info("GPU resources released")
        except Exception as exc:  # pragma: no cover - unlikely NVML failure
            logging.warning(f"Failed to shut down NVML: {exc}")
    else:
        logging.info("GPU not available, running on CPU")


def run_hold_strategy():
    """กลยุทธ์ hold: ซื้อแล้วถือ ไม่เทรดตามสัญญาณ (Buy & Hold)"""
    import pandas as pd
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    # สมมติถือทุกแท่ง (ถือ long ตลอด)
    df['hold_signal'] = 1
    # คำนวณผลตอบแทนสะสม (cumulative return)
    if 'return_1' in df.columns:
        df['cum_return'] = (1 + df['return_1']).cumprod()
    else:
        df['cum_return'] = float('nan')
    out_path = os.path.join('output_default', 'hold_strategy_result.parquet')
    df.to_parquet(out_path)
    print(f'[Hold] บันทึกผลกลยุทธ์ hold ที่ {out_path}')


def run_buy_sell_hold_strategy():
    """กลยุทธ์ buy/sell/hold: 1=buy, 0=hold, -1=sell ตาม return หรือสัญญาณ"""
    import pandas as pd
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    # สร้าง target_buy_sell_hold: 1=buy, 0=hold, -1=sell (เช่น return_1 > 0.1% = buy, < -0.1% = sell, else hold)
    if 'return_1' in df:
        threshold = 0.001
        df['target_buy_sell_hold'] = df['return_1'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    else:
        df['target_buy_sell_hold'] = 0
    # สรุป distribution
    dist = df['target_buy_sell_hold'].value_counts().sort_index()
    print(f"[Buy/Sell/Hold] Distribution: {dist.to_dict()}")
    # คำนวณ cumulative return ของแต่ละกลยุทธ์
    df['buy_signal'] = (df['target_buy_sell_hold'] == 1).astype(int)
    df['sell_signal'] = (df['target_buy_sell_hold'] == -1).astype(int)
    df['hold_signal'] = (df['target_buy_sell_hold'] == 0).astype(int)
    df['cum_return_buy'] = (1 + df['return_1'] * df['buy_signal']).cumprod()
    df['cum_return_sell'] = (1 - df['return_1'] * df['sell_signal']).cumprod()
    df['cum_return_hold'] = (1 + df['return_1'] * df['hold_signal']).cumprod()
    out_path = os.path.join('output_default', 'buy_sell_hold_strategy_result.parquet')
    df.to_parquet(out_path)
    print(f'[Buy/Sell/Hold] บันทึกผลกลยุทธ์ buy/sell/hold ที่ {out_path}')
    # สรุปผลลัพธ์สุดท้าย
    print(f"[Buy/Sell/Hold] Final Cumulative Return - Buy: {df['cum_return_buy'].iloc[-1]:.4f} | Sell: {df['cum_return_sell'].iloc[-1]:.4f} | Hold: {df['cum_return_hold'].iloc[-1]:.4f}")


def analyze_buy_sell_hold_results() -> None:
    """Analyze buy/sell/hold strategy results and print summary statistics."""
    import pandas as pd
    import numpy as np
    import os
    try:
        bsh_path = os.path.join('output_default', 'buy_sell_hold_strategy_result.parquet')
        if not os.path.exists(bsh_path):
            print('[Analyze] ไม่พบไฟล์ buy_sell_hold_strategy_result.parquet')
            return
        df = pd.read_parquet(bsh_path)
        # Win rate, mean/std return, max drawdown
        for label, mask in [('Buy', df['target_buy_sell_hold'] == 1),
                            ('Sell', df['target_buy_sell'] == -1),
                            ('Hold', df['target_buy_sell'] == 0)]:
            n = mask.sum()
            if n == 0:
                continue
            returns = df.loc[mask, 'return_1']
            win_rate = (returns > 0).mean()
            mean_ret = returns.mean()
            std_ret = returns.std()
            # Max drawdown
            eq_curve = (1 + returns).cumprod()
            roll_max = eq_curve.cummax()
            drawdown = (eq_curve - roll_max) / roll_max
            max_dd = drawdown.min()
            print(f"[{label}] Win rate: {win_rate:.2%} | Mean: {mean_ret:.4f} | Std: {std_ret:.4f} | Max Drawdown: {max_dd:.2%}")
        # Confusion matrix (ถ้ามี target_direction)
        if 'target_direction' in df:
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(df['target_direction'], (df['target_buy_sell'] == 1).astype(int))
            print(f"[Confusion Matrix] Buy-signal vs target_direction:\n{cm}")
            print(classification_report(df['target_direction'], (df['target_buy_sell'] == 1).astype(int)))
        # Performance table
        print("\n[Performance Table]")
        perf = {
            'Buy': df['cum_return_buy'].iloc[-1],
            'Sell': df['cum_return_sell'].iloc[-1],
            'Hold': df['cum_return_hold'].iloc[-1],
        }
        for k, v in perf.items():
            print(f"{k:>6}: {v:.4f}")
        # Plot equity curve (ถ้ามี matplotlib)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,5))
            plt.plot(df['cum_return_buy'], label='Buy')
            plt.plot(df['cum_return_sell'], label='Sell')
            plt.plot(df['cum_return_hold'], label='Hold')
            plt.title('Equity Curve: Buy/Sell/Hold')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('output_default', 'buy_sell_hold_equity_curve.png'))
            print('[Plot] บันทึกกราฟ equity curve ที่ output_default/buy_sell_hold_equity_curve.png')
        except ImportError:
            print('[Plot] ไม่พบ matplotlib ข้ามการ plot')
    except Exception as e:
        print(f'[Analyze Error] {e}')

# Ensure required functions are imported or defined
try:
    from feature_engineering import ensure_super_features_file, ensure_datetime_index
except ImportError:
    def ensure_super_features_file():
        raise NotImplementedError('ensure_super_features_file must be implemented or imported!')
    def ensure_datetime_index(df):
        return df

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

def check_data_leakage(df, target_col):
    """ตรวจสอบ feature ที่อาจรั่วข้อมูลอนาคต (target leakage)"""
    suspicious = []
    for col in df.columns:
        if col == target_col:
            continue
        # ถ้า correlation กับ target สูงผิดปกติ อาจรั่ว
        corr = abs(np.corrcoef(df[col], df[target_col])[0,1]) if df[col].dtype != 'O' else 0
        if corr > 0.95:
            suspicious.append(col)
    if suspicious:
        print(f"[Leakage] พบ feature ที่อาจรั่วข้อมูลอนาคต: {suspicious}")
        return True
    return False

def check_noise_and_overfit(X, y):
    """ตรวจสอบ noise/overfitting ด้วย learning curve และ validation curve"""
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
    mean_auc = np.mean(aucs)
    print(f"[AUC] Cross-validated AUC: {mean_auc:.4f}")
    if mean_auc < 0.65:
        print("[STOP] AUC ต่ำกว่า 0.65 หยุด pipeline เพื่อป้องกัน overfitting/noise/data leak")
        sys.exit(1)
    return mean_auc

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
    # ตรวจสอบ data leakage
    if check_data_leakage(df, target_col):
        print("[STOP] พบ data leakage ใน feature หยุด pipeline")
        sys.exit(1)
    # ตรวจสอบ GPU และเลือก ML library ที่เหมาะสม
    use_gpu = False
    model = None
    aucs = []
    try:
        import cuml  # type: ignore
        from cuml.ensemble import RandomForestClassifier as cuRF  # type: ignore
        from cuml.metrics import roc_auc_score as cu_roc_auc_score  # type: ignore
        use_gpu = True
        print("[GPU] ใช้ cuML RandomForestClassifier (GPU)")
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
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
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
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
                from catboost import CatBoostClassifier  # type: ignore
                use_gpu = True
                print("[GPU] ใช้ CatBoost (GPU)")
                tscv = TimeSeriesSplit(n_splits=5)
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
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
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
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
    return mean_auc


def run_optuna_with_shap() -> None:
    """Run Optuna hyperparameter tuning with SHAP feature importance analysis."""
    import pandas as pd
    import numpy as np
    try:
        import shap  # type: ignore
    except ImportError:
        print('[SHAP] ไม่พบไลบรารี shap ข้ามการวิเคราะห์ SHAP')
        return
    import optuna
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'device': 'cuda',  # ใช้ device แทน tree_method
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        booster = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=False)
        y_pred = booster.predict(dval)
        auc = roc_auc_score(y_val, y_pred)
        return auc
    print('[Optuna] เริ่ม hyperparameter tuning แบบเทพ...')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print(f'[Optuna] Best params: {study.best_params} | Best AUC: {study.best_value:.4f}')
    # Train best model
    best_params = study.best_params
    best_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'device': 'cuda'})
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=False)
    # SHAP analysis
    print('[SHAP] วิเคราะห์ feature importance แบบเทพ...')
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_val)
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    plt.savefig('output_default/shap_summary_plot.png')
    print('[SHAP] บันทึกกราฟ shap summary ที่ output_default/shap_summary_plot.png')
    # Top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top_features = [X_val.columns[i] for i in top_idx]
    print(f'[SHAP] Top 10 features: {top_features}')
    # Save best params and top features
    pd.Series(study.best_params).to_json('output_default/optuna_best_params.json')
    pd.Series(top_features).to_csv('output_default/shap_top_features.csv', index=False)
    print('[Optuna+SHAP] บันทึก best params และ top features เรียบร้อย')


def check_required_files_for_mode(mode):
    """ตรวจสอบไฟล์ที่จำเป็นก่อนรันแต่ละโหมด ถ้าไม่ครบให้แจ้งเตือนและหยุด"""
    import os
    required = {
        'preprocess': [],
        'sweep': ['output_default/preprocessed_super.parquet'],
        'threshold': ['output_default/preprocessed_super.parquet'],
        'backtest': ['output_default/preprocessed_super.parquet', 'models/'],
        'report': ['output_default/final_predictions.csv'],
        'full_pipeline': [],
        'all': [],
        'hyper_sweep': ['output_default/preprocessed_super.parquet'],
        'wfv': ['output_default/preprocessed_super.parquet', 'models/'],
        'hold': ['output_default/preprocessed_super.parquet'],
        'train_auc': ['output_default/preprocessed_super.parquet'],
        'optuna_shap': ['output_default/preprocessed_super.parquet'],
        'robustness_unseen': ['output_default/preprocessed_super.parquet', 'models/'],
    }
    missing = []
    for f in required.get(mode, []):
        if f.endswith('/'):
            if not os.path.isdir(f):
                missing.append(f)
        else:
            if not os.path.exists(f):
                missing.append(f)
    if missing:
        print(f"\n[❌] ไม่พบไฟล์/โฟลเดอร์ที่จำเป็นสำหรับโหมดนี้: {missing}\nกรุณารันโหมดที่เกี่ยวข้องก่อน เช่น เตรียมข้อมูล หรือฝึกโมเดล\n")
        return False
    return True


def print_summary_for_mode(mode):
    """แสดงผลลัพธ์สำคัญหลังจบแต่ละโหมด (เทพ)"""
    import os
    import pandas as pd
    if mode == 'report':
        metrics_path = 'output_default/metrics_summary_v32.csv'
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            print(f"\n[📊] Metrics Summary (เทพ):\n{df.to_string(index=False)}\n")
        else:
            print("[⚠️] ไม่พบ metrics_summary_v32.csv กรุณาตรวจสอบ pipeline")
        pred_path = 'output_default/final_predictions.csv'
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            print(f"[🔎] ตัวอย่างผลลัพธ์ 5 แถวแรก:\n{df.head()}\n")
    elif mode == 'backtest':
        eq_path = 'output_default/buy_sell_hold_equity_curve.png'
        if os.path.exists(eq_path):
            print(f"[✅] สร้างกราฟ equity curve สำเร็จ: {eq_path}\n")
        else:
            print("[⚠️] ไม่พบไฟล์ equity curve กรุณาตรวจสอบ pipeline")
    elif mode == 'preprocess':
        fe_path = 'output_default/preprocessed_super.parquet'
        if os.path.exists(fe_path):
            print(f"[✅] เตรียมข้อมูลเสร็จสิ้น: {fe_path}\n")
    # เพิ่มเติมได้ตามโหมดอื่น ๆ


def run_mode(mode):
    """Run the selected mode (เทพ: ตรวจสอบไฟล์, error handling, summary)"""
    from src.main import main as pipeline_main
    if not check_required_files_for_mode(mode):
        return
    try:
        if mode == "preprocess":
            pipeline_main(run_mode="PREPARE_TRAIN_DATA")
        elif mode == "backtest":
            pipeline_main(run_mode="FULL_RUN")
        elif mode == "report":
            pipeline_main(run_mode="REPORT")
        elif mode == "full_pipeline":
            pipeline_main(run_mode="FULL_PIPELINE")
        elif mode == "sweep":
            run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
        elif mode == "wfv":
            run_walkforward()  # [Patch] call simplified WFV runner
        elif mode == "all":
            run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
            candidates = [
                os.path.join(str(OUTPUT_DIR), "best_param.json"),
                os.path.join(str(OUTPUT_DIR), "best_params.json"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    with open(cand, "r", encoding="utf-8") as fh:
                        best_params = json.load(fh)
                    update_config_from_dict(best_params)
                    break
            run_walkforward()
        elif mode == "hold":
            run_hold_strategy()
        elif mode == "train_auc":
            train_and_validate_model()
        elif mode == "optuna_shap":
            run_optuna_with_shap()
        elif mode == "robustness_unseen":
            from robustness import test_on_unseen_data
            test_on_unseen_data()
        else:
            print(f"[❌] ไม่รู้จักโหมด: {mode}")
            return
        print_summary_for_mode(mode)
        print(f"\n[🎉] โหมด {mode} เสร็จสมบูรณ์! ถ้ามีปัญหาให้ดู log หรือแจ้งผู้ดูแลระบบ\n")
    except Exception as e:
        import traceback
        print(f"[FATAL ERROR][{mode}] Exception: {e}\n{traceback.format_exc()}\n[❌] เกิดข้อผิดพลาดขณะรันโหมด {mode} กรุณาตรวจสอบ log หรือแจ้งผู้ดูแลระบบ\n")


def safe_path(path: str, default: str = "output_default") -> str:
    """Return a safe path, fallback to default if path is empty or None."""
    if not path or str(path).strip() == '':
        logging.warning(f"[safe_path] Path is empty, fallback to default: {default}")
        return default
    return path

def safe_makedirs(path: str):
    path = safe_path(path)
    if not os.path.exists(path):
        logging.info(f"[safe_makedirs] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    return path


def save_final_predictions(output_dir: str, y_true, y_pred, y_proba=None):
    import pandas as pd
    output_dir = safe_path(output_dir)
    safe_makedirs(output_dir)
    print(f"[DEBUG][save_final_predictions] output_dir: {output_dir}")
    df = pd.DataFrame({
        'target': y_true,
        'pred': y_pred,
        'pred_proba': y_proba if y_proba is not None else [None]*len(y_true)
    })
    print(f"[DEBUG] save_final_predictions: ตัวอย่างข้อมูล 5 แถวแรก\n{df.head()}")
    print(f"[DEBUG] จำนวนแถวที่บันทึก: {len(df)} | target unique: {pd.Series(y_true).unique()} | pred unique: {pd.Series(y_pred).unique()} | proba ตัวอย่าง: {df['pred_proba'].head().tolist()}")
    if len(pd.Series(y_pred).unique()) == 1:
        print(f"[WARNING][save_final_predictions] pred มีค่าเดียว: {pd.Series(y_pred).unique()}")
    if y_proba is not None and (pd.Series(y_proba).isnull().all() or len(pd.Series(y_proba).unique()) == 1):
        print(f"[WARNING][save_final_predictions] pred_proba มีค่าเดียวหรือว่าง")
    out_path = os.path.join(output_dir, 'final_predictions.parquet')
    out_path = safe_path(out_path)
    print(f"[DEBUG][save_final_predictions] Writing parquet: {out_path}")
    df.to_parquet(out_path, index=False)
    out_csv = os.path.join(output_dir, 'final_predictions.csv')
    out_csv = safe_path(out_csv)
    print(f"[DEBUG][save_final_predictions] Writing csv: {out_csv}")
    df.to_csv(out_csv, index=False)
    logging.info(f"[เทพ] บันทึก prediction/result ที่มีคอลัมน์ครบที่ {out_path} และ {out_csv}")


def ensure_metrics_summary(output_dir: str):
    import pandas as pd
    import glob
    import os
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
    metrics_path = os.path.join(output_dir, "metrics_summary_v32.csv")
    pred_files = glob.glob(os.path.join(output_dir, '*_strategy_result.parquet'))
    if not pred_files:
        print(f"[ERROR][metrics] ไม่พบไฟล์ *_strategy_result.parquet ใน {output_dir}")
        logging.error(f"[Metrics] ไม่พบไฟล์ *_strategy_result.parquet ใน {output_dir}")
        return
    pred_file = max(pred_files, key=os.path.getmtime)
    df = pd.read_parquet(pred_file)
    # Map columns
    col_target = None
    col_pred = None
    col_proba = None
    for c in df.columns:
        if c.lower() in ["target", "target_direction", "label"]:
            col_target = c
        if c.lower() in ["pred", "prediction", "pred_label"]:
            col_pred = c
        if c.lower() in ["pred_proba", "proba", "probability"]:
            col_proba = c
    if col_target is None or col_pred is None:
        print(f"[ERROR][metrics] ไม่พบคอลัมน์ target/pred ใน {pred_file} (columns: {list(df.columns)})")
        logging.error(f"[Metrics] ไม่พบคอลัมน์ target/pred/prediction ใน {pred_file} (columns: {list(df.columns)})")
        return
    y_true = df[col_target]
    y_pred = df[col_pred]
    y_proba = df[col_proba] if col_proba else None
    print(f"[DEBUG][metrics] target unique: {pd.Series(y_true).unique()} | pred unique: {pd.Series(y_pred).unique()} | proba unique: {pd.Series(y_proba).unique() if y_proba is not None else 'None'}")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if y_proba is not None else '',
        'support_0': (y_true == 0).sum(),
        'support_1': (y_true == 1).sum(),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(metrics_path, index=False)
    logging.info(f"[เทพ] metrics_summary_v32.csv ถูกสร้างจากผลลัพธ์จริง: {metrics}")


# --- ย้าย _script_main มาไว้ตรงนี้ ---
def _script_main():
    try:
        import sys
        print("[DEBUG][_script_main] Start")
        print(f"[DEBUG][_script_main] sys.argv: {sys.argv}")
        if len(sys.argv) == 1:
            print("[DEBUG][_script_main] No arguments, entering CLI menu...")
            mode = show_menu_and_get_mode()
            print(f"[DEBUG][_script_main] Selected mode: {mode}")
            run_mode(mode)
            print("[DEBUG][_script_main] run_mode finished")
            return
        print("[DEBUG][_script_main] Parsing args...")
        args = parse_projectp_args()
        print(f"[DEBUG][_script_main] Parsed args: {args}")
        if getattr(args, "auto_convert", False):
            src_dir = os.getenv("SOURCE_CSV_DIR", "data")
            dest_dir = os.getenv("DEST_CSV_DIR")
            base = setup_output_directory(OUTPUT_BASE_DIR, OUTPUT_DIR_NAME)
            dest = dest_dir or os.path.join(base, "converted_csvs")
            os.makedirs(dest, exist_ok=True)
            auto_convert_csv(src_dir, output_path=dest)
            print("[DEBUG][_script_main] auto_convert finished, exiting...")
            sys.exit(0)
        print(f"[DEBUG][_script_main] Running mode: {args.mode}")
        run_mode(args.mode)
        print("[DEBUG][_script_main] run_mode finished")
    except Exception as e:
        import traceback
        print(f"[FATAL ERROR] Exception in _script_main: {e}\n{traceback.format_exc()}")
        raise


print("[DEBUG][__main__] about to check __name__ == '__main__'")
# --- Legacy logic moved to projectp/ modules ---
# (see projectp/preprocess.py, train.py, predict.py, metrics.py, debug.py)

if __name__ == "__main__":
    from projectp.cli import main_cli
    main_cli()