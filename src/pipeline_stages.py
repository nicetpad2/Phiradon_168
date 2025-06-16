import subprocess
import logging
from typing import Any
from src.utils.errors import PipelineError
import os
import pandas as pd
from utils import load_trade_log
from src.utils.model_utils import get_latest_model_and_threshold

logger = logging.getLogger(__name__)

def run_preprocess(config, runner=subprocess.run) -> None:
    logger.info("[Stage] preprocess")
    from pathlib import Path
    parquet_output_dir_str = getattr(config, "parquet_dir", None)
    if not parquet_output_dir_str:
        base_data_dir = getattr(config, "data_dir", "./data")
        parquet_output_dir = Path(base_data_dir) / "parquet_cache"
        logger.warning(
            "[AutoConvert] 'data.parquet_dir' not set in config. Defaulting to: %s",
            parquet_output_dir,
        )
    else:
        parquet_output_dir = Path(parquet_output_dir_str)
    source_csv_path = getattr(config, "data_path", None) or getattr(
        config, "raw_m1_filename", None
    )
    if source_csv_path:
        from src.data_loader import auto_convert_gold_csv
        from main import auto_convert_csv_to_parquet
        auto_convert_csv_to_parquet(source_path=source_csv_path, dest_folder=parquet_output_dir)
    else:
        logger.error(
            "[AutoConvert] 'data.path' is not defined in config. Skipping conversion."
        )
    m1_path = config.raw_m1_filename
    if os.path.exists(m1_path):
        from src import csv_validator
        try:
            csv_validator.validate_and_convert_csv(m1_path)
        except FileNotFoundError as exc:
            logger.error("[Validation] CSV file not found: %s", exc)
        except ValueError as exc:
            logger.error("[Validation] CSV validation error: %s", exc)
        except Exception as exc:
            logger.error("[Validation] CSV validation failed: %s", exc)
    else:
        logger.warning("[Validation] CSV file not found: %s", m1_path)
    from src.data_loader import auto_convert_gold_csv
    auto_convert_gold_csv(os.path.dirname(m1_path), output_path=m1_path)
    fill_method = getattr(config, "cleaning_fill_method", "drop")
    try:
        runner(
            [
                os.environ.get("PYTHON", "python"),
                "src/data_cleaner.py",
                m1_path,
                "--fill",
                fill_method,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Preprocess failed", exc_info=True)
        raise PipelineError("preprocess stage failed") from exc

def run_sweep(config, runner=subprocess.run) -> None:
    logger.info("[Stage] sweep")
    try:
        runner(
            [os.environ.get("PYTHON", "python"), "tuning/hyperparameter_sweep.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Sweep failed", exc_info=True)
        raise PipelineError("sweep stage failed") from exc

def run_threshold(config, runner=subprocess.run) -> None:
    logger.info("[Stage] threshold")
    try:
        runner(
            [os.environ.get("PYTHON", "python"), "threshold_optimization.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Threshold optimization failed", exc_info=True)
        raise PipelineError("threshold stage failed") from exc

def run_backtest(config, pipeline_func=None) -> None:
    logger.info("[Stage] backtest")
    from config import config as cfg
    if pipeline_func is None:
        pipeline_func = run_backtest_pipeline
    try:
        trade_log_file = getattr(cfg, "TRADE_LOG_PATH", None)
        trade_df = load_trade_log(
            trade_log_file,
            min_rows=getattr(cfg, "MIN_TRADE_ROWS", 10),
        )
    except FileNotFoundError as exc:
        logger.error("Trade log file not found: %s", exc)
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    except ValueError as exc:
        logger.error("Invalid trade log format: %s", exc)
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    except Exception as exc:
        logger.error("Failed loading trade log: %s", exc)
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    else:
        logger.debug("Loaded trade log with %d rows", len(trade_df))
    model_dir = config.model_dir
    model_path, threshold = get_latest_model_and_threshold(
        model_dir, config.threshold_file
    )
    try:
        pipeline_func(pd.DataFrame(), pd.DataFrame(), model_path, threshold)
    except Exception as exc:
        logger.error("Backtest failed", exc_info=True)
        raise PipelineError("backtest stage failed") from exc

def run_report(config) -> None:
    logger.info("[Stage] report")
    try:
        from src.main import run_pipeline_stage
        run_pipeline_stage("report")
    except Exception as exc:
        logger.error("Report failed", exc_info=True)
        raise PipelineError("report stage failed") from exc

def run_backtest_pipeline(features_df, price_df, model_path, threshold) -> None:
    logger.info("Running backtest with model=%s threshold=%s", model_path, threshold)
    try:
        from src.main import run_pipeline_stage
        run_pipeline_stage("backtest")
    except Exception:
        logger.exception("Internal backtest error")
        raise
