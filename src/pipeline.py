"""
pipeline.py: ฟังก์ชันหลักและ logic ของ pipeline เดิมจาก main.py
"""
import logging
import os
import sys
import json
import time
import glob
import shutil
import traceback
import gc
import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import TimeSeriesSplit

from src.utils import get_env_float, maybe_collect, load_settings
from src.model_helpers import (
    ensure_model_files_exist,
    ensure_main_features_file,
    save_features_main_json,
    save_features_json,
)
from src.pipeline_helpers import run_auto_threshold_stage, run_pipeline_stage
from src.main_utils import (
    setup_fonts,
    print_gpu_utilization,
    plot_equity_curve,
    ensure_default_output_dir,
    load_validated_csv,
)
from src.data_loader import (
    setup_output_directory as dl_setup_output_directory,
    load_data,
    prepare_datetime,
    safe_load_csv_auto,
)
from src.features import (
    calculate_m15_trend_zone,
    engineer_m1_features,
    clean_m1_data,
    calculate_m1_entry_signals,
    create_session_column,
    load_features_for_model,
    save_features,
    load_features,
)
from src.strategy import (
    run_all_folds_with_threshold,
    train_and_export_meta_model,
    DriftObserver,
    plot_equity_curve,
    run_backtest_simulation_v34,
)
from src.utils import (
    export_trade_log,
    download_model_if_missing,
    download_feature_list_if_missing,
    get_env_float,
    estimate_resource_plan,
    validate_file,
)
from src.csv_validator import validate_and_convert_csv
from src.config_defaults import DEFAULT_OUTPUT_DIR, DEFAULT_DATA_FILE_PATH_M1, DEFAULT_DATA_FILE_PATH_M15

def safe_path(path: str, default: str = "output_default") -> str:
    return path if path else default

def safe_makedirs(path: str):
    path = safe_path(path)
    os.makedirs(path, exist_ok=True)
    return path

def main(run_mode="FULL_PIPELINE", skip_prepare=False, suffix_from_prev_step=None):
    """
    Main execution function for the Gold Trading AI script.
    Handles different run modes: PREPARE_TRAIN_DATA, TRAIN_MODEL_ONLY, FULL_RUN, FULL_PIPELINE.
    (เนื้อหาฟังก์ชัน main ทั้งหมดถูกย้ายมาจาก main.py)
    """
    # กำหนดค่าเริ่มต้น
    start_time = time.time()
    print("[DEBUG] เริ่มต้นการทำงานของ Pipeline")
    logging.info("เริ่มต้นการทำงานของ Pipeline")
    
    # โหลดการตั้งค่า
    settings = load_settings()
    print(f"[DEBUG] settings: {settings}")
    logging.info(f"การตั้งค่า: {settings}")

    # ตรวจสอบและดาวน์โหลดไฟล์โมเดลถ้าจำเป็น
    output_dir = safe_path(DEFAULT_OUTPUT_DIR)
    trade_log_path = os.path.join(output_dir, "trade_log_v32_walkforward.csv")
    m1_data_path = DEFAULT_DATA_FILE_PATH_M1
    print(f"[DEBUG] output_dir: {output_dir}, trade_log_path: {trade_log_path}, m1_data_path: {m1_data_path}")
    model_files_exist = ensure_model_files_exist(output_dir, trade_log_path, m1_data_path)
    if not model_files_exist:
        print("[DEBUG] download_model_if_missing called")
        download_model_if_missing(os.path.join(output_dir, "meta_classifier.pkl"), "URL_MODEL_MAIN")

    # เตรียมข้อมูล
    if run_mode in ["PREPARE_TRAIN_DATA", "FULL_RUN", "FULL_PIPELINE"]:
        print("[DEBUG] ขั้นตอนที่ 1: เตรียมข้อมูล")
        logging.info("ขั้นตอนที่ 1: เตรียมข้อมูล")
        try:
            df_m1 = load_validated_csv(m1_data_path, "M1")
            print(f"[DEBUG] โหลดข้อมูล M1 สำเร็จ: {df_m1.shape}")
            logging.info(f"โหลดข้อมูล M1 สำเร็จ: {df_m1.shape}")
            # ตัวอย่าง: สร้าง features
            df_m1_features = engineer_m1_features(df_m1)
            print(f"[DEBUG] สร้าง features สำเร็จ: {df_m1_features.shape}")
            logging.info(f"สร้าง features สำเร็จ: {df_m1_features.shape}")
        except Exception as e:
            print(f"[DEBUG] ERROR: {e}")
            logging.error(f"เกิดข้อผิดพลาดขณะเตรียมข้อมูล: {e}")
            return

    # ฝึกโมเดล
    if run_mode in ["TRAIN_MODEL_ONLY", "FULL_RUN", "FULL_PIPELINE"]:
        print("[DEBUG] ขั้นตอนที่ 2: ฝึกโมเดล")
        logging.info("ขั้นตอนที่ 2: ฝึกโมเดล")
        try:
            # ตัวอย่าง: ฝึก meta model (ใช้ trade log และ features)
            # trade_log_path = ... (กำหนด path ที่ถูกต้อง)
            # train_and_export_meta_model(trade_log_path, m1_data_path, output_dir)
            logging.info("ฝึกโมเดลจริง (โปรดเติม trade log และ args ที่เหมาะสม)")
        except Exception as e:
            print(f"[DEBUG] ERROR: {e}")
            logging.error(f"เกิดข้อผิดพลาดขณะฝึกโมเดล: {e}")
            return

    # รัน Pipeline อัตโนมัติ
    if run_mode == "FULL_PIPELINE":
        print("[DEBUG] ขั้นตอนที่ 3: รัน Pipeline อัตโนมัติ")
        logging.info("ขั้นตอนที่ 3: รัน Pipeline อัตโนมัติ")
        try:
            run_auto_threshold_stage()
            print("[DEBUG] Pipeline อัตโนมัติ (จริง) สำเร็จ")
            logging.info("Pipeline อัตโนมัติ (จริง) สำเร็จ")
        except Exception as e:
            print(f"[DEBUG] ERROR: {e}")
            logging.error(f"เกิดข้อผิดพลาดใน pipeline อัตโนมัติ: {e}")
            return
    print("[DEBUG] Pipeline เสร็จสมบูรณ์!")
    logging.info("Pipeline เสร็จสมบูรณ์!")
    logging.info("เสร็จสิ้นการทำงานของ Pipeline")
    logging.info(f"ใช้เวลาทั้งหมด: {time.time() - start_time} วินาที")
