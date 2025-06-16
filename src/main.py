# main.py (entry point) สำหรับเรียกใช้งาน pipeline/main logic
import logging
from src.pipeline import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["preprocess", "backtest", "report"], default="full"
    )
    args = parser.parse_args()
    stage_map = {
        "preprocess": "PREPARE_TRAIN_DATA",
        "backtest": "FULL_RUN",
        "report": "REPORT",
    }
    selected_run_mode = stage_map.get(args.stage, "FULL_PIPELINE")
    logging.info(f"(Starting) กำลังเริ่มการทำงานหลัก (main) ในโหมด: {selected_run_mode}...")
    main(run_mode=selected_run_mode)
