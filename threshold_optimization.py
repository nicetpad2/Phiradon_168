import os
import argparse
import pandas as pd
import logging

try:  # [Patch v5.10.3] robust fallback when heavy config import fails
    import importlib
    _cfg = importlib.import_module("src.config")
    logger = getattr(_cfg, "logger", logging.getLogger("threshold_optimization"))
    optuna = getattr(_cfg, "optuna", None)
except Exception:  # pragma: no cover - optional dependency
    logger = logging.getLogger("threshold_optimization")
    try:
        import optuna
    except Exception:
        optuna = None
# [Patch v5.9.17] provide basic logger/optuna if src.config import fails

# [Patch v5.5.14] Improved threshold optimization with Optuna


def parse_args(args=None) -> argparse.Namespace:
    """แปลงค่า argument จาก command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="models", help="โฟลเดอร์บันทึกผลลัพธ์")
    parser.add_argument("--trials", type=int, default=10, help="จำนวน trial ในการค้นหา")
    parser.add_argument("--study-name", default="threshold-wfv", help="ชื่อ Optuna study")
    parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="ทิศทาง optimization",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="เวลาสูงสุด (วินาที)"
    )
    return parser.parse_args(args)


def safe_path(path: str, default: str = "output_default") -> str:
    return path if path else default


def run_threshold_optimization(
    output_dir: str = "models",
    trials: int = 10,
    study_name: str = "threshold-wfv",
    direction: str = "maximize",
    timeout: int | None = None,
) -> pd.DataFrame:
    """รัน Optuna เพื่อตามหาค่า threshold ที่ดีที่สุด"""

    output_dir = safe_path(output_dir)
    if optuna is None:  # pragma: no cover - optuna อาจไม่ติดตั้งในบางสภาพแวดล้อม
        logger.warning("optuna not available; using default threshold=0.5")
        result = pd.DataFrame({"best_threshold": [0.5], "best_value": [0.0]})
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(
            os.path.join(output_dir, "threshold_wfv_optuna_results.csv"),
            index=False,
        )
        result.to_json(
            os.path.join(output_dir, "threshold_wfv_optuna_results.json"),
            orient="records",
        )
        return result

    def objective(trial: "optuna.trial.Trial") -> float:
        threshold = trial.suggest_float("threshold", 0.0, 1.0)
        parquet_path = os.path.join("output_default", "preprocessed_super.parquet")
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                # สมมติว่ามีคอลัมน์ 'target' และ 'pred_proba' (หรือปรับชื่อให้ตรงกับไฟล์จริง)
                if 'target' in df.columns and 'pred_proba' in df.columns:
                    preds = (df['pred_proba'] >= threshold).astype(int)
                    acc = (preds == df['target']).mean()
                    # ถ้ามี sklearn.metrics.roc_auc_score ให้คำนวณ auc ด้วย
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(df['target'], df['pred_proba'])
                    except Exception:
                        auc = 0.0
                    # เลือก metric ที่ต้องการ optimize (เช่น accuracy หรือ auc)
                    value = acc  # หรือ value = auc
                    logger.info(f"[Auto] threshold={threshold:.4f} acc={acc:.4f} auc={auc:.4f}")
                    return value
                else:
                    logger.warning("[Auto] ไม่พบคอลัมน์ 'target' หรือ 'pred_proba' ในไฟล์ preprocessed_super.parquet ใช้ mock objective แทน")
            except Exception as e:
                logger.warning(f"[Auto] อ่านไฟล์ preprocessed_super.parquet ไม่สำเร็จ: {e} ใช้ mock objective แทน")
        # fallback: mock objective
        value = 1.0 - abs(threshold - 0.5)
        return value

    def log_progress(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        logger.info(
            f"Trial {trial.number}: threshold={trial.params['threshold']:.4f} value={trial.value:.4f}"
        )

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(
        direction=direction, study_name=study_name, sampler=sampler
    )
    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout,
        callbacks=[log_progress],
    )

    best_threshold = study.best_params.get("threshold", 0.5)
    best_value = study.best_value

    df = pd.DataFrame({"best_threshold": [best_threshold], "best_value": [best_value]})
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "threshold_wfv_optuna_results.csv")
    json_path = os.path.join(output_dir, "threshold_wfv_optuna_results.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")
    return df


def main(args=None) -> int:
    parsed = parse_args(args)
    run_threshold_optimization(
        output_dir=parsed.output_dir,
        trials=parsed.trials,
        study_name=parsed.study_name,
        direction=parsed.direction,
        timeout=parsed.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
