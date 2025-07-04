import pytest
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

from src.utils.auto_train_meta_classifiers import auto_train_meta_classifiers
from src.config import logger


def test_auto_train_meta_classifiers_loads_and_trains(tmp_path, caplog):
    """Should load trade log file and train model when present."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    df = pd.DataFrame({"x": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]})
    df.to_csv(
        Path(cfg.OUTPUT_DIR) / "trade_log_v32_walkforward.csv.gz",
        index=False,
        compression="gzip",
    )
    (tmp_path / "features_main.json").write_text("[\"x\"]")
    with caplog.at_level('INFO', logger=logger.name):
        result = auto_train_meta_classifiers(
            cfg,
            None,
            models_dir=str(tmp_path),
            features_dir=str(tmp_path),
        )
    assert isinstance(result, dict) and "model_path" in result
    assert Path(result["model_path"]).exists()
    assert any("Meta-classifier trained" in m for m in caplog.messages)
    assert any("metrics - accuracy" in m for m in caplog.messages)


def test_auto_train_meta_classifiers_missing_trade_log(tmp_path, caplog):
    """Should log an error when trade log is absent."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    with caplog.at_level('ERROR', logger=logger.name):
        result = auto_train_meta_classifiers(cfg, None)
    assert result is None
    assert any("Patch v6.4.2" in m and "not found" in m for m in caplog.messages)


def test_auto_train_meta_classifiers_with_data(tmp_path):
    """Should train using provided DataFrame."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text("[\"f\"]")
    data = pd.DataFrame({"f": [0, 1, 0, 1, 0, 1], "target": [0, 1, 0, 1, 0, 1]})
    res = auto_train_meta_classifiers(
        cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
    )
    assert Path(res["model_path"]).exists()


def test_auto_train_meta_classifiers_missing_target(tmp_path, caplog):
    """Should skip training when 'target' column missing."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text("[\"f\"]")
    data = pd.DataFrame({"f": [1, 0, 1, 0, 1]})
    with caplog.at_level('WARNING', logger=logger.name):
        res = auto_train_meta_classifiers(
            cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
        )
    assert res == {}
    assert any("6.5.10" in m and "target" in m for m in caplog.messages)



def test_auto_train_meta_classifiers_derive_target(tmp_path, caplog):
    """Should derive target from profit column when missing."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text("[\"f\"]")
    data = pd.DataFrame({"f": [1, 0, 1, 0, 1], "profit": [1.0, -0.5, 2.0, -1.0, 0.4]})
    with caplog.at_level('INFO', logger=logger.name):
        res = auto_train_meta_classifiers(
            cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
        )
    assert Path(res["model_path"]).exists()
    assert any("Auto-generating 'target' from 'profit'" in m for m in caplog.messages)


def test_auto_train_meta_classifiers_derive_target_alt_column(tmp_path, caplog):
    """Should derive target from alternate profit column when missing."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text("[\"f\"]")
    data = pd.DataFrame({"f": [1, 0, 1, 0, 1], "pnl_usd_net": [1.0, -0.5, 2.0, -1.0, 0.4]})
    with caplog.at_level('INFO', logger=logger.name):
        res = auto_train_meta_classifiers(
            cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
        )
    assert Path(res["model_path"]).exists()
    assert any("Auto-generating 'target' from 'pnl_usd_net'" in m for m in caplog.messages)



def test_auto_train_meta_classifiers_warns_missing_features(tmp_path, caplog):
    """Should log warning when some features are missing."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text('["f1", "f2"]')
    data = pd.DataFrame({"f1": [1, 0, 1, 0, 1], "profit": [1.0, -1.0, 2.0, -0.5, 1.5]})
    with caplog.at_level('WARNING', logger=logger.name):
        result = auto_train_meta_classifiers(
            cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
        )
    assert Path(result["model_path"]).exists()
    assert any("missing features" in m for m in caplog.messages)


def test_auto_train_meta_classifiers_zero_profit(tmp_path, caplog):
    """Should skip training when all profit values are zero."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text('["f"]')
    data = pd.DataFrame({"f": [1, 0, 1, 0, 1], "profit": [0, 0, 0, 0, 0]})
    with caplog.at_level('ERROR', logger=logger.name):
        res = auto_train_meta_classifiers(
            cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
        )
    assert res is None
    assert any("6.6.12" in m and "All profit values are 0" in m for m in caplog.messages)

def test_auto_train_meta_classifiers_fallback_profit(tmp_path, caplog):
    """Should use profit column as fallback feature when no others available."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    (tmp_path / "features_main.json").write_text('["f1", "f2"]')
    data = pd.DataFrame({"profit": [1.0, -1.0, 0.5, -0.2, 1.2]})
    with caplog.at_level('WARNING', logger=logger.name):
        result = auto_train_meta_classifiers(
            cfg, data, models_dir=str(tmp_path), features_dir=str(tmp_path)
        )
    assert Path(result["model_path"]).exists()
    assert any("fallback feature" in m for m in caplog.messages)
