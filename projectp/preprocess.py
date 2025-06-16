import os
import sys
import pandas as pd

# NOTE: SELECTED_FEATURES จะถูก set จาก pipeline หลัก (global)
SELECTED_FEATURES = None

def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    feature_cols = [c for c in df.columns if c not in ['target_event','target_direction','Date','Time','Symbol','datetime','target','pred_proba'] and df[c].dtype != 'O']
    global SELECTED_FEATURES
    if SELECTED_FEATURES:
        feature_cols = [c for c in feature_cols if c in SELECTED_FEATURES]
    if 'target_event' in df.columns:
        target_col = 'target_event'
    elif 'target_direction' in df.columns:
        target_col = 'target_direction'
    else:
        raise ValueError('ไม่พบ target ที่เหมาะสมในไฟล์ feature engineered')
    return feature_cols, target_col

# NOTE: ensure_super_features_file ควร import จาก feature_engineering หรือ implement stub
# ไม่ import ตรง ๆ เพื่อหลีกเลี่ยง error ในขั้นตอน static analysis

def ensure_super_features_file() -> str:
    """
    ตรวจสอบและคืน path ของไฟล์ features หลัก (เทพ/robust):
    - ถ้าเจอ output_default/preprocessed_super.parquet ให้ใช้ไฟล์นี้
    - ถ้าเจอ features_main.json ให้ใช้ไฟล์นี้ (feature list)
    - ถ้าไม่เจอ ให้แจ้ง error พร้อม UX/สี/เสียง/ASCII Art
    """
    import os
    import sys
    from .debug import print_logo
    try:
        from .cli import color, beep, ascii_error
    except ImportError:
        def color(x: str, c: str) -> str: return x
        def beep() -> None: pass
        def ascii_error() -> str: return ''
    parquet_path = os.path.join('output_default', 'preprocessed_super.parquet')
    json_path = 'features_main.json'
    if os.path.exists(parquet_path):
        print(color(f"[OK] พบไฟล์ features: {parquet_path}", 'green'))
        return parquet_path
    elif os.path.exists(json_path):
        print(color(f"[OK] พบไฟล์ features: {json_path}", 'green'))
        return json_path
    else:
        print_logo()
        print(color("[❌] ไม่พบไฟล์ features หลัก (preprocessed_super.parquet หรือ features_main.json)", 'red'))
        print(color("[💡] กรุณารันโหมดเตรียมข้อมูล (Preprocess) หรือเตรียมไฟล์ features_main.json ก่อน!", 'yellow'))
        print(ascii_error())
        beep()
        sys.exit(1)

def _safe_unique(val) -> list[str]:
    # Helper: convert unique() result to list, fallback to str if error
    try:
        return list(val.unique())
    except Exception:
        return [str(x) for x in val]

def run_preprocess():
    """Preprocess pipeline: โหลด feature engineered + target อัตโนมัติ, เลือก feature/target ใหม่, บันทึก preprocessed_super.parquet"""
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    print(f'ใช้ feature: {feature_cols}')
    print(f'ใช้ target: {target_col}')
    df_out = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    # ไม่สร้างคอลัมน์ 'target' ซ้ำถ้ามี target อยู่แล้ว
    if 'target' not in df_out.columns:
        df_out['target'] = df_out[target_col]
    if 'pred_proba' not in df_out.columns:
        df_out['pred_proba'] = 0.5  # default dummy value
    print(f'[DEBUG][preprocess] df_out shape: {df_out.shape}')
    print(f'[DEBUG][preprocess] target unique: {_safe_unique(df_out[target_col])}')
    for col in feature_cols:
        print(f'[DEBUG][preprocess] {col} unique: {_safe_unique(df_out[col])[:5]}')
    if len(_safe_unique(df_out[target_col])) == 1:
        print(f"[STOP][preprocess] Target มีค่าเดียว: {_safe_unique(df_out[target_col])} หยุด pipeline")
        sys.exit(1)
    out_path = os.path.join('output_default', 'preprocessed_super.parquet')
    df_out.to_parquet(out_path)
    print(f'บันทึกไฟล์ preprocessed_super.parquet ด้วย feature/target ใหม่ ({target_col})')
