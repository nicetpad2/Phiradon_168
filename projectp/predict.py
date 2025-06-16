import os
import pandas as pd


def safe_path(path: str, default: str = "output_default") -> str:
    if not path or str(path).strip() == '':
        return default
    return path


def safe_makedirs(path: str):
    path = safe_path(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def save_final_predictions(output_dir, y_true, y_pred, y_proba=None):
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
    print(f"[เทพ] บันทึก prediction/result ที่มีคอลัมน์ครบที่ {out_path} และ {out_csv}")
    # เทพ: export excel, json เพิ่มเติม
    out_xlsx = os.path.join(output_dir, 'final_predictions.xlsx')
    df.to_excel(out_xlsx, index=False)
    out_json = os.path.join(output_dir, 'final_predictions.json')
    df.to_json(out_json, orient='records', lines=True, force_ascii=False)
    print(f"[เทพ] Export เพิ่มเติม: {out_xlsx}, {out_json}")
