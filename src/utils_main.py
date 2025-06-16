"""
utils_main.py: รวมฟังก์ชัน util เฉพาะที่ใช้ใน main/pipeline
"""
import os

# ตัวอย่างฟังก์ชัน util ที่อาจย้ายมาไว้ที่นี่ (หากมีใน main.py)
def parse_arguments():
    """Stubbed argument parser."""
    return {}

def setup_output_directory(base_dir, dir_name):
    from src.data_loader import setup_output_directory as dl_setup_output_directory
    return dl_setup_output_directory(base_dir, dir_name)

def load_features_from_file(_):
    return {}

def drop_nan_rows(df):
    return df.dropna()

def convert_to_float32(df):
    return df.astype("float32", errors="ignore")

def run_initial_backtest():
    return None

def save_final_data(df, path):
    # ป้องกัน path ว่างและสร้างโฟลเดอร์ให้เสมอ
    path = path if path else "output_default/final_data.csv"
    dir_path = os.path.dirname(path) or "output_default"
    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(path)
