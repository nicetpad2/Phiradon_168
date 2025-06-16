import pandas as pd
import os

FE_PATH = os.path.join('data', 'parquet_cache', 'XAUUSD_M1_fe.parquet')

df = pd.read_parquet(FE_PATH)

print('--- ข้อมูลตัวอย่าง 10 แถวแรก ---')
print(df.head(10))

print('\n--- ขนาดข้อมูล (rows, columns) ---')
print(df.shape)

print('\n--- รายชื่อคอลัมน์ ---')
print(list(df.columns))

print('\n--- สรุปสถิติข้อมูลเชิงตัวเลข ---')
print(df.describe())

# ตรวจสอบ target/label distribution ถ้ามี
label_cols = [c for c in df.columns if 'target' in c.lower() or 'label' in c.lower()]
if label_cols:
    print(f"\n--- Distribution ของ {label_cols[0]} ---")
    print(df[label_cols[0]].value_counts())
else:
    print('\n--- ไม่พบคอลัมน์ target หรือ label ---')
