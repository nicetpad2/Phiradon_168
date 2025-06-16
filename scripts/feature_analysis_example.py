import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to your Parquet file (M1 timeframe as example)
PARQUET_PATH = os.path.join('data', 'parquet_cache', 'XAUUSD_M1.parquet')

# Load data
df = pd.read_parquet(PARQUET_PATH)

# --- 1. ตรวจสอบ target/label ---
target_col_candidates = [c for c in df.columns if 'target' in c.lower() or 'label' in c.lower()]
if not target_col_candidates:
    print('ไม่พบคอลัมน์ target หรือ label ในไฟล์ข้อมูล')
else:
    target_col = target_col_candidates[0]
    print(f'ใช้ target: {target_col}')
    print(df[target_col].value_counts())
    df[target_col].value_counts().plot(kind='bar', title='Target Distribution')
    plt.show()
    # --- 2. ตรวจสอบ class balance ---
    print('Class balance (สัดส่วนแต่ละ class):')
    print(df[target_col].value_counts(normalize=True))

# --- 3. Feature Engineering ตัวอย่าง ---
# สร้าง Moving Average, RSI, Momentum, Volatility
if 'Close' in df.columns:
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['volatility_10'] = df['Close'].rolling(window=10).std()
    print('เพิ่ม feature: ma_10, ma_50, momentum_10, volatility_10')
    # Plot ตัวอย่าง
    df[['Close','ma_10','ma_50']].plot(figsize=(12,5), title='Close & Moving Averages')
    plt.show()

# --- 4. Correlation Heatmap ---
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# --- 5. Save feature engineered data ---
FE_PATH = os.path.join('data', 'parquet_cache', 'XAUUSD_M1_fe.parquet')
df.to_parquet(FE_PATH)
print(f'บันทึกไฟล์ feature engineered: {FE_PATH}')
