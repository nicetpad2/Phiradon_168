import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Path to your Parquet file (M1 timeframe as example)
PARQUET_PATH = os.path.join('data', 'parquet_cache', 'XAUUSD_M1.parquet')
FE_PATH = os.path.join('data', 'parquet_cache', 'XAUUSD_M1_fe_super.parquet')

# Load data
df = pd.read_parquet(PARQUET_PATH)

# --- 1. สร้าง target/label อัตโนมัติ (binary) ---
# ตัวอย่าง: ทำนายทิศทางราคาถัดไป (1=ขึ้น, 0=ไม่ขึ้น)
if 'Close' in df.columns:
    df['target_direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    print('สร้าง target_direction (1=ขึ้น, 0=ไม่ขึ้น) แบบ binary สำเร็จ')
else:
    print('ไม่พบคอลัมน์ Close สำหรับสร้าง target')

# --- 2. Feature Engineering อัตโนมัติ ---
if 'Close' in df.columns:
    # Moving averages (window สั้นลง)
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    # Momentum/returns
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['volatility_5'] = df['Close'].rolling(window=5).std()
    df['return_1'] = df['Close'].pct_change(1)
    df['return_3'] = df['Close'].pct_change(3)
    df['high_low_spread'] = df['High'] - df['Low']
    # --- เทพ: Technical indicators ---
    df['rsi_7'] = ta.rsi(df['Close'], length=7)
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=4)
    if macd is not None:
        df['macd'] = macd['MACD_6_13_4']
        df['macd_signal'] = macd['MACDs_6_13_4']
        df['macd_hist'] = macd['MACDh_6_13_4']
    df['ema_7'] = ta.ema(df['Close'], length=7)
    bbands = ta.bbands(df['Close'], length=7)
    if bbands is not None:
        df['bb_upper'] = bbands['BBU_7_2.0']
        df['bb_lower'] = bbands['BBL_7_2.0']
        df['bb_middle'] = bbands['BBM_7_2.0']
    # Lagged features (ลดเหลือ 1,2,3,5)
    for lag in [1,2,3,5]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'return_lag_{lag}'] = df['Close'].pct_change(lag)
    # Rolling min/max (window สั้นลง)
    df['roll_min_5'] = df['Close'].rolling(window=5).min()
    df['roll_max_5'] = df['Close'].rolling(window=5).max()
    print('เพิ่ม feature อัตโนมัติ: ma, momentum, volatility, return, high_low_spread, RSI, MACD, EMA, BBands, lagged, rolling min/max (window สั้นลง)')

# --- Feature เฉพาะทาง ---
# Session (Asia, London, NY)
def get_session(hour):
    if 0 <= hour < 7:
        return 'Asia'
    elif 7 <= hour < 15:
        return 'London'
    else:
        return 'NY'
if 'Time' in df.columns:
    df['session'] = df['Time'].apply(lambda t: get_session(int(str(t)[:2])))
    df = pd.get_dummies(df, columns=['session'])
    print('เพิ่ม feature session (Asia, London, NY)')
# Volatility regime (high/low)
if 'volatility_5' in df.columns:
    vol_th = df['volatility_5'].median()
    df['vol_regime'] = (df['volatility_5'] > vol_th).astype(int)
    print('เพิ่ม feature volatility regime (high/low)')
# Pattern (bullish/bearish candle)
if 'Open' in df.columns and 'Close' in df.columns:
    df['bullish_candle'] = (df['Close'] > df['Open']).astype(int)
    df['bearish_candle'] = (df['Close'] < df['Open']).astype(int)
    print('เพิ่ม feature pattern (bullish/bearish candle)')

# --- 3. ลบแถวที่มี missing values (NaN) ---
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)
print(f'หลังสร้าง feature ทั้งหมด: ขนาดข้อมูล = {df.shape}, NaN ทั้งหมด = {df.isna().sum().sum()}')

# --- 4. สรุป target distribution ---
print('Distribution ของ target_direction:')
print(df['target_direction'].value_counts())

# Plot distribution ของ target
plt.figure(figsize=(4,2))
df['target_direction'].value_counts().sort_index().plot(kind='bar')
plt.title('Target Direction Distribution')
plt.xlabel('target_direction')
plt.ylabel('count')
plt.tight_layout()
plt.savefig('output_default/target_direction_distribution.png')
plt.close()

# Sanity check: class imbalance
vc = df['target_direction'].value_counts(normalize=True)
if (vc > 0.8).any():
    print('WARNING: พบ class imbalance รุนแรงใน target_direction:', vc.to_dict())

# Correlation matrix (feature กับ target)
feature_cols = [c for c in df.columns if c not in ['target_direction','Date','Time','Symbol'] and df[c].dtype != 'O']
cor = df[feature_cols + ['target_direction']].corr()
print('Correlation matrix (feature กับ target):')
print(cor['target_direction'].sort_values(ascending=False))
cor['target_direction'].sort_values(ascending=False).to_csv('output_default/feature_target_correlation.csv')

# --- Baseline model: logistic regression (multiclass)
X = df[feature_cols].values
y = df['target_direction'].values
try:
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'Baseline LogisticRegression accuracy (train): {acc:.4f}')
    with open('output_default/baseline_logreg_report.txt','w') as f:
        f.write(classification_report(y, y_pred))
except Exception as e:
    print('Baseline model error:', e)

# --- Baseline model: DecisionTree ---
print('--- Baseline DecisionTree ---')
try:
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, y)
    y_pred_dt = dt.predict(X)
    acc_dt = accuracy_score(y, y_pred_dt)
    print(f'Baseline DecisionTree accuracy (train): {acc_dt:.4f}')
except Exception as e:
    print('DecisionTree error:', e)

# --- หลัง baseline model ---
# Feature importance (RandomForest)
print('--- เริ่มขั้นตอน RandomForest ---')
try:
    # ลดขนาดข้อมูลสำหรับ RandomForest
    X_rf = X[:20000]
    y_rf = y[:20000]
    rf = RandomForestClassifier(n_estimators=20, max_depth=7, random_state=42)
    rf.fit(X_rf, y_rf)
    print('RandomForest fit เสร็จสมบูรณ์')
    importances = rf.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print('Top 10 Feature Importances (RandomForest):')
    print(fi.head(10))
    fi.to_csv('output_default/feature_importance_rf.csv')
    print('บันทึกไฟล์ feature_importance_rf.csv สำเร็จ')
    # Plot
    plt.figure(figsize=(8,4))
    fi.head(20).plot(kind='bar')
    plt.title('Top 20 Feature Importances (RandomForest)')
    plt.tight_layout()
    plt.savefig('output_default/feature_importance_rf.png')
    plt.close()
    print('บันทึกไฟล์ feature_importance_rf.png สำเร็จ')
except Exception as e:
    print('Feature importance error:', e)
    import traceback
    with open('output_default/feature_importance_rf_error.txt','w') as f:
        f.write(traceback.format_exc())
print('--- จบขั้นตอน RandomForest ---')

# --- Multi-timeframe feature (เทพ) ---
if 'Close' in df.columns and 'Time' in df.columns and 'Date' in df.columns:
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    df = df.set_index('datetime')
    # M5
    df_m5 = df.resample('5T').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
    df_m5['ma_m5'] = df_m5['Close'].rolling(3).mean()
    df_m5['rsi_m5'] = ta.rsi(df_m5['Close'], length=7)
    df_m5['return_m5'] = df_m5['Close'].pct_change(1)
    df = df.merge(df_m5[['ma_m5','rsi_m5','return_m5']], left_index=True, right_index=True, how='left')
    # M15
    df_m15 = df.resample('15T').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
    df_m15['ma_m15'] = df_m15['Close'].rolling(3).mean()
    df_m15['rsi_m15'] = ta.rsi(df_m15['Close'], length=7)
    df_m15['return_m15'] = df_m15['Close'].pct_change(1)
    df = df.merge(df_m15[['ma_m15','rsi_m15','return_m15']], left_index=True, right_index=True, how='left')
    print('เพิ่ม multi-timeframe feature (M5, M15) สำเร็จ')
    df = df.reset_index(drop=True)

# --- Seasonality feature ---
if 'datetime' in df.columns:
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek
    print('เพิ่ม seasonality feature (hour, dayofweek)')

# --- Target event-based (เทพ): return 5 แท่งข้างหน้า > threshold ---
if 'Close' in df.columns:
    threshold = 0.0015  # 15 pip (สำหรับทอง XAUUSD)
    df['target_event'] = (df['Close'].shift(-5) / df['Close'] - 1 > threshold).astype(int)
    print('สร้าง target_event (return 5 แท่งข้างหน้า > 0.15%) สำเร็จ')

# --- 5. Save feature engineered + target data ---
df.to_parquet(FE_PATH)
print(f'บันทึกไฟล์ feature engineered + target: {FE_PATH}')
