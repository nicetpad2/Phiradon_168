import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from projectp.plot import plot_metrics_summary, plot_predictions
import yaml
import requests

def load_trading_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/settings.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('trading', {})

def send_trade_order(symbol: str, side: str, qty: float, price: str | None = None, comment: str | None = None) -> dict[str, str]:
    cfg = load_trading_config()
    if not cfg.get('enabled', False):
        return {'status': 'disabled', 'msg': 'Trading integration disabled'}
    payload: dict[str, str | float | None] = {
        'symbol': symbol,
        'side': side,
        'qty': qty,
        'price': price,
        'comment': comment,
        'mode': cfg.get('mode', 'demo'),
        'exchange': cfg.get('exchange', 'binance')
    }
    try:
        resp = requests.post(cfg.get('webhook_url'), json=payload, timeout=5)
        return {'status': 'sent', 'resp': resp.text}
    except Exception as e:
        return {'status': 'error', 'msg': str(e)}

def main():
    st.set_page_config(page_title="ProjectP Dashboard", layout="wide")
    st.title("📊 ProjectP เทรด AI Dashboard")
    output_dir = "output_default"
    metrics_csv = os.path.join(output_dir, "metrics_summary_v32.csv")
    pred_parquet = os.path.join(output_dir, "final_predictions.parquet")
    st.sidebar.header("🔧 ตัวเลือก")
    st.sidebar.write(f"Output Dir: `{output_dir}`")
    # ปุ่มเทรด/Integration
    st.sidebar.header("🚀 Integration/Trading")
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.rerun()
    if st.sidebar.button("💸 Execute Trade (Demo)"):
        st.success("[DEMO] ส่งคำสั่งเทรดสำเร็จ! (เชื่อมต่อ API จริงได้)")
    st.sidebar.markdown("---")
    st.sidebar.write("**Integration API:** สามารถเชื่อมต่อ exchange/broker ผ่าน Webhook, REST, หรือ Python SDK ได้ทันที\nตัวอย่าง: Binance, Bitkub, Alpaca, Interactive Brokers, MetaTrader, ฯลฯ")
    # Metrics summary
    st.header("Metrics Summary")
    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        st.dataframe(df)
        # Plot metrics bar
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
        values = [df.get(m, [None])[0] for m in metrics]
        sns.barplot(x=metrics, y=values, palette="viridis", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Metrics Summary")
        st.pyplot(fig)
        # Plot confusion matrix
        if 'confusion_matrix' in df.columns:
            import ast
            cm = ast.literal_eval(df['confusion_matrix'][0])
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
            ax2.set_title("Confusion Matrix")
            st.pyplot(fig2)
    else:
        st.warning(f"ไม่พบ metrics summary: {metrics_csv}")
    # Prediction analysis
    st.header("Prediction Analysis")
    if os.path.exists(pred_parquet):
        dfp = pd.read_parquet(pred_parquet)
        st.dataframe(dfp.head(100))
        if 'pred_proba' in dfp.columns:
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.histplot(dfp['pred_proba'], bins=30, kde=True, ax=ax3)
            ax3.set_title("Prediction Probability Distribution")
            st.pyplot(fig3)
        if 'target' in dfp.columns and 'pred' in dfp.columns:
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            ax4.scatter(dfp['target'], dfp['pred'], alpha=0.2)
            ax4.set_xlabel("Target")
            ax4.set_ylabel("Predicted")
            ax4.set_title("Target vs Predicted Scatter")
            st.pyplot(fig4)
    else:
        st.warning(f"ไม่พบ prediction file: {pred_parquet}")
    st.info("💡 คุณสามารถนำผลลัพธ์ไปวิเคราะห์, plot, หรือเชื่อมต่อระบบเทรดจริงได้ทันที!")
    trading_cfg = load_trading_config()
    st.sidebar.header('⚡ Trading/Integration')
    st.sidebar.write(f"Mode: {trading_cfg.get('mode','demo').upper()} | Exchange: {trading_cfg.get('exchange','binance').capitalize()}")
    symbol = st.sidebar.text_input('Symbol', 'XAUUSD')
    side = st.sidebar.selectbox('Side', ['BUY', 'SELL'])
    qty = st.sidebar.number_input('Qty', min_value=0.01, value=1.0)
    price = st.sidebar.text_input('Price (optional)', '')
    comment = st.sidebar.text_input('Comment', '')
    if st.sidebar.button('🚀 Execute Trade (Real)'):
        result = send_trade_order(symbol, side, qty, price or None, comment)
        if result['status'] == 'sent':
            st.sidebar.success('ส่งคำสั่งเทรดจริงสำเร็จ!')
        elif result['status'] == 'disabled':
            st.sidebar.warning('Trading integration ถูกปิดใช้งานใน config')
        else:
            st.sidebar.error(f"Error: {result['msg']}")
    st.sidebar.markdown('---')
    st.sidebar.write('**Integration API:** รองรับ REST/webhook/SDK, เพิ่มเติมใน settings.yaml')
    st.sidebar.write('**Real-time/Backtest:** (Demo toggle)')
    mode = st.sidebar.radio('Dashboard Mode', ['Real-time', 'Backtest', 'Demo'], index=0)
    st.session_state['dashboard_mode'] = mode
    st.sidebar.info(f'โหมดปัจจุบัน: {mode}')
    # --- Test Set Section ---
    st.header("🧪 Test Set Evaluation (เทพ)")
    test_pred_csv = os.path.join(output_dir, "test_predictions.csv")
    test_metrics_json = os.path.join(output_dir, "test_metrics.json")
    test_cm_png = os.path.join(output_dir, "test_confusion_matrix.png")
    test_hist_png = os.path.join(output_dir, "test_pred_proba_hist.png")
    if os.path.exists(test_pred_csv):
        test_pred_df = pd.read_csv(test_pred_csv)
        st.subheader("Test Predictions (head)")
        st.dataframe(test_pred_df.head(100))
    if os.path.exists(test_metrics_json):
        import json
        with open(test_metrics_json, 'r', encoding='utf-8') as f:
            test_metrics = json.load(f)
        st.subheader("Test Metrics")
        st.write(f"AUC: {test_metrics.get('auc', None):.4f}")
        st.write("Classification Report:")
        st.json(test_metrics.get('report', {}))
        st.write("Confusion Matrix:")
        st.write(test_metrics.get('confusion_matrix', []))
    if os.path.exists(test_cm_png):
        st.subheader("Test Confusion Matrix (กราฟ)")
        st.image(test_cm_png)
    if os.path.exists(test_hist_png):
        st.subheader("Test Prediction Probability Histogram")
        st.image(test_hist_png)

    # --- Real-time Monitor ---
    st.sidebar.header('⏱️ Real-time Monitor')
    auto_refresh = st.sidebar.checkbox('Auto-refresh (5s)', value=False)
    if auto_refresh:
        import time
        st.experimental_rerun()
        time.sleep(5)

    # --- Test Set Drilldown ---
    st.header('🔬 Test Set Drilldown')
    if os.path.exists(test_pred_csv):
        test_pred_df = pd.read_csv(test_pred_csv)
        st.subheader('Filter/Sort/Search Test Predictions')
        # Filter by prediction probability
        min_proba, max_proba = float(test_pred_df['y_pred_proba'].min()), float(test_pred_df['y_pred_proba'].max())
        proba_range = st.slider('Prediction Probability Range', min_value=min_proba, max_value=max_proba, value=(min_proba, max_proba), step=0.01)
        filtered = test_pred_df[(test_pred_df['y_pred_proba'] >= proba_range[0]) & (test_pred_df['y_pred_proba'] <= proba_range[1])]
        # Filter by true/false positive/negative
        filter_type = st.selectbox('Filter by Type', ['All', 'True Positive', 'True Negative', 'False Positive', 'False Negative'])
        if filter_type != 'All':
            if filter_type == 'True Positive':
                filtered = filtered[(filtered['y_true'] == 1) & (filtered['y_pred'] == 1)]
            elif filter_type == 'True Negative':
                filtered = filtered[(filtered['y_true'] == 0) & (filtered['y_pred'] == 0)]
            elif filter_type == 'False Positive':
                filtered = filtered[(filtered['y_true'] == 0) & (filtered['y_pred'] == 1)]
            elif filter_type == 'False Negative':
                filtered = filtered[(filtered['y_true'] == 1) & (filtered['y_pred'] == 0)]
        # Search by row (orig index)
        search_row = st.text_input('ค้นหา row (index เดิม, comma-separated)', '')
        if search_row:
            try:
                idxs = [int(i.strip()) for i in search_row.split(',') if i.strip().isdigit()]
                filtered = filtered[filtered['row'].isin(idxs)]
            except Exception:
                st.warning('Row index format ไม่ถูกต้อง')
        st.dataframe(filtered.head(200))
        # Drilldown plot: scatter true vs pred, histogram, error cases
        st.subheader('Drilldown Plots')
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6,3))
        sns.scatterplot(x=filtered['y_true'], y=filtered['y_pred_proba'], hue=filtered['y_pred'], alpha=0.5, ax=ax)
        ax.set_title('True vs Predicted Probability (Drilldown)')
        st.pyplot(fig)
        fig2, ax2 = plt.subplots(figsize=(6,3))
        sns.histplot(filtered['y_pred_proba'], bins=30, kde=True, ax=ax2)
        ax2.set_title('Filtered Prediction Probability Histogram')
        st.pyplot(fig2)
        st.info('💡 สามารถ filter/sort/plot ข้อมูล test set ได้ละเอียดแบบเทพ!')

if __name__ == "__main__":
    main()
