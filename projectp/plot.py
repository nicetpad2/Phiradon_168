import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_summary(metrics_csv_path: str, output_dir: str = "output_default"):
    """
    Auto-plot metrics summary (accuracy, precision, recall, f1, auc, mcc) as bar chart and confusion matrix.
    """
    if not os.path.exists(metrics_csv_path):
        print(f"[plot] ไม่พบไฟล์ metrics summary: {metrics_csv_path}")
        return
    df = pd.read_csv(metrics_csv_path)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
    values = [df.get(m, [None])[0] for m in metrics]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=metrics, y=values, palette="viridis")
    plt.title("Metrics Summary")
    plt.ylim(0, 1)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_barplot.png")
    plt.savefig(out_path)
    print(f"[plot] บันทึกกราฟ metrics barplot ที่ {out_path}")
    # Plot confusion matrix if available
    if 'confusion_matrix' in df.columns:
        import ast
        cm = ast.literal_eval(df['confusion_matrix'][0])
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        out_cm = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(out_cm)
        print(f"[plot] บันทึก confusion matrix ที่ {out_cm}")

def plot_predictions(pred_path: str, output_dir: str = "output_default"):
    """
    Auto-plot prediction/probability distribution and target/pred scatter.
    """
    if not os.path.exists(pred_path):
        print(f"[plot] ไม่พบไฟล์ prediction: {pred_path}")
        return
    df = pd.read_parquet(pred_path) if pred_path.endswith('.parquet') else pd.read_csv(pred_path)
    if 'pred_proba' in df.columns:
        plt.figure(figsize=(6, 3))
        sns.histplot(df['pred_proba'], bins=30, kde=True)
        plt.title("Prediction Probability Distribution")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "pred_proba_hist.png")
        plt.savefig(out_path)
        print(f"[plot] บันทึกกราฟ pred_proba distribution ที่ {out_path}")
    if 'target' in df.columns and 'pred' in df.columns:
        plt.figure(figsize=(6, 3))
        plt.scatter(df['target'], df['pred'], alpha=0.2)
        plt.title("Target vs Predicted Scatter")
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "target_pred_scatter.png")
        plt.savefig(out_path)
        print(f"[plot] บันทึกกราฟ target vs pred scatter ที่ {out_path}")
