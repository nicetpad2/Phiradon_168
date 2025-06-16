import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.emoji import Emoji
from rich.layout import Layout
from rich.text import Text
import hashlib
import datetime
import traceback
import requests

console = Console()

def hash_file(path: str) -> str:
    if not os.path.exists(path): return ''
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def export_audit_report(output_dir, metrics, pred_file):
    audit = {
        'datetime': datetime.datetime.now().isoformat(),
        'metrics_file': os.path.join(output_dir, "metrics_summary_v32.csv"),
        'metrics_hash': hash_file(os.path.join(output_dir, "metrics_summary_v32.csv")),
        'pred_file': pred_file,
        'pred_hash': hash_file(pred_file),
        'cwd': os.getcwd(),
        'python_version': os.sys.version,
        'env': dict(os.environ),
    }
    audit_path = os.path.join(output_dir, "metrics_audit_report.json")
    import json
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    console.print(f"[green][Audit] Export audit report: {audit_path}")

def notify_line(msg: str, token: str = None):
    """แจ้งเตือนผ่าน LINE Notify (option)"""
    if not token:
        return
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': msg}
    try:
        requests.post(url, headers=headers, data=data, timeout=5)
    except Exception:
        pass

def upload_to_webhook(file_path: str, webhook_url: str = None):
    """อัปโหลดไฟล์ผลลัพธ์ไปยัง webhook (option)"""
    if not webhook_url or not os.path.exists(file_path):
        return
    try:
        with open(file_path, 'rb') as f:
            requests.post(webhook_url, files={'file': f}, timeout=10)
    except Exception:
        pass

def print_advanced_summary(metrics):
    """แสดง summary table/section แบบเทพ พร้อมสี/ไอคอน/next step"""
    table = Table(title="[bold blue]Metrics Summary (เทพ)", show_lines=True)
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="magenta", justify="left")
    for k, v in metrics.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        elif isinstance(v, dict) or isinstance(v, list):
            continue
        table.add_row(str(k), str(v))
    console.print(table)
    if metrics.get('auc', 0) > 0.7:
        console.print("[green]AUC ดีมาก! พร้อม deploy/production")
    elif metrics.get('auc', 0) < 0.6:
        console.print("[yellow]AUC ต่ำ อาจต้องปรับ feature/model")
    if metrics.get('support_0', 0) < 10 or metrics.get('support_1', 0) < 10:
        console.print("[yellow]Class imbalance/test set เล็ก ควรเพิ่มข้อมูล")
    if metrics.get('mcc', 0) < 0.1:
        console.print("[red]MCC ต่ำมาก อาจมี leakage หรือ class imbalance")
    console.print("[cyan]ดู HTML dashboard: metrics_summary_v32.html (option)")

def print_beautiful_summary(metrics):
    """สรุปผลแบบ infographic/section/emoji/ASCII Art/next step"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=2),
        Layout(name="footer", size=5)
    )
    # Header: ASCII Art + Emoji
    header = Align.center(Text("""
██████╗ ██╗   ██╗██╗     ██╗     ██████╗ ██╗██████╗ ██╗     ██╗███╗   ██╗
██╔══██╗██║   ██║██║     ██║     ██╔══██╗██║██╔══██╗██║     ██║████╗  ██║
██████╔╝██║   ██║██║     ██║     ██║  ██║██║██████╔╝██║     ██║██╔██╗ ██║
██╔═══╝ ██║   ██║██║     ██║     ██║  ██║██║██╔═══╝ ██║     ██║██║╚██╗██║
██║     ╚██████╔╝███████╗███████╗██████╔╝██║██║     ███████╗██║██║ ╚████║
╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═════╝ ╚═╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝
""", style="bold magenta"), vertical="middle")
    header2 = Align.center(Text(Emoji.replace(":trophy: :chart_with_upwards_trend: :star2: :rocket:"), style="bold yellow"))
    layout["header"].update(Panel(header, title="[bold cyan]Full Pipeline Summary", border_style="magenta"))
    # Body: Metrics Table
    table = Table(title="[bold green]Key Metrics", show_lines=True, box=None)
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="magenta", justify="left")
    for k, v in metrics.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        elif isinstance(v, dict) or isinstance(v, list):
            continue
        emoji = ":white_check_mark:" if (isinstance(v, str) and float(v) > 0.7) else ":warning:"
        table.add_row(str(k), f"{Emoji.replace(emoji)} {v}")
    layout["body"].update(Panel(table, border_style="green"))
    # Footer: Next Step/Advice
    advice = ""
    if metrics.get('auc', 0) > 0.7:
        advice += "[green]AUC ดีมาก! พร้อม deploy/production\n"
    elif metrics.get('auc', 0) < 0.6:
        advice += "[yellow]AUC ต่ำ อาจต้องปรับ feature/model\n"
    if metrics.get('support_0', 0) < 10 or metrics.get('support_1', 0) < 10:
        advice += "[yellow]Class imbalance/test set เล็ก ควรเพิ่มข้อมูล\n"
    if metrics.get('mcc', 0) < 0.1:
        advice += "[red]MCC ต่ำมาก อาจมี leakage หรือ class imbalance\n"
    advice += "[cyan]ดู HTML dashboard: metrics_summary_v32.html (option)\n"
    advice += "[bold magenta]Next: วิเคราะห์ผล, ปรับ threshold, หรือ deploy AI!"
    layout["footer"].update(Panel(advice, border_style="yellow", title="[bold blue]Next Steps & Advice"))
    console.print(layout)

def auto_health_check(output_dir):
    """ตรวจสอบ health ของระบบ/ไฟล์/ข้อมูล/โมเดล/feature แบบเทพ"""
    from rich.progress import Progress
    from rich.panel import Panel
    import glob
    import pandas as pd
    import os
    issues = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Health Check: Files", total=5)
        # 1. Check main files
        files = [
            os.path.join(output_dir, 'preprocessed_super.parquet'),
            os.path.join(output_dir, 'metrics_summary_v32.csv'),
            os.path.join(output_dir, 'final_predictions.parquet'),
        ]
        for f in files:
            if not os.path.exists(f):
                issues.append(f"[❌] Missing file: {f}")
            progress.update(task, advance=1)
        # 2. Check config
        config_path = 'config/settings.yaml'
        if not os.path.exists(config_path):
            issues.append(f"[❌] Missing config: {config_path}")
        else:
            import yaml
            try:
                with open(config_path, encoding='utf-8') as f:
                    yaml.safe_load(f)
            except Exception as e:
                issues.append(f"[❌] Config error: {e}")
        progress.update(task, advance=1)
        # 3. Check feature/model
        feature_path = 'features_main.json'
        if not os.path.exists(feature_path):
            issues.append(f"[❌] Missing feature list: {feature_path}")
        model_files = glob.glob(os.path.join(output_dir, '*.joblib')) + glob.glob(os.path.join(output_dir, '*.pkl'))
        if not model_files:
            issues.append(f"[❌] Missing model file in {output_dir}")
        progress.update(task, advance=1)
        # 4. Check data health
        try:
            df = pd.read_parquet(os.path.join(output_dir, 'preprocessed_super.parquet'))
            nans = df.isnull().sum().sum()
            dups = df.duplicated().sum()
            if nans > 0:
                issues.append(f"[⚠️] Missing values: {nans}")
            if dups > 0:
                issues.append(f"[⚠️] Duplicate rows: {dups}")
            if df.shape[0] < 100:
                issues.append(f"[⚠️] Data rows too few: {df.shape[0]}")
        except Exception as e:
            issues.append(f"[❌] Data error: {e}")
        progress.update(task, advance=1)
        # 5. Check prediction/metrics
        try:
            dfm = pd.read_csv(os.path.join(output_dir, 'metrics_summary_v32.csv'))
            if 'auc' in dfm.columns and dfm['auc'].iloc[0] < 0.6:
                issues.append(f"[⚠️] Low AUC: {dfm['auc'].iloc[0]:.4f}")
        except Exception as e:
            issues.append(f"[❌] Metrics error: {e}")
        progress.update(task, advance=1)
    # สรุปผล health check
    if issues:
        console.print(Panel("\n".join(issues), title="[red]Health Check: พบปัญหา/ข้อควรระวัง!", border_style="red"))
    else:
        console.print(Panel("[green]ระบบสมบูรณ์ทุกจุด!", title="[green]Health Check: OK", border_style="green"))
    return issues

def check_resource_efficiency():
    """ตรวจสอบการใช้ RAM/GPU และแนะนำการเพิ่มประสิทธิภาพ/ลดคอขวด"""
    import psutil
    import platform
    import os
    from rich.panel import Panel
    issues = []
    # RAM usage
    mem = psutil.virtual_memory()
    if mem.percent > 80:
        issues.append(f"[⚠️] RAM ใช้งานสูง {mem.percent:.1f}% (Total: {mem.total/1e9:.1f} GB)")
    # Swap usage
    swap = psutil.swap_memory()
    if swap.percent > 50:
        issues.append(f"[⚠️] Swap ใช้งานสูง {swap.percent:.1f}% (อาจช้า/คอขวด)")
    # GPU usage (ถ้ามี)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            if gpu.memoryUtil > 0.8:
                issues.append(f"[⚠️] GPU {gpu.name} memory ใช้งานสูง {gpu.memoryUtil*100:.1f}% ({gpu.memoryUsed:.1f}MB/{gpu.memoryTotal:.1f}MB)")
            if gpu.load > 0.95:
                issues.append(f"[⚠️] GPU {gpu.name} load สูง {gpu.load*100:.1f}%")
    except Exception:
        pass
    # Platform info
    sysinfo = f"OS: {platform.system()} {platform.release()} | Python: {platform.python_version()}"
    # แนะนำ optimization
    advice = []
    if mem.percent > 80 or swap.percent > 50:
        advice.append("[💡] ลดขนาด batch, ใช้ chunk, หรือเพิ่ม RAM")
    if issues:
        console.print(Panel("\n".join(issues)+"\n"+"\n".join(advice), title="[yellow]Resource Efficiency/คอขวด", border_style="yellow"))
    else:
        console.print(Panel(f"[green]RAM/GPU usage OK\n{sysinfo}", title="[green]Resource Efficiency", border_style="green"))

def memory_efficient_read_parquet(path, columns=None):
    import pandas as pd
    return pd.read_parquet(path, columns=columns, engine='pyarrow')

# --- Parallel/Distributed Compute (Dask) ---
def dask_read_parquet(path, columns=None):
    try:
        import dask.dataframe as dd
        return dd.read_parquet(path, columns=columns)
    except ImportError:
        return None

# --- GPU DataFrame (cuDF) ---
def cudf_read_parquet(path, columns=None):
    try:
        import cudf
        return cudf.read_parquet(path, columns=columns)
    except ImportError:
        return None

# --- AutoML/Hyperparameter Optimization (Optuna) ---
def run_optuna_automl(X, y):
    try:
        import optuna
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 16)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            return cross_val_score(clf, X, y, cv=3, scoring='roc_auc').mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        return study.best_params, study.best_value
    except ImportError:
        return None, None

# --- Experiment Tracking (MLflow) ---
def log_metrics_mlflow(metrics, params=None):
    try:
        import mlflow
        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except ImportError:
        pass

# --- Advanced Visualization (Plotly) ---
def plotly_metrics_bar(metrics):
    try:
        import plotly.graph_objects as go
        keys = [k for k in metrics if isinstance(metrics[k], (int, float))]
        vals = [metrics[k] for k in keys]
        fig = go.Figure([go.Bar(x=keys, y=vals)])
        fig.update_layout(title='Metrics Bar (Plotly)', template='plotly_dark')
        fig.write_html('output_default/metrics_bar_plotly.html')
    except ImportError:
        pass

# --- Realtime Alert (Slack) ---
def notify_slack(msg, webhook_url=None):
    import requests
    if webhook_url:
        try:
            requests.post(webhook_url, json={'text': msg}, timeout=5)
        except Exception:
            pass

# --- Self-Healing/Retry Decorator ---
def self_heal_retry(func):
    import functools
    def wrapper(*args, **kwargs):
        for _ in range(2):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                console.print(f"[yellow][Self-heal] Retry after error: {e}")
        return None
    return functools.wraps(func)(wrapper)

# --- Next Step Suggestion ---
def print_next_steps(metrics):
    advice = []
    if metrics.get('auc', 0) > 0.7:
        advice.append("[green]AUC ดีมาก! พร้อม deploy/production")
    else:
        advice.append("[yellow]AUC ยังต่ำ ลองปรับ feature/model หรือใช้ AutoML")
    advice.append("[cyan]ลองใช้ Dask/cuDF/Optuna/MLflow/Plotly/Slack integration เพิ่มเติม!")
    advice.append("[magenta]Next: วิเคราะห์ผล, ปรับ threshold, deploy, หรือเชื่อมต่อ production API")
    console.print("\n".join(advice))

def ensure_metrics_summary(output_dir):
    metrics_path = os.path.join(output_dir, "metrics_summary_v32.csv")
    pred_files = glob.glob(os.path.join(output_dir, '*_strategy_result.parquet'))
    progress = Progress(console=console)
    with progress:
        task = progress.add_task("[cyan]ตรวจสอบไฟล์ prediction/result...", total=5)
        if not pred_files:
            progress.console.print("[red][metrics] ไม่พบไฟล์ *_strategy_result.parquet ใน {}".format(output_dir))
            fallback_pred = os.path.join(output_dir, 'final_predictions.parquet')
            if os.path.exists(fallback_pred):
                pred_files = [fallback_pred]
            else:
                progress.console.print("[red][metrics] ไม่พบไฟล์ prediction/result ใด ๆ ใน {}".format(output_dir))
                progress.console.print("[yellow][metrics] สร้าง metrics_summary_v32.csv แบบ minimal (autocreated)")
                df_metrics = pd.DataFrame([{'metric': 0.0, 'note': 'autocreated'}])
                df_metrics.to_csv(metrics_path, index=False)
                progress.update(task, advance=5)
                return
        progress.update(task, advance=1)
        pred_file = max(pred_files, key=os.path.getmtime)
        try:
            df = pd.read_parquet(pred_file)
        except Exception as e:
            progress.console.print(f"[red][metrics] อ่านไฟล์ prediction error: {e}")
            df_metrics = pd.DataFrame([{'metric': 0.0, 'note': f'autocreated: {e}'}])
            df_metrics.to_csv(metrics_path, index=False)
            progress.update(task, advance=4)
            return
        progress.update(task, advance=1)
        col_target = col_pred = col_proba = None
        for c in df.columns:
            if c.lower() in ["target", "target_direction", "label", "target_event", "target_buy_sell_hold"]:
                col_target = c
            if c.lower() in ["pred", "prediction", "pred_label", "buy_signal", "sell_signal", "hold_signal"]:
                col_pred = c
            if c.lower() in ["pred_proba", "proba", "probability"]:
                col_proba = c
        if col_target is None or col_pred is None:
            progress.console.print(f"[red][metrics] ไม่พบคอลัมน์ target/pred ใน {pred_file}")
            df_metrics = pd.DataFrame([{'metric': 0.0, 'note': 'autocreated: missing target/pred'}])
            df_metrics.to_csv(metrics_path, index=False)
            progress.update(task, advance=3)
            return
        progress.update(task, advance=1)
        y_true = df[col_target]
        y_pred = df[col_pred]
        y_proba = df[col_proba] if col_proba else None
        y_true_valid = y_true[y_true != -1] if -1 in pd.Series(y_true).unique() else y_true
        y_pred_valid = y_pred[y_true != -1] if -1 in pd.Series(y_true).unique() else y_pred
        from sklearn.metrics import matthews_corrcoef, classification_report
        average_type = 'binary' if set(pd.Series(y_true_valid).unique()) <= {0, 1} else 'macro'
        try:
            metrics = {
                'accuracy': accuracy_score(y_true_valid, y_pred_valid),
                'precision': precision_score(y_true_valid, y_pred_valid, zero_division=0, average=average_type),
                'recall': recall_score(y_true_valid, y_pred_valid, zero_division=0, average=average_type),
                'f1': f1_score(y_true_valid, y_pred_valid, zero_division=0, average=average_type),
                'auc': roc_auc_score(y_true_valid, y_proba) if y_proba is not None and average_type == 'binary' else '',
                'mcc': matthews_corrcoef(y_true_valid, y_pred_valid),
                'support_0': (y_true_valid == 0).sum(),
                'support_1': (y_true_valid == 1).sum(),
                'confusion_matrix': confusion_matrix(y_true_valid, y_pred_valid).tolist(),
                'report': classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0, digits=4)
            }
            progress.console.print("[green][metrics] คำนวณ metrics สำเร็จ!")
        except Exception as e:
            progress.console.print(f"[red][metrics] {e}\n{traceback.format_exc()}")
            metrics = {'metric': 0.0, 'note': f'autocreated: {e}'}
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(metrics_path, index=False)
        progress.update(task, advance=1)
        # Export JSON
        metrics_json_path = os.path.join(output_dir, "metrics_summary_v32.json")
        import json
        import numpy as np
        def convert_np(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(v) for v in obj]
            else:
                return obj
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(convert_np(metrics), f, ensure_ascii=False, indent=2)
        progress.console.print(f"[green][metrics] Export metrics summary JSON: {metrics_json_path}")
        # Export audit report
        export_audit_report(output_dir, metrics, pred_file)
        # Advanced: export HTML summary (option)
        try:
            import dataframe_image as dfi
            html_path = os.path.join(output_dir, "metrics_summary_v32.html")
            dfi.export(df_metrics, html_path)
            progress.console.print(f"[green][metrics] Export HTML summary: {html_path}")
        except Exception:
            pass
        # Adaptive suggestion
        if metrics.get('support_0', 0) < 10 or metrics.get('support_1', 0) < 10:
            progress.console.print("[yellow][metrics] [คำแนะนำ] class imbalance หรือ test set เล็กมาก! ควรเพิ่มข้อมูลหรือปรับ threshold")
        if metrics.get('auc', 0) < 0.6:
            progress.console.print("[yellow][metrics] [คำแนะนำ] AUC ต่ำ อาจ underfit หรือข้อมูลไม่พอ ลองปรับ feature/model")
        if metrics.get('mcc', 0) < 0.1:
            progress.console.print("[yellow][metrics] [คำแนะนำ] MCC ต่ำมาก อาจมีปัญหา leakage หรือ class imbalance")
        progress.console.print(Panel.fit("[bold green]Metrics summary/report/audit/export ครบทุก format!", border_style="green"))
        try:
            # ฟีเจอร์เทพ: แสดง summary แบบเทพ
            print_advanced_summary(metrics)
            print_beautiful_summary(metrics)
            plotly_metrics_bar(metrics)
            log_metrics_mlflow(metrics)
            notify_slack(f"[ProjectP] Metrics summary: AUC={metrics.get('auc',0):.4f}", os.environ.get('SLACK_WEBHOOK_URL'))
            print_next_steps(metrics)
            # แจ้งเตือนผ่าน LINE Notify (option)
            line_token = os.environ.get('LINE_NOTIFY_TOKEN')
            if line_token:
                notify_line(f"[ProjectP] Metrics summary: AUC={metrics.get('auc',0):.4f} | MCC={metrics.get('mcc',0):.4f}", line_token)
            # อัปโหลดไฟล์ไป webhook (option)
            webhook_url = os.environ.get('PROJECTP_WEBHOOK_URL')
            if webhook_url:
                upload_to_webhook(metrics_path, webhook_url)
            # Self-healing: retry minimal export หาก export หลักล้มเหลว
        except Exception as e:
            console.print(f"[red][metrics] Export/notify error: {e}")
            # retry minimal export
            try:
                df_metrics = pd.DataFrame([{'metric': 0.0, 'note': f'autocreated: {e}'}])
                df_metrics.to_csv(metrics_path, index=False)
                console.print("[yellow][metrics] Retry: สร้าง metrics_summary_v32.csv แบบ minimal")
            except Exception:
                pass
    check_resource_efficiency()
    health_issues = auto_health_check(output_dir)
    if health_issues:
        console.print("[yellow][Auto-heal] พบปัญหา/ข้อควรระวังในระบบ! Pipeline จะพยายามดำเนินการต่อแบบ robust...")
    # --- Dask/cuDF/AutoML/MLflow/Plotly/Slack integration ---
    # 1. ลองอ่าน parquet แบบ Dask/GPU
    ddf = dask_read_parquet(os.path.join(output_dir, 'preprocessed_super.parquet'))
    gdf = cudf_read_parquet(os.path.join(output_dir, 'preprocessed_super.parquet'))
    # 2. AutoML (Optuna) ตัวอย่าง
    try:
        import pandas as pd
        df = pd.read_parquet(os.path.join(output_dir, 'preprocessed_super.parquet'))
        feature_cols = [c for c in df.columns if c not in ['target','target_event','target_direction','pred_proba']]
        X, y = df[feature_cols], df[df.columns[-1]]
        best_params, best_auc = run_optuna_automl(X, y)
        if best_params:
            console.print(f"[blue][AutoML] Best params: {best_params} | AUC: {best_auc:.4f}")
    except Exception:
        pass
    try:
        # ฟีเจอร์เทพ: แสดง summary แบบเทพ
        print_advanced_summary(metrics)
        print_beautiful_summary(metrics)
        plotly_metrics_bar(metrics)
        log_metrics_mlflow(metrics)
        notify_slack(f"[ProjectP] Metrics summary: AUC={metrics.get('auc',0):.4f}", os.environ.get('SLACK_WEBHOOK_URL'))
        print_next_steps(metrics)
        # --- Production/Enterprise Integration ---
        export_metrics_badge(metrics)
        log_metrics_enterprise(metrics)
        upload_metrics_to_cloud(os.path.join(output_dir, 'metrics_summary_v32.csv'), provider='s3', bucket=os.environ.get('S3_BUCKET'), key='metrics_summary_v32.csv')
        send_metrics_to_queue(metrics)
        # API: Uncomment to start REST API
        # start_metrics_api()
        # ...existing code...
    except Exception as e:
        console.print(f"[red][metrics] Export/notify error: {e}")
        # retry minimal export
        try:
            df_metrics = pd.DataFrame([{'metric': 0.0, 'note': f'autocreated: {e}'}])
            df_metrics.to_csv(metrics_path, index=False)
            console.print("[yellow][metrics] Retry: สร้าง metrics_summary_v32.csv แบบ minimal")
        except Exception:
            pass

# --- Production REST API (FastAPI) ---
def start_metrics_api():
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
        import pandas as pd
        app = FastAPI(title="ProjectP Metrics API")
        @app.get("/metrics")
        def get_metrics():
            try:
                df = pd.read_csv('output_default/metrics_summary_v32.csv')
                return JSONResponse(df.to_dict(orient='records')[0])
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except ImportError:
        console.print("[yellow]FastAPI/uvicorn not installed. API not started.")

# --- Async/Queue Integration (Celery/RabbitMQ) ---
def send_metrics_to_queue(metrics):
    try:
        from celery import Celery
        app = Celery('projectp', broker='pyamqp://guest@localhost//')
        app.send_task('projectp.tasks.process_metrics', args=[metrics])
        console.print("[green]Metrics sent to queue (Celery/RabbitMQ)")
    except ImportError:
        pass

# --- Cloud Storage Integration (S3/GCS/Azure) ---
def upload_metrics_to_cloud(file_path, provider='s3', bucket=None, key=None):
    try:
        if provider == 's3':
            import boto3
            s3 = boto3.client('s3')
            s3.upload_file(file_path, bucket, key or file_path)
            console.print(f"[green]Uploaded to S3: s3://{bucket}/{key or file_path}")
        elif provider == 'gcs':
            from google.cloud import storage
            client = storage.Client()
            bucket_obj = client.bucket(bucket)
            blob = bucket_obj.blob(key or file_path)
            blob.upload_from_filename(file_path)
            console.print(f"[green]Uploaded to GCS: gs://{bucket}/{key or file_path}")
        elif provider == 'azure':
            from azure.storage.blob import BlobServiceClient
            blob_service = BlobServiceClient.from_connection_string(os.environ['AZURE_STORAGE_CONNECTION_STRING'])
            blob_client = blob_service.get_blob_client(container=bucket, blob=key or file_path)
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            console.print(f"[green]Uploaded to Azure Blob: {bucket}/{key or file_path}")
    except ImportError:
        pass

# --- CI/CD Integration (GitHub Actions/Badge) ---
def export_metrics_badge(metrics):
    try:
        auc = metrics.get('auc', 0)
        color = 'green' if auc > 0.7 else 'yellow' if auc > 0.6 else 'red'
        badge = f"https://img.shields.io/badge/AUC-{auc:.2f}-{color}"
        with open('output_default/metrics_badge_url.txt', 'w') as f:
            f.write(badge)
        console.print(f"[blue]Metrics badge URL: {badge}")
    except Exception:
        pass

# --- Enterprise Logging (ELK/Datadog/Splunk) ---
def log_metrics_enterprise(metrics):
    try:
        import requests
        elk_url = os.environ.get('ELK_METRICS_URL')
        if elk_url:
            requests.post(elk_url, json=metrics, timeout=5)
            console.print("[green]Metrics logged to ELK/Datadog/Splunk endpoint")
    except Exception:
        pass
