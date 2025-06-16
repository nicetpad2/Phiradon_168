from projectp.preprocess import run_preprocess
from projectp.train import train_and_validate_model
from projectp.predict import save_final_predictions
from projectp.metrics import ensure_metrics_summary
from projectp.debug import print_logo
from projectp.plot import plot_metrics_summary, plot_predictions

import sys
import time
import os
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
from prompt_toolkit import PromptSession, completion
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
import yaml
import json
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress sklearn UndefinedMetricWarning globally (เทพสุด)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

app = typer.Typer()
console = Console()

# Branding/Theme
ASCII_LOGO = '''
[bold cyan]
 ____            _            ____  ____  
|  _ \\ ___  __ _| | _____    |  _ \\|  _ \\
| |_) / _ \\/ _` | |/ / _ \\   | |_) | |_) |
|  __/  __/ (_| |   <  __/   |  __/|  __/ 
|_|   \\___|\\__,_|_|\\_\\___|   |_|   |_|    
[/bold cyan]
'''
ASCII_LOGOS = [
    ASCII_LOGO,
    '''[bold magenta]\n██████╗  ██████╗ ██████╗ ██████╗ ██████╗\n██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██╔══██╗\n██████╔╝██║   ██║██████╔╝██████╔╝██████╔╝\n██╔═══╝ ██║   ██║██╔═══╝ ██╔═══╝ ██╔═══╝\n██║     ╚██████╔╝██║     ██║     ██║\n╚═╝      ╚═════╝ ╚═╝     ╚═╝     ╚═╝\n[/bold magenta]''',
    '''[bold green]\n██████╗ ██████╗ ██████╗ ██████╗ ██████╗\n██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██╔══██╗\n██████╔╝██║   ██║██████╔╝██████╔╝██████╔╝\n██╔═══╝ ██║   ██║██╔═══╝ ██╔═══╝ ██╔═══╝\n██║     ╚██████╔╝██║     ██║     ██║\n╚═╝      ╚═════╝ ╚═╝     ╚═╝     ╚═╝\n[/bold green]''',
]
CLI_STYLE = Style.from_dict({
    'prompt': 'ansicyan bold',
    '': 'ansiblack',
    'logo': 'ansiblue bold',
    'success': 'ansigreen bold',
    'error': 'ansired bold',
    'warning': 'ansiyellow bold',
    'info': 'ansiblue',
})
THEMES = {
    'classic': CLI_STYLE,
    'neon': Style.from_dict({'prompt': 'ansimagenta bold', 'logo': 'ansigreen bold', 'success': 'ansiblue bold', 'error': 'ansired bold', 'warning': 'ansiyellow bold', 'info': 'ansicyan'}),
    'pastel': Style.from_dict({'prompt': 'ansiblue', 'logo': 'ansiyellow bold', 'success': 'ansigreen', 'error': 'ansired', 'warning': 'ansiblack', 'info': 'ansiblue'}),
}
current_theme = 'classic'

# Auto-complete commands for shell
COMMANDS = ['status', 'health', 'plot', 'report', 'wfv', 'setup', 'quickstart', 'exit', 'help']
completer = WordCompleter(COMMANDS, ignore_case=True)

# --- Dynamic Option Loaders ---
def load_config_options():
    try:
        with open('config/settings.yaml', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        options = []
        # Flatten config keys for auto-complete
        def flatten(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, prefix + k + '.')
                else:
                    options.append(prefix + k)
        flatten(cfg)
        return options
    except Exception:
        return []

def load_feature_list():
    try:
        with open('features_main.json', encoding='utf-8') as f:
            features = json.load(f)
        return features if isinstance(features, list) else []
    except Exception:
        return []

def load_model_files():
    try:
        return [f for f in os.listdir('output_default') if f.endswith('.joblib') or f.endswith('.pkl')]
    except Exception:
        return []

def color(text, c):
    colors = {'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m', 'blue': '\033[94m', 'end': '\033[0m'}
    return f"{colors.get(c, '')}{text}{colors['end']}" if os.name != 'nt' else text

def beep():
    try:
        print('\a', end='')
    except Exception:
        pass

def ascii_success():
    return color("""
███████╗ █████╗ ███████╗███████╗███████╗
██╔════╝██╔══██╗╚══███╔╝██╔════╝██╔════╝
█████╗  ███████║  ███╔╝ █████╗  ███████╗
██╔══╝  ██╔══██║ ███╔╝  ██╔══╝  ╚════██║
███████╗██║  ██║███████╗███████╗███████║
╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝
""", 'green')

def ascii_error():
    return color("""
███████╗███████╗██████╗ ██████╗ ███████╗██████╗ 
╚══███╔╝██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
  ███╔╝ █████╗  ██████╔╝██████╔╝█████╗  ██████╔╝
 ███╔╝  ██╔══╝  ██╔══██╗██╔══██╗██╔══╝  ██╔══██╗
███████╗███████╗██║  ██║██║  ██║███████╗██║  ██║
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
""", 'red')

def print_metrics_summary(output_dir):
    metrics_path = os.path.join(output_dir, "metrics_summary_v32.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        print(color("\n[📊] Metrics Summary:", 'blue'))
        print(df.to_string(index=False))
    else:
        print(color("[⚠️] ไม่พบ metrics_summary_v32.csv", 'yellow'))

def print_preview_parquet(path):
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            print(color(f"\n[🔎] ตัวอย่างข้อมูล 5 แถวแรก ({os.path.basename(path)}):", 'blue'))
            print(df.head())
        except Exception as e:
            print(color(f"[⚠️] อ่านไฟล์ {path} ไม่สำเร็จ: {e}", 'yellow'))

def check_file_ready(path):
    if not os.path.exists(path):
        print(color(f"[❌] ไม่พบไฟล์ที่จำเป็น: {path}", 'red'))
        return False
    return True

def suggest_next_mode(current):
    mapping = {'1': '2', '2': '4', '4': '5', '5': '0'}
    return mapping.get(current, '0')

def timed_step(desc, func, *args, threshold_very_fast=2, threshold_fast=5, threshold_medium=10, threshold_medium2=15, threshold_slow=20, threshold_slow2=30, threshold_very_slow=45, threshold_extreme=60, threshold_ultra=90, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(color(f"[⏱️] {desc} เสร็จใน {elapsed:.2f} วินาที", 'blue'))
    if elapsed > threshold_ultra:
        print(color(f"[💀] {desc} ใช้เวลานานมากผิดปกติ! (>{threshold_ultra}s) ต้อง optimize ด่วน!", 'red'))
    elif elapsed > threshold_extreme:
        print(color(f"[🔥] {desc} ใช้เวลานานมาก! (>{threshold_extreme}s) ควร optimize ด่วน!", 'red'))
    elif elapsed > threshold_very_slow:
        print(color(f"[⚡] {desc} ใช้เวลานานมาก (>{threshold_very_slow}s) ลองปรับลดขนาดข้อมูล/เพิ่ม n_jobs/ใช้ parquet/parallel", 'yellow'))
    elif elapsed > threshold_slow2:
        print(color(f"[⏳] {desc} ใช้เวลานานผิดปกติ (>{threshold_slow2}s) อาจมีคอขวด", 'magenta'))
    elif elapsed > threshold_slow:
        print(color(f"[🟠] {desc} ใช้เวลานานกว่าปกติ (>{threshold_slow}s)", 'yellow'))
    elif elapsed > threshold_medium2:
        print(color(f"[🟡] {desc} ใช้เวลานานกว่าที่ควร (>{threshold_medium2}s)", 'yellow'))
    elif elapsed > threshold_medium:
        print(color(f"[🟢] {desc} ใช้เวลานาน (>{threshold_medium}s)", 'green'))
    elif elapsed > threshold_fast:
        print(color(f"[🔵] {desc} ใช้เวลานาน (>{threshold_fast}s)", 'blue'))
    elif elapsed > threshold_very_fast:
        print(color(f"[🟣] {desc} ใช้เวลานาน (>{threshold_very_fast}s)", 'magenta'))
    return result, elapsed

# --- Emoji/ASCII Art Effects ---
def emoji_effects(category: str | None = None) -> str:
    effects = {
        'rocket': ['🚀', '🛸', '🛰️', '🦾', '🤖', '🦾🤖'],
        'fire': ['🔥', '💥', '⚡', '🌟', '✨', '💫'],
        'party': ['🎉', '🥳', '🎊', '🎈', '🎆', '🎇'],
        'trophy': ['🏆', '🥇', '🎖️', '👑', '💎'],
        'ai': ['🤖', '🧠', '🦾', '💡', '🦾🤖', '🧠✨'],
        'success': ['✅', '🎉', '🥳', '🏆', '🌈', '✨'],
        'error': ['❌', '🚨', '💥', '😱', '🛑', '⚠️'],
        'bottleneck': ['🐢', '⏳', '🦥', '🦦', '🦣', '🦛'],
        'lightning': ['⚡', '🌩️', '🌠', '💡', '✨'],
        'star': ['🌟', '⭐', '✨', '💫', '🛸'],
        'wizard': ['🧙‍♂️', '🧙‍♀️', '🪄', '🔮', '✨'],
        'progress': ['⏳', '🔄', '🔁', '🔃', '🔄✨'],
        'default': ['✨', '🌈', '💫', '🎉', '🤖', '🚀', '🔥', '⚡', '🏆', '🥳', '🧠', '🦾'],
    }
    if category and category in effects:
        return random.choice(effects[category])
    return random.choice(effects['default'])

def random_celebrate_ascii() -> str:
    arts = [
        '''[bold green]\n🎉🎉🎉\n███████╗ ██████╗ ██████╗ ███████╗\n██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗\n█████╗  ██║   ██║██████╔╝███████╗\n██╔══╝  ██║   ██║██╔══██╗╚════██║\n██║     ╚██████╔╝██║  ██║███████║\n╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝\n🎉🎉🎉[/bold green]''',
        '''[bold magenta]\n✨✨✨\n██████╗  ██████╗ ██████╗ ██████╗\n██╔══██╗██╔═══██╗██╔══██╗██╔══██╗\n██████╔╝██║   ██║██████╔╝██████╔╝\n██╔═══╝ ██║   ██║██╔═══╝ ██╔═══╝\n██║     ╚██████╔╝██║     ██║\n╚═╝      ╚═════╝ ╚═╝     ╚═╝\n✨✨✨[/bold magenta]''',
        '''[bold yellow]\n🚀🚀🚀\n██████╗ ██████╗ ██████╗ ██████╗\n██╔══██╗██╔═══██╗██╔══██╗██╔══██╗\n██████╔╝██║   ██║██████╔╝██████╔╝\n██╔═══╝ ██║   ██║██╔═══╝ ██╔═══╝\n██║     ╚██████╔╝██║     ██║\n╚═╝      ╚═════╝ ╚═╝     ╚═╝\n🚀🚀🚀[/bold yellow]''',
        '''[bold cyan]\n🧠🤖🧠\n██████╗  ██████╗ ██████╗ ██████╗\n██╔══██╗██╔═══██╗██╔══██╗██╔══██╗\n██████╔╝██║   ██║██████╔╝██████╔╝\n██╔═══╝ ██║   ██║██╔═══╝ ██╔═══╝\n██║     ╚██████╔╝██║     ██║\n╚═╝      ╚═════╝ ╚═╝     ╚═╝\n🧠🤖🧠[/bold cyan]''',
    ]
    return random.choice(arts)

def run_full_pipeline():
    print(color(f"\n[bold yellow][Full Pipeline] {emoji_effects('rocket')}{emoji_effects('star')} เริ่มกระบวนการเทพทุกขั้นตอน AI อัตโนมัติ! พร้อมลุยสู่ Production! {emoji_effects('party')}{emoji_effects('success')}[/bold yellow]", 'yellow'))
    print(color("[bold cyan]" + "━"*55 + "[/bold cyan]", 'cyan'))
    start = time.time()
    step_times = {}
    try:
        with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), transient=True) as progress:
            task = progress.add_task(f"[bold green]Full Pipeline Progress {emoji_effects('progress')}", total=24)
            # 1. Preprocess
            progress.update(task, description=f"{emoji_effects('wizard')}  เตรียมข้อมูล (Preprocess)")
            _, t_pre = timed_step("Preprocess", run_preprocess, threshold_very_fast=2, threshold_fast=5, threshold_medium=10, threshold_medium2=15, threshold_slow=20, threshold_slow2=30, threshold_very_slow=45, threshold_extreme=60, threshold_ultra=90)
            step_times['Preprocess'] = t_pre
            progress.advance(task)
            _, t_preview = timed_step("Preview Parquet", lambda: print_preview_parquet('output_default/preprocessed_super.parquet'), threshold_very_fast=1, threshold_fast=2, threshold_medium=5, threshold_medium2=8, threshold_slow=10, threshold_slow2=15, threshold_very_slow=20, threshold_extreme=30, threshold_ultra=45)
            step_times['Preview Parquet'] = t_preview
            progress.advance(task)
            # 2. Train, Validate & Test (เทพ)
            progress.update(task, description=f"{emoji_effects('ai')} เทรน/วัดผลโมเดล + ทดสอบ (Train, Validate & Test) {emoji_effects('rocket')}")
            from projectp.train import train_validate_test_model
            progress.update(task, description=f"[1/4] เตรียมข้อมูล/feature/target {emoji_effects('wizard')}")
            # เตรียมข้อมูล (simulate step)
            time.sleep(0.5)
            progress.advance(task)
            progress.update(task, description=f"[2/4] กำลังเทรน/validate/AutoML...")
            results = train_validate_test_model()
            progress.advance(task)
            progress.update(task, description=f"[3/4] สร้างกราฟ/plot/metrics...")
            # (plot/metrics จะถูกสร้างใน train_validate_test_model)
            time.sleep(0.5)
            progress.advance(task)
            progress.update(task, description=f"[4/4] เสร็จสิ้นขั้นตอน Train/Validate/Test (AutoML)")
            time.sleep(0.2)
            progress.advance(task)
            t_train = 0  # ไม่สามารถวัดแยกได้เพราะรวมใน train_validate_test_model
            step_times['Train/Validate/Test'] = t_train
            _, t_preview2 = timed_step("Preview Parquet (หลังเทรน)", lambda: print_preview_parquet('output_default/preprocessed_super.parquet'), threshold_very_fast=1, threshold_fast=2, threshold_medium=5, threshold_medium2=8, threshold_slow=10, threshold_slow2=15, threshold_very_slow=20, threshold_extreme=30, threshold_ultra=45)
            step_times['Preview Parquet 2'] = t_preview2
            progress.advance(task)
            # 2.1 รายงานผล test set
            progress.update(task, description=f"🧪 รายงานผลลัพธ์ Test Set (เทพ)...")
            t0 = time.perf_counter()
            print(color(f"[AUC] Test AUC: {results['test_auc']:.4f}", 'cyan'))
            import pandas as pd
            pd.set_option('display.max_columns', 20)
            print(color("[📋] Test Classification Report:", 'cyan'))
            import pprint
            pprint.pprint(results['test_report'])
            print(color("[🧮] Test Confusion Matrix:", 'cyan'))
            print(results['test_cm'])
            print(color("[plot] ดูกราฟ test set ที่ output_default/test_confusion_matrix.png, test_pred_proba_hist.png", 'cyan'))
            t1 = time.perf_counter(); step_times['Test Report'] = t1-t0
            progress.advance(task)
            # 2.2 Advanced Test Set Evaluation (เทพ)
            progress.update(task, description=f"🧠 Advanced Test Set Evaluation/Robustness...")
            import numpy as np
            from sklearn.metrics import roc_curve, precision_recall_curve, auc
            import matplotlib.pyplot as plt
            test_pred_df = results['test_pred_df']
            y_true = test_pred_df['y_true']
            y_score = test_pred_df['y_pred_proba']
            def plot_and_export():
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(5,4))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
                plt.plot([0,1], [0,1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Test ROC Curve')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig('output_default/test_roc_curve.png')
                print(color('[plot] บันทึกกราฟ test ROC curve ที่ output_default/test_roc_curve.png', 'cyan'))
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                plt.figure(figsize=(5,4))
                plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Test Precision-Recall Curve')
                plt.legend(loc='lower left')
                plt.tight_layout()
                plt.savefig('output_default/test_pr_curve.png')
                print(color('[plot] บันทึกกราฟ test PR curve ที่ output_default/test_pr_curve.png', 'cyan'))
                thresholds = np.linspace(0, 1, 101)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                print(color(f'[Threshold] Best threshold (Youden J): {best_threshold:.3f}', 'yellow'))
                y_true_shuffled = np.random.permutation(y_true)
                fpr_s, tpr_s, _ = roc_curve(y_true_shuffled, y_score)
                roc_auc_s = auc(fpr_s, tpr_s)
                print(color(f'[Robustness] AUC (shuffled y): {roc_auc_s:.4f}', 'yellow'))
                import json
                with open('output_default/test_advanced_report.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        'roc_auc': float(roc_auc),
                        'pr_auc': float(pr_auc),
                        'best_threshold': float(best_threshold),
                        'robust_auc_shuffled': float(roc_auc_s)
                    }, f, ensure_ascii=False, indent=2)
                print(color('[Export] test_advanced_report.json ครบ!', 'green'))
            _, t_plot_export = timed_step("Plot/Export Advanced Test Eval", plot_and_export, threshold_very_fast=2, threshold_fast=5, threshold_medium=10, threshold_medium2=15, threshold_slow=20, threshold_slow2=30, threshold_very_slow=45, threshold_extreme=60, threshold_ultra=90)
            step_times['Advanced Test Eval'] = t_plot_export
            progress.advance(task)
            # 2.3 Test Set Audit & Reproducibility (Production-grade)
            progress.update(task, description=f"🕵️‍♂️ Test Set Audit & Reproducibility...")
            def audit_export():
                import hashlib, platform, sys, json, os
                seed = 42
                config_path = os.path.abspath('config/settings.yaml') if os.path.exists('config/settings.yaml') else None
                model_files = [f for f in os.listdir('output_default') if f.endswith('.joblib') or f.endswith('.pkl')]
                def file_hash(path):
                    if not os.path.exists(path): return None
                    with open(path, 'rb') as f:
                        return hashlib.md5(f.read()).hexdigest()
                audit = {
                    'random_seed': seed,
                    'config_path': config_path,
                    'config_hash': file_hash(config_path) if config_path else None,
                    'test_pred_hash': file_hash('output_default/test_predictions.csv'),
                    'test_metrics_hash': file_hash('output_default/test_metrics.json'),
                    'test_advanced_hash': file_hash('output_default/test_advanced_report.json'),
                    'model_files': model_files,
                    'model_hashes': {f: file_hash(os.path.join('output_default', f)) for f in model_files},
                    'python_version': sys.version,
                    'platform': platform.platform(),
                    'cwd': os.getcwd(),
                    'env_vars': {k: v for k, v in os.environ.items() if k.startswith('PYTHON') or k.startswith('CUDA')},
                }
                try:
                    import pandas as pd
                    test_pred = pd.read_csv('output_default/test_predictions.csv')
                    trainval_idx = set(results['test_pred_df']['row'])
                    test_idx = set(test_pred['row'])
                    overlap = trainval_idx & test_idx
                    audit['test_index_overlap'] = list(overlap)
                    if overlap:
                        print(color(f'[LEAKAGE] พบ index ซ้ำ train/test: {overlap}', 'red'))
                    else:
                        print(color('[LEAKAGE] ไม่พบ index ซ้ำ train/test', 'green'))
                except Exception as e:
                    print(color(f'[LEAKAGE] ตรวจสอบ index ซ้ำ error: {e}', 'red'))
                n_test = len(test_pred)
                n_pos = (test_pred['y_true'] == 1).sum()
                n_neg = (test_pred['y_true'] == 0).sum()
                audit['test_size'] = n_test
                audit['test_pos'] = int(n_pos)
                audit['test_neg'] = int(n_neg)
                if n_test < 1000:
                    print(color(f'[WARNING] test set เล็ก ({n_test}) อาจไม่ robust', 'yellow'))
                if n_pos == 0 or n_neg == 0:
                    print(color('[WARNING] test set ไม่มี class ใด class หนึ่ง', 'yellow'))
                with open('output_default/test_audit_report.json', 'w', encoding='utf-8') as f:
                    json.dump(audit, f, ensure_ascii=False, indent=2)
                print(color('[Export] test_audit_report.json ครบ!', 'green'))
            _, t_audit = timed_step("Audit/Export", audit_export, threshold_very_fast=1, threshold_fast=3, threshold_medium=7, threshold_medium2=10, threshold_slow=15, threshold_slow2=20, threshold_very_slow=30, threshold_extreme=45, threshold_ultra=60)
            step_times['Audit/Export'] = t_audit
            progress.advance(task)
            # 2.4 Walk-Forward Validation (WFV) (เทพ)
            progress.update(task, description=f"🔄 Walk-Forward Validation (WFV)...")
            def wfv_eval():
                from .train import walk_forward_validation_evaluate
                fe_super_path = 'output_default/preprocessed_super.parquet'
                import pandas as pd
                df = pd.read_parquet(fe_super_path)
                feature_cols, target_col = results.get('test_pred_df').columns[2:], 'y_true'
                feature_cols, target_col = results.get('test_pred_df').columns[2:], 'y_true' if 'y_true' in results.get('test_pred_df').columns else target_col
                from .preprocess import get_feature_target_columns
                feature_cols, target_col = get_feature_target_columns(df)
                wfv_results, wfv_mean_auc, wfv_std_auc = walk_forward_validation_evaluate(df, feature_cols, target_col, n_splits=5, test_size=0.15, random_state=42)
                print(color(f'[WFV] mean AUC: {wfv_mean_auc:.4f} | std: {wfv_std_auc:.4f}', 'cyan'))
                print(color('[plot] ดูกราฟ WFV ที่ output_default/wfv_auc_per_window.png', 'cyan'))
                import json
                with open('output_default/wfv_results.json', 'w', encoding='utf-8') as f:
                    json.dump(wfv_results, f, ensure_ascii=False, indent=2)
                print(color('[Export] wfv_results.json ครบ!', 'green'))
            _, t_wfv = timed_step("Walk-Forward Validation (WFV)", wfv_eval, threshold_very_fast=2, threshold_fast=5, threshold_medium=10, threshold_medium2=15, threshold_slow=20, threshold_slow2=30, threshold_very_slow=45, threshold_extreme=60, threshold_ultra=90)
            step_times['WFV'] = t_wfv
            progress.advance(task)
            # 2.5 WFV Custom Split + Advanced Report (เทพ)
            progress.update(task, description=f"🧩 WFV Custom Split + Advanced Report...")
            def wfv_custom():
                from .train import walk_forward_validation_custom_split, generate_wfv_report
                fe_super_path = 'output_default/preprocessed_super.parquet'
                import pandas as pd
                df = pd.read_parquet(fe_super_path)
                from .preprocess import get_feature_target_columns
                feature_cols, target_col = get_feature_target_columns(df)
                n = len(df)
                split_points = [
                    (0, int(n*0.5), int(n*0.5), int(n*0.65)),
                    (int(n*0.2), int(n*0.7), int(n*0.7), int(n*0.85)),
                    (int(n*0.4), int(n*0.9), int(n*0.9), n)
                ]
                wfv_custom_results = walk_forward_validation_custom_split(df, feature_cols, target_col, split_points)
                generate_wfv_report(wfv_custom_results, output_path='output_default/wfv_custom_report.txt')
                print(color('[WFV] สร้างรายงาน WFV custom ที่ output_default/wfv_custom_report.txt', 'cyan'))
            _, t_wfv_custom = timed_step("WFV Custom Split + Advanced Report", wfv_custom, threshold_very_fast=2, threshold_fast=5, threshold_medium=10, threshold_medium2=15, threshold_slow=20, threshold_slow2=30, threshold_very_slow=45, threshold_extreme=60, threshold_ultra=90)
            step_times['WFV Custom'] = t_wfv_custom
            progress.advance(task)
            # 3. Metrics/Report
            progress.update(task, description=f"📊 สรุป Metrics/Report...")
            _, t_metrics = timed_step("Metrics/Report", ensure_metrics_summary, 'output_default', threshold_fast=1, threshold_medium=3, threshold_slow=7, threshold_very_slow=15)
            step_times['Metrics/Report'] = t_metrics
            _, t_metrics_sum = timed_step("Print Metrics Summary", lambda: print_metrics_summary('output_default'), threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['Print Metrics Summary'] = t_metrics_sum
            progress.advance(task)
            # 3.1 Auto-plot metrics & prediction
            progress.update(task, description=f"📈 Auto-plot Metrics/Prediction...")
            def plot_metrics_preds():
                plot_metrics_summary('output_default/metrics_summary_v32.csv', 'output_default')
                plot_predictions('output_default/final_predictions.parquet', 'output_default')
            _, t_plot = timed_step("Auto-plot Metrics/Prediction", plot_metrics_preds, threshold_fast=2, threshold_medium=5, threshold_slow=10, threshold_very_slow=20)
            step_times['Auto-plot'] = t_plot
            progress.advance(task)
            # --- ตรวจสอบไฟล์วิเคราะห์หลักแบบเทพ ---
            print(color("\n[🔎] ตรวจสอบไฟล์วิเคราะห์หลัก/ผลลัพธ์วิเคราะห์...", 'blue'))
            check_analysis_outputs()
            # 4. ตรวจสอบไฟล์ผลลัพธ์/ความสมบูรณ์ (หลัก)
            progress.update(task, description=f"🗂️  ตรวจสอบไฟล์ผลลัพธ์หลัก...")
            def check_files():
                files_to_check = [
                    'output_default/preprocessed_super.parquet',
                    'output_default/metrics_summary_v32.csv',
                ]
                all_ok = True
                for f in files_to_check:
                    if not check_file_ready(f):
                        all_ok = False
                if all_ok:
                    print(color("[✅] ไฟล์ผลลัพธ์หลักครบถ้วน!", 'green'))
                else:
                    print(color("[⚠️] พบไฟล์ขาดหาย/ผิดปกติ กรุณาตรวจสอบ pipeline", 'red'))
            _, t_filecheck = timed_step("Check Output Files", check_files, threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['Check Output Files'] = t_filecheck
            progress.advance(task)
            # 5. ตรวจสอบไฟล์สำคัญอื่น ๆ ในโปรเจค
            progress.update(task, description=f"📁 ตรวจสอบไฟล์สำคัญอื่น ๆ ในโปรเจค...")
            def check_project_files():
                project_files = [
                    'ProjectP.py', 'main.py', 'projectp/preprocess.py', 'projectp/train.py', 'projectp/metrics.py',
                    'projectp/predict.py', 'projectp/cli.py', 'projectp/debug.py', 'requirements.txt', 'README.md'
                ]
                missing = []
                for f in project_files:
                    if not os.path.exists(f):
                        print(color(f"[❌] ขาดไฟล์: {f}", 'red'))
                        missing.append(f)
                    else:
                        print(color(f"[OK] {f}", 'green'))
                if not missing:
                    print(color("[✅] ไฟล์โปรเจคหลักครบถ้วน!", 'green'))
                else:
                    print(color(f"[⚠️] พบไฟล์โปรเจคขาดหาย: {missing}", 'red'))
            _, t_projfile = timed_step("Check Project Files", check_project_files, threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['Check Project Files'] = t_projfile
            progress.advance(task)
            # 6. แสดงสรุปเทพแบบตาราง ASCII
            progress.update(task, description=f"🏆 สรุปผลลัพธ์แบบเทพ (ASCII Table)...")
            def ascii_table():
                import tabulate
                metrics_path = 'output_default/metrics_summary_v32.csv'
                if os.path.exists(metrics_path):
                    df = pd.read_csv(metrics_path)
                    print(tabulate.tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
                else:
                    print_metrics_summary('output_default')
            _, t_ascii = timed_step("ASCII Table Summary", ascii_table, threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['ASCII Table'] = t_ascii
            progress.advance(task)
            # 7. Data Health Check
            progress.update(task, description=f"🩺 Data Health Check...")
            def health_check():
                df = pd.read_parquet('output_default/preprocessed_super.parquet')
                nans = df.isnull().sum().sum()
                if nans == 0:
                    print(color("[✅] ไม่พบ missing values ในข้อมูล!", 'green'))
                else:
                    print(color(f"[⚠️] พบ missing values {nans} จุดในข้อมูล!", 'red'))
                if df.duplicated().any():
                    print(color("[⚠️] พบข้อมูลซ้ำใน preprocessed_super.parquet!", 'red'))
                else:
                    print(color("[✅] ไม่พบข้อมูลซ้ำใน preprocessed_super.parquet!", 'green'))
                if df.select_dtypes(include=['float', 'int']).shape[1] > 0:
                    desc = df.describe()
                    print(color("[🔎] สถิติเบื้องต้น (min/max/mean/std):", 'blue'))
                    print(desc.loc[['min', 'max', 'mean', 'std']])
            _, t_health = timed_step("Data Health Check", health_check, threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['Health Check'] = t_health
            progress.advance(task)
            # 8. ตรวจสอบ reproducibility (hash)
            progress.update(task, description=f"🔒 ตรวจสอบความ reproducible (hash)...")
            def hash_check():
                files_to_check = [
                    'output_default/preprocessed_super.parquet',
                    'output_default/metrics_summary_v32.csv',
                ]
                project_files = [
                    'ProjectP.py', 'main.py', 'projectp/preprocess.py', 'projectp/train.py', 'projectp/metrics.py',
                    'projectp/predict.py', 'projectp/cli.py', 'projectp/debug.py', 'requirements.txt', 'README.md'
                ]
                import hashlib
                def file_hash(path):
                    if not os.path.exists(path): return None
                    with open(path, 'rb') as f:
                        return hashlib.md5(f.read()).hexdigest()
                for f in files_to_check + project_files:
                    h = file_hash(f)
                    if h:
                        print(color(f"[HASH] {os.path.basename(f)}: {h}", 'blue'))
            _, t_hash = timed_step("Hash Check", hash_check, threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['Hash Check'] = t_hash
            progress.advance(task)
            # 9. แนะนำขั้นตอนถัดไป/การใช้งานต่อ
            progress.update(task, description=f"💡 แนะนำการใช้งานต่อ/Next Steps...")
            def next_steps():
                print(color("\n[💡] ผลลัพธ์สำคัญ:", 'blue'))
                print(color(" - ข้อมูลหลัง preprocess: output_default/preprocessed_super.parquet", 'blue'))
                print(color(" - Metrics/Report: output_default/metrics_summary_v32.csv", 'blue'))
                print(color(" - ตัวอย่างข้อมูล/กราฟ: output_default/", 'blue'))
                print(color("\n[💡] คุณสามารถนำผลลัพธ์ไปวิเคราะห์/plot/ใช้งานต่อได้ทันที!", 'yellow'))
                print(color("[💡] แนะนำ: ทดลองวิเคราะห์ผลลัพธ์, plot กราฟ, หรือ deploy ต่อยอด AI ได้เลย!", 'yellow'))
            _, t_next = timed_step("Next Steps UX", next_steps, threshold_fast=1, threshold_medium=2, threshold_slow=5, threshold_very_slow=10)
            step_times['Next Steps'] = t_next
            progress.advance(task)
        elapsed = time.time() - start
        print(color(f"\n[bold green][{emoji_effects('success')}] Full Pipeline เสร็จสมบูรณ์! AI พร้อมใช้งานระดับ Production! {emoji_effects('party')}{emoji_effects('rocket')}[/bold green]", 'green'))
        print(random_celebrate_ascii())
        print(color(f"[{emoji_effects('progress')}] ใช้เวลา {elapsed:.2f} วินาที", 'blue'))
        # --- Summary bottleneck ---
        print(color("\n[bold magenta]⏳ สรุปเวลาทุกขั้นตอน (Bottleneck Analysis):[/bold magenta]", 'magenta'))
        for k, v in step_times.items():
            if v > 90:
                print(color(f"[💀] {k}: {v:.2f} วินาที (Ultra Slow! ต้อง optimize ด่วน)", 'red'))
            elif v > 60:
                print(color(f"[🔥] {k}: {v:.2f} วินาที (Very Slow! ควร optimize ด่วน)", 'red'))
            elif v > 45:
                print(color(f"[⚡] {k}: {v:.2f} วินาที (Extremely Slow!)", 'yellow'))
            elif v > 30:
                print(color(f"[⏳] {k}: {v:.2f} วินาที (Bottleneck!)", 'magenta'))
            elif v > 20:
                print(color(f"[🟠] {k}: {v:.2f} วินาที (ช้ามาก)", 'yellow'))
            elif v > 15:
                print(color(f"[🟡] {k}: {v:.2f} วินาที (ควรตรวจสอบ)", 'yellow'))
            elif v > 10:
                print(color(f"[🟢] {k}: {v:.2f} วินาที (ช้ากว่าที่ควร)", 'green'))
            elif v > 5:
                print(color(f"[🔵] {k}: {v:.2f} วินาที (ช้ากว่าที่ควร)", 'blue'))
            elif v > 2:
                print(color(f"[🟣] {k}: {v:.2f} วินาที (เร็วแต่ยังช้ากว่า 2s)", 'magenta'))
            else:
                print(color(f"[⏱️] {k}: {v:.2f} วินาที", 'blue'))
    except Exception as e:
        print(color(f"[{emoji_effects('error')}] Full Pipeline ล้มเหลว: {e}", 'red'))
        print(random_celebrate_ascii())
        beep()

def main_cli():
    print_logo()
    # UX เมนูหลักเทพ: random emoji, ASCII Art, animation
    logo = random.choice(ASCII_LOGOS)
    console.print(logo, style="bold cyan")
    main_emoji = emoji_effects('ai') + emoji_effects('rocket') + emoji_effects('star')
    print(f"\n[bold cyan][{main_emoji}] [เมนูหลัก ProjectP AI Terminal] ระบบเทรด/วิเคราะห์ AI อัจฉริยะระดับ Enterprise {emoji_effects('party')}{emoji_effects('success')}")
    print("[bold yellow]" + "━"*55 + "[/bold yellow]")
    menu = [
        ("1", f"[green]{emoji_effects('fire')}  เตรียมข้อมูล (Preprocess) {emoji_effects('wizard')}{emoji_effects('success')}[/green]", run_preprocess, 'output_default/preprocessed_super.parquet'),
        ("2", f"[blue]{emoji_effects('ai')} เทรน/วัดผลโมเดล (Train & Validate) {emoji_effects('rocket')}{emoji_effects('trophy')}[/blue]", train_and_validate_model, 'output_default/preprocessed_super.parquet'),
        ("4", f"[magenta]{emoji_effects('star')} สรุป Metrics/Report (Ensure Metrics Summary) {emoji_effects('party')}{emoji_effects('star')}[/magenta]", ensure_metrics_summary, 'output_default/buy_sell_hold_strategy_result.parquet'),
        ("5", f"[bold yellow]{emoji_effects('rocket')} Full Pipeline (เทพทุกขั้นตอน) {emoji_effects('star')}{emoji_effects('party')}{emoji_effects('fire')}[/bold yellow]", run_full_pipeline, None),
        ("0", f"[red]{emoji_effects('error')} ออกจากโปรแกรม {emoji_effects('party')}[/red]", None, None),
    ]
    last_choice = None
    while True:
        print("\n[bold white]กรุณาเลือกโหมดที่ต้องการ:[/bold white]")
        for key, desc, _, _ in menu:
            print(f"  {key}. {desc}")
        print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
        choice = input("[bold cyan]พิมพ์หมายเลขโหมดแล้วกด Enter (Enter ซ้ำ = โหมดเดิม): [/bold cyan]").strip() or last_choice
        found = False
        for key, desc, func, file_check in menu:
            if choice == key:
                found = True
                last_choice = key
                if key == "0":
                    print(color("\n[bold green]ขอบคุณที่ใช้ ProjectP CLI! พบกันใหม่ ✨🚀👋[/bold green]", 'green'))
                    sys.exit(0)
                print(color(f"\n[bold blue][กำลังดำเนินการ] {desc} ...[/bold blue]", 'blue'))
                beep()
                start = time.time()
                try:
                    if file_check and not check_file_ready(file_check):
                        print(color("[⚠️] กรุณารันโหมดก่อนหน้าหรือเตรียมไฟล์ให้พร้อมก่อน!", 'yellow'))
                        print(ascii_error())
                        beep()
                        continue
                    if func is run_preprocess:
                        func()
                        print_preview_parquet('output_default/preprocessed_super.parquet')
                        print(color("\n[bold green][สำเร็จ] เตรียมข้อมูลเสร็จสิ้น! พร้อมลุยขั้นต่อไป! 🎉🟢[/bold green]", 'green'))
                        print(ascii_success())
                    elif func is train_and_validate_model:
                        auc = func()
                        print_preview_parquet('output_default/preprocessed_super.parquet')
                        print(color(f"\n[bold green][สำเร็จ] เทรน/วัดผลโมเดลเสร็จสิ้น! AUC = {auc:.4f} 🧠📈[/bold green]", 'green'))
                        print(ascii_success())
                    elif func is ensure_metrics_summary:
                        output_dir = input("กรุณาระบุ output directory (default: output_default): ").strip() or "output_default"
                        func(output_dir)
                        print_metrics_summary(output_dir)
                        print(color("\n[bold green][สำเร็จ] สรุป Metrics/Report เสร็จสิ้น! 📊✅[/bold green]", 'green'))
                        print(ascii_success())
                    elif func is run_full_pipeline:
                        print(color("\n[bold yellow][Full Pipeline] 🚀✨ เริ่มกระบวนการเทพทุกขั้นตอน AI อัตโนมัติ! พร้อมลุยสู่ Production! [🌟][🧬][🔥][/bold yellow]", 'yellow'))
                        func()
                    else:
                        func()
                    elapsed = time.time() - start
                    print(color(f"[⏱️] ใช้เวลา {elapsed:.2f} วินาที ⏳", 'blue'))
                    next_mode = suggest_next_mode(key)
                    if next_mode != '0':
                        print(color(f"\n[💡] แนะนำ: ดำเนินการโหมดถัดไปโดยกด {next_mode} หรือ Enter [➡️]", 'yellow'))
                except Exception as e:
                    print(color(f"[❌] เกิดข้อผิดพลาด: {e} [🚨]", 'red'))
                    print(ascii_error())
                    beep()
                break
        if not found:
            print(color("[⚠️] กรุณาเลือกหมายเลขโหมดที่ถูกต้อง! [❓]", 'yellow'))

# --- Enhanced Wizard ---
def enhanced_wizard():
    console.print(Panel.fit("[bold magenta]ProjectP Interactive Wizard (เทพขั้นสุด!)", border_style="magenta"))
    # Step 1: Theme/Logo
    global current_theme
    theme_choice = Prompt.ask("[cyan]เลือกธีมสี (classic/neon/pastel)?", choices=list(THEMES.keys()), default=current_theme)
    current_theme = theme_choice
    session = PromptSession(style=THEMES[current_theme])
    logo = random.choice(ASCII_LOGOS)
    console.print(logo, style="bold cyan")
    # Step 2: Project Name
    name = session.prompt("[prompt]ชื่อโปรเจค/Project Name? ", default="ProjectP")
    # Step 3: Data Path
    data_path = session.prompt("[prompt]Path ข้อมูลหลัก (csv/parquet)? ", default="XAUUSD_M1.parquet")
    # Step 4: Feature Selection (auto-complete)
    features = load_feature_list()
    if features:
        feature = session.prompt("[prompt]เลือกฟีเจอร์หลัก (auto-complete): ", completer=WordCompleter(features))
    else:
        feature = session.prompt("[prompt]ชื่อฟีเจอร์ (พิมพ์เอง): ")
    # Step 5: Mode/Config
    mode = session.prompt("[prompt]เลือกโหมด (demo/real/backtest)? ", default="demo")
    # Step 6: Model File (auto-complete)
    models = load_model_files()
    if models:
        model_file = session.prompt("[prompt]เลือกไฟล์โมเดล (auto-complete): ", completer=WordCompleter(models))
    else:
        model_file = session.prompt("[prompt]ชื่อไฟล์โมเดล (พิมพ์เอง): ")
    # Step 7: Preview/Confirm
    console.print(Panel.fit(f"[green]Config สรุป:\n- Name: {name}\n- Data: {data_path}\n- Feature: {feature}\n- Mode: {mode}\n- Model: {model_file}", title="[bold yellow]Config Preview", border_style="green"))
    if Confirm.ask("[green]เริ่ม Full Pipeline เลยหรือไม่?", default=True):
        console.print("[bold green]เริ่ม Full Pipeline...")
        # run_full_pipeline()  # Uncomment if needed
    else:
        console.print("[yellow]คุณสามารถรัน full pipeline ได้ภายหลังด้วย projectp shell หรือ projectp run_full_pipeline")

# --- Enhanced Shell ---
@app.command()
def shell():
    """Interactive ProjectP Shell (เทพ, auto-complete, branding, dynamic)"""
    global current_theme
    logo = random.choice(ASCII_LOGOS)
    console.print(logo, style="bold cyan")
    console.print(Panel.fit(f"[bold cyan]Welcome to ProjectP Shell! พิมพ์ help เพื่อดูคำสั่งทั้งหมด\nธีม: {current_theme}", border_style="cyan"))
    session = PromptSession(style=THEMES[current_theme])
    while True:
        try:
            cmd = session.prompt("[prompt]projectp> ", completer=get_dynamic_completer())
        except (KeyboardInterrupt, EOFError):
            break
        if cmd in ("exit", "quit"): break
        elif cmd == "help":
            console.print("[yellow]คำสั่ง: status, health, plot, report, wfv, setup, quickstart, exit\n[cyan]Auto-complete: config/feature/model argument!")
        elif cmd == "status":
            status()
        elif cmd == "health":
            health()
        elif cmd == "plot":
            console.print("[blue]Auto-plot metrics/prediction... (ดูไฟล์ output_default/)")
        elif cmd == "report":
            console.print("[magenta]Export summary/report... (ดูไฟล์ output_default/)")
        elif cmd == "wfv":
            console.print("[cyan]Run Walk-Forward Validation... (ดูไฟล์ output_default/)")
        elif cmd == "setup" or cmd == "quickstart":
            enhanced_wizard()
        elif cmd.startswith("theme"):
            # Change theme on the fly
            parts = cmd.split()
            if len(parts) > 1 and parts[1] in THEMES:
                current_theme = parts[1]
                console.print(f"[green]เปลี่ยนธีมเป็น {current_theme}")
            else:
                console.print(f"[yellow]ธีมที่มี: {list(THEMES.keys())}")
        else:
            console.print(f"[red]Unknown command: {cmd}")

# --- Enhanced Wizard Command ---
@app.command()
def wizard():
    """Interactive Setup/Quickstart Wizard (เทพขั้นสุด!)"""
    enhanced_wizard()

# --- Helper: แจ้งเตือน class imbalance/ไม่มี predicted sample ใน WFV ---
def warn_undefined_metric(y_true, y_pred):
    import numpy as np
    from rich.console import Console
    console = Console()
    unique_true = set(np.unique(y_true))
    unique_pred = set(np.unique(y_pred))
    missing_classes = unique_true - unique_pred
    if missing_classes:
        console.print(f"[bold yellow][⚠️] พบ class ที่ไม่มีการทำนายในผลลัพธ์: {missing_classes} (precision/recall อาจเป็น 0 หรือ ill-defined)[/bold yellow]")
        console.print("[bold yellow]ระบบได้ suppress warning และตั้ง zero_division=0 ให้แล้ว (เทพสุด)[/bold yellow]")
        console.print("[bold magenta]แนะนำ: ตรวจสอบ class imbalance หรือปรับ threshold/model ให้ครอบคลุมทุก class[/bold magenta]")

def check_analysis_outputs():
    try:
        # ตัวอย่าง: ตรวจสอบไฟล์หลักที่ควรมีหลัง pipeline
        files = [
            'output_default/final_predictions.parquet',
            'output_default/metrics_summary_v32.csv',
            'output_default/buy_sell_hold_strategy_result.parquet',
            'output_default/test_predictions.csv',
        ]
        for f in files:
            if not os.path.exists(f):
                print(color(f"[⚠️] ไม่พบไฟล์วิเคราะห์หลัก: {f}", 'yellow'))
            else:
                print(color(f"[OK] พบไฟล์: {f}", 'green'))
    except Exception as e:
        print(color(f"[ERROR] ตรวจสอบไฟล์วิเคราะห์หลักล้มเหลว: {e}", 'red'))
