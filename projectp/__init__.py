# ProjectP package init

# Dashboard/Trading integration imports
from .dashboard import main as dashboard_main

def run_dashboard():
    """Run the Streamlit dashboard (for CLI integration)"""
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'projectp/dashboard.py'])
