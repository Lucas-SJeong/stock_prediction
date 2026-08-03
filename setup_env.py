"""
Stock Prediction Dashboard - Automated Environment Setup & Dependency Manager
-------------------------------------------------------------------------------
This script automatically checks Python compatibility, sets up a virtual environment,
installs/verifies all required packages, configures VSCode settings, and verifies module integrity.
"""

import os
import sys
import subprocess
import shutil
import json

# Force UTF-8 output encoding if possible to prevent Windows CP949 print errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")
SETTINGS_FILE = os.path.join(BASE_DIR, ".vscode", "settings.json")

REQUIRED_MODULES = [
    ("dash", "Dash"),
    ("plotly", "Plotly"),
    ("tensorflow", "TensorFlow"),
    ("yfinance", "Yahoo Finance API"),
    ("feedparser", "Feedparser"),
    ("joblib", "Joblib"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("sklearn", "Scikit-Learn"),
    ("requests", "Requests")
]

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_python_version():
    print_header("1. Checking Python Environment")
    version = sys.version_info
    print(f"  Current Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  [ERROR] Python 3.9 or higher is required.")
        sys.exit(1)
    elif version.major == 3 and version.minor > 12:
        print("  [WARNING] Python 3.13+ detected. Pre-built TensorFlow wheels may require Python 3.10 - 3.12.")
    else:
        print("  [OK] Python version compatibility verified.")

def get_venv_python():
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def setup_virtualenv():
    print_header("2. Setting up Virtual Environment (.venv)")
    venv_python = get_venv_python()
    
    if not os.path.exists(venv_python):
        print(f"  Creating new virtual environment at: {VENV_DIR}")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            print("  [OK] Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print(f"  [OK] Existing virtual environment found at: {VENV_DIR}")
        
    return venv_python

def install_dependencies(py_exec):
    print_header("3. Upgrading Pip & Installing Dependencies")
    print(f"  Using Python Executable: {py_exec}")
    
    try:
        print("  Upgrading pip, setuptools, and wheel...")
        subprocess.check_call([py_exec, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        
        if os.path.exists(REQUIREMENTS_FILE):
            print(f"  Installing packages from {REQUIREMENTS_FILE}...")
            subprocess.check_call([py_exec, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
            print("  [OK] All dependencies installed successfully.")
        else:
            print("  [ERROR] requirements.txt file not found.")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Error during package installation: {e}")
        sys.exit(1)

def configure_vscode(py_exec):
    print_header("4. Configuring VS Code Settings")
    vscode_dir = os.path.dirname(SETTINGS_FILE)
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir, exist_ok=True)
        
    settings = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        except Exception:
            settings = {}
            
    settings["python.defaultInterpreterPath"] = py_exec
    settings["code-runner.executorMap"] = {
        "python": f"{py_exec} -u"
    }
    settings["python-envs.defaultEnvManager"] = "ms-python.python:conda"
    settings["python-envs.defaultPackageManager"] = "ms-python.python:conda"
    
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
        
    print(f"  [OK] Updated {SETTINGS_FILE} with Python path: {py_exec}")

def verify_modules(py_exec):
    print_header("5. Verifying Installed Modules & Dependencies")
    check_code = """
import sys
modules = %s
failed = []
for mod, label in modules:
    try:
        __import__(mod)
        print(f'  [OK] {label} ({mod})')
    except ImportError as e:
        print(f'  [FAIL] {label} ({mod}): {e}')
        failed.append(mod)

if failed:
    sys.exit(1)
""" % (str(REQUIRED_MODULES))

    try:
        subprocess.check_call([py_exec, "-c", check_code])
        print("  [OK] Module verification completed! All required packages are functioning.")
    except subprocess.CalledProcessError:
        print("  [ERROR] Some modules failed to load. Please review output above.")
        sys.exit(1)

def create_runner_scripts(py_exec):
    print_header("6. Creating One-Click Dashboard Launchers")
    
    # Windows Launcher
    bat_path = os.path.join(BASE_DIR, "run_dashboard.bat")
    bat_content = f"""@echo off
title NASDAQ AI Stock Dashboard Launcher
echo Launching NASDAQ AI Stock Dashboard...
"{py_exec}" "%~dp0stock_chart.py"
pause
"""
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)
    print(f"  [OK] Created Windows launcher: {bat_path}")
    
    # Unix Launcher
    sh_path = os.path.join(BASE_DIR, "run_dashboard.sh")
    sh_content = f"""#!/bin/bash
echo "Launching NASDAQ AI Stock Dashboard..."
"{py_exec}" "$(dirname "$0")/stock_chart.py"
"""
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(sh_content)
    try:
        os.chmod(sh_path, 0o755)
    except Exception:
        pass
    print(f"  [OK] Created Unix/macOS launcher: {sh_path}")

def main():
    print_header("NASDAQ AI Stock Dashboard - Environment Setup")
    check_python_version()
    
    py_exec = setup_virtualenv()
    install_dependencies(py_exec)
    configure_vscode(py_exec)
    verify_modules(py_exec)
    create_runner_scripts(py_exec)
    
    print_header("Setup Complete!")
    print("You can now run the dashboard using any of the following commands:")
    print("  1. Windows One-Click:  run_dashboard.bat")
    print(f"  2. Direct Python:      {py_exec} stock_chart.py")
    print("  3. VS Code:            Open stock_chart.py and press Run (Ctrl+F5)")
    print("\nThe Dashboard will start at: http://127.0.0.1:8050/\n")

if __name__ == "__main__":
    main()
