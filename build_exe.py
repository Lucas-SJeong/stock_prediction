import os
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYINSTALLER_EXE = r"C:\Users\kccistc\.conda\envs\my_stock\Scripts\pyinstaller.exe"

if not os.path.exists(PYINSTALLER_EXE):
    PYINSTALLER_EXE = "pyinstaller"

cmd = [
    PYINSTALLER_EXE,
    "--noconfirm",
    "--onedir",
    "--name=stock_dashboard_app",
    "--collect-all=dash",
    "--collect-all=plotly",
    "--collect-all=yfinance",
    "--collect-all=feedparser",
    "--collect-all=keras",
    "--add-data=project;project",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/Library/bin/libssl-3-x64.dll;.",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/Library/bin/libcrypto-3-x64.dll;.",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/msvcp140.dll;.",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/msvcp140_1.dll;.",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/msvcp140_2.dll;.",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/vcruntime140.dll;.",
    "--add-binary=C:/Users/kccistc/.conda/envs/my_stock/vcruntime140_1.dll;.",
    "stock_chart.py"
]

print("  Building standalone Windows executable using PyInstaller...")
try:
    subprocess.check_call(cmd, cwd=BASE_DIR)
    print("  [OK] Build completed successfully!")
    print(f"  [OK] Executable directory: {os.path.join(BASE_DIR, 'dist', 'stock_dashboard_app')}")
    print(f"  [OK] Main executable: {os.path.join(BASE_DIR, 'dist', 'stock_dashboard_app', 'stock_dashboard_app.exe')}")
except Exception as e:
    print(f"  [ERROR] Build failed: {e}")
    sys.exit(1)

