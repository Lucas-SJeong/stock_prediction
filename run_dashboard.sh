#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "🚀 Launching NASDAQ AI Stock Prediction Dashboard..."
python3 "$DIR/stock_chart.py"
