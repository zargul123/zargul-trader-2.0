#!/bin/bash

# =============================================
# ZARGUL TRADER 2.0 - SAFE START SCRIPT
# =============================================

# 1. Set CPU optimization flags
export OMP_NUM_THREADS=6
export TF_ENABLE_ONEDNN_OPTS=1

# 2. Kill any existing processes
pkill -f "scripts/main.py" 2>/dev/null
pkill -f "dashboard.py" 2>/dev/null

# 3. Start TRADING SYSTEM + AUTO TRAINER
echo "🚀 Starting Trading Core..."
PYTHONPATH=$PYTHONPATH:. python scripts/main.py &

echo "🕒 Starting Auto Trainer (daily at 3 AM UTC)..."
PYTHONPATH=$PYTHONPATH:. python auto_trainer.py &

# 4. Crash protection loop
while true; do
    if ! pgrep -f "scripts/main.py" > /dev/null; then
        echo -e "\n💥 TRADING SYSTEM CRASHED! Restarting in 10s..."
        sleep 10
        PYTHONPATH=$PYTHONPATH:. python scripts/main.py &
        echo "✅ Trading system restarted"
    fi
    
    if ! pgrep -f "auto_trainer.py" > /dev/null; then
        echo -e "\n💥 AUTO TRAINER CRASHED! Restarting in 10s..."
        sleep 10
        PYTHONPATH=$PYTHONPATH:. python auto_trainer.py &
        echo "✅ Auto trainer restarted"
    fi
    
    sleep 30  # Check every 30 seconds
done