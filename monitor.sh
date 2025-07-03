#!/bin/bash
while true; do
    pgrep -f "main.py" || ./_start.sh
    sleep 60
done