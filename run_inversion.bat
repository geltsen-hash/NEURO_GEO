@echo off
cd /d "%~dp0"
title MCU Inversion Autonomous Benchmark
python tools\run_autonomous_inversion.py --port COM6
pause
