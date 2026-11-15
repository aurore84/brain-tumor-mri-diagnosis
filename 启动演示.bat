@echo off
cd /d %~dp0
echo Starting Brain Tumor MRI Diagnosis System...
echo Open http://127.0.0.1:7860 in your browser.
echo (Close this window to stop the system.)
"C:\Users\20240\AppData\Roaming\kimi-desktop\daimon-share\daimon\runtime\python\.venv\Scripts\python.exe" app\app.py
pause
