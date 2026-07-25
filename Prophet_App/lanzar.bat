@echo off
cd /d "%~dp0"
call C:\Users\FEVER\anaconda3\Scripts\activate.bat
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
