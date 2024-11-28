@echo off
REM 临时设置 Python 环境变量
set PYTHONPATH=.\python-3.12.7-embed-amd64

REM 执行 Python 脚本
python .\ui.py

pause