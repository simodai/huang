@echo off
setlocal
title 启动企业信用管理系统
echo 正在启动企业信用管理系统...
echo.

REM 检查 Python
where python >nul 2>nul
if errorlevel 1 (
	echo 未检测到 Python，请先安装 Python 3.8+ 并加入 PATH。
	pause
	exit /b 1
)

REM 创建虚拟环境（如不存在）
if not exist .venv (
	echo 正在创建虚拟环境 .venv ...
	python -m venv .venv
)

REM 激活虚拟环境
call .\.venv\Scripts\activate

REM 安装依赖
echo 正在安装依赖（requirements.txt）...
pip install -r requirements.txt

echo.
echo 如首次使用，请复制 .env.example 为 .env 并填写 KIMI_API_KEY。
echo 系统将监听: http://localhost:5006
echo.

python app.py

echo.
echo 已退出应用。
pause
endlocal