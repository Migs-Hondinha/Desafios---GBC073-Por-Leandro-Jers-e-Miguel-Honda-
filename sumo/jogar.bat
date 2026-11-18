@echo off
setlocal
rem Sumo x Neuronio launcher: installs uv on first run, then uv fetches Python + deps.
cd /d "%~dp0"

where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo Instalando o uv - so na primeira vez...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo Nao foi possivel instalar o uv. Verifique a internet e tente de novo.
pause
exit /b 1

:have_uv
set "ARGS="
if defined SUMO_SMOKE set "ARGS=--smoke"
echo Preparando o jogo - na primeira vez baixa cerca de 200 MB, aguarde...
uv run --project "%~dp0." python "%~dp0jogo.py" %ARGS%
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" if not defined SUMO_SMOKE pause
exit /b %CODE%
