@echo off
cd /d "%~dp0"

echo ===============================
echo Iniciando Catalogos Hansair
echo ===============================

REM 1) Abrir Docker Desktop
echo Abriendo Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

REM 2) Esperar a que Docker esté listo
echo Esperando a que Docker inicie...
:wait_docker
docker info >nul 2>&1
if errorlevel 1 (
    timeout /t 5 >nul
    goto wait_docker
)

echo Docker listo.

REM 3) Levantar contenedores
docker compose up -d --build
if errorlevel 1 (
    echo ERROR al levantar contenedores
    pause
    exit /b 1
)

REM 4) Esperar API
timeout /t 5 >nul

REM 5) Abrir navegador
start "" "http://localhost:8000/"
start "" "http://localhost:8000/docs"

echo Listo 🚀
