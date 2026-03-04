@echo off
setlocal enabledelayedexpansion

:: 1. Check if hf-cli is installed
where hf >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [Installer] hf-cli not found. Installing now...
    
    :: Install via PowerShell (the official hf-cli install method for Windows)
    powershell -Command "iwr https://hf.co/cli/install.ps1 -useb | iex"
    
    :: Add the typical install path to the current session
    set "PATH=%PATH%;%USERPROFILE%\.local\bin;%USERPROFILE%\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts"
    
    :: Re-check if hf exists
    where hf >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo [!] Hugging Face CLI not found after setup. Check manual installation: https://hf.co/cli
        pause
        exit /b 1
    )
) else (
    echo [Check] Hugging Face CLI is already installed.
)

:: 2. List of models (translated from MODELS array)
set "MODELS=HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

echo Checking models... this may take a second.

:: 3. Download loop
for %%m in (%MODELS%) do (
    echo ---------------------------------------------------
    echo Checking: %%m
    hf download %%m
)

echo ---------------------------------------------------
echo ✅ All models processed! The Beast is ready.
hf scan-cache
pause
