@echo off
REM Batch script to run the complete project
REM Friend Recommendation GNN Project

echo ========================================
echo Friend Recommendation GNN Project
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

echo Step 1: Installing dependencies...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install
    echo Continuing anyway...
)
echo.

echo Step 2: Creating synthetic dataset...
echo.
python scripts/download_and_prepare.py --dataset synthetic --preprocess
if errorlevel 1 (
    echo ERROR: Failed to create dataset
    pause
    exit /b 1
)
echo.

echo Step 3: Training GraphSAGE model...
echo.
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml
if errorlevel 1 (
    echo ERROR: Training failed
    pause
    exit /b 1
)
echo.

echo Step 4: Evaluating model...
echo.
python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --dataset synthetic --config configs/graphsage_config.yaml
if errorlevel 1 (
    echo WARNING: Evaluation failed, but continuing...
)
echo.

echo ========================================
echo Project setup complete!
echo ========================================
echo.
echo To run the demo app, execute:
echo   python -m streamlit run demo/streamlit_app.py
echo.
echo Or press any key to start the demo app now...
pause >nul

echo.
echo Starting Streamlit demo app...
echo.
python -m streamlit run demo/streamlit_app.py

pause

