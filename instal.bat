@echo off
echo Instalando dependencias...
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python
echo.
echo Pronto! Agora rode: python run.py
pause