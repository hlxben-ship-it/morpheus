# Starts the local ComfyUI server that morpheus.py talks to (http://127.0.0.1:8188).
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\ComfyUI\main.py" --listen 127.0.0.1 --port 8188 --disable-auto-launch
