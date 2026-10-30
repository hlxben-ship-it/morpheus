# Starts the Morpheus web UI on http://127.0.0.1:5196 and opens it in the browser.
# ComfyUI is started automatically with the first image if it is not running yet.
Start-Process "http://127.0.0.1:5196"
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\server.py"
