# Start the sign-language sidecar (port 5055) - needs the Python 3.10 venv
# from the SignSpeak project because MediaPipe does not support Python 3.13.
$python = if ($env:SIGNLANG_PYTHON) { $env:SIGNLANG_PYTHON } else { "D:\GenAI Prac\.venv\Scripts\python.exe" }
Set-Location "$PSScriptRoot\services\signlang"
& $python app.py
