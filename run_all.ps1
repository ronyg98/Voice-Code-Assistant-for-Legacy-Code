# Launch backend + sign-language sidecar + Streamlit UI, each in its own window.
Start-Process powershell -ArgumentList "-NoExit", "-File", "$PSScriptRoot\run_backend.ps1"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-File", "$PSScriptRoot\run_signlang.ps1"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-File", "$PSScriptRoot\run_ui.ps1"
Write-Host "Backend  -> http://127.0.0.1:8000/docs"
Write-Host "Sign svc -> http://127.0.0.1:5055"
Write-Host "UI       -> http://127.0.0.1:8501"
