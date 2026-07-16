# Hands-free live voice assistant (wake word + VAD + streaming TTS + barge-in)
param(
    [string]$Repo = "sample_legacy",
    [string]$Username = "dev",
    [string]$Password = "dev123"
)
Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m voice.live_assistant --repo $Repo --username $Username --password $Password
