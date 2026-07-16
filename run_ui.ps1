# Start the Streamlit UI (port 8501)
Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run ui\streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
