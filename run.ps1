# ── Step 1: Install dependencies ─────────────────────────────────────────────
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# ── Step 2: Train the model ───────────────────────────────────────────────────
Write-Host "`nTraining model..." -ForegroundColor Cyan
python model/train_model.py

# ── Step 3: Start Flask API in background ─────────────────────────────────────
Write-Host "`nStarting Flask API on http://127.0.0.1:5000 ..." -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "api/app.py"
Start-Sleep -Seconds 2

# ── Step 4: Launch Streamlit frontend ─────────────────────────────────────────
Write-Host "`nLaunching Streamlit frontend..." -ForegroundColor Cyan
streamlit run frontend/streamlit_app.py
