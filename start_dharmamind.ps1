# DharmaMind PowerShell Launcher
Write-Host "🕉️ DharmaMind Complete System Launcher 🕉️" -ForegroundColor Magenta
Write-Host "=" * 50 -ForegroundColor Cyan

# Set location
Set-Location "D:\new complete apps"
Write-Host "📁 Working Directory: $(Get-Location)" -ForegroundColor Green

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv_clean\Scripts\Activate.ps1"

# Check if activation was successful
if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

# Show package list
Write-Host "`n📦 Installed packages:" -ForegroundColor Cyan
pip list --format=columns

Write-Host "`n🚀 Starting DharmaMind Backend..." -ForegroundColor Magenta
Write-Host "Backend URL: http://localhost:8000" -ForegroundColor Green
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor Green
Write-Host ""

# Start the backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
