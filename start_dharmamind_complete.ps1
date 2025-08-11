#!/usr/bin/env powershell
# DharmaMind Complete System Launcher

Write-Host "ðŸ•‰ï¸ Starting DharmaMind Complete System..." -ForegroundColor Magenta

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start Backend
Write-Host "ðŸ”± Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\Activate.ps1; python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

# Wait for backend to start
Start-Sleep -Seconds 5

# Start Chat App
Write-Host "ðŸ’¬ Starting Chat Application..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'dharmamind-chat'; npm run dev"

# Start Brand Website
Write-Host "ðŸŒ Starting Brand Website..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Brand_Webpage'; npm run dev"

# Start Community App
Write-Host "ðŸ‘¥ Starting Community Application..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'DhramaMind_Community'; npm run dev"

Write-Host "
ðŸŽ‰ DharmaMind System Starting!" -ForegroundColor Green
Write-Host "ðŸ”— Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "ðŸ’¬ Chat App: http://localhost:3001" -ForegroundColor Cyan  
Write-Host "ðŸŒ Brand Website: http://localhost:3000" -ForegroundColor Cyan
Write-Host "ðŸ‘¥ Community: http://localhost:3003" -ForegroundColor Cyan
Write-Host "ðŸ“š API Docs: http://localhost:8000/docs" -ForegroundColor Cyan

# Wait and then open browser
Start-Sleep -Seconds 10
Start-Process "http://localhost:8000"
