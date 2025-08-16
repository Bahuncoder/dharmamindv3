#!/usr/bin/env powershell
# DharmaMind Frontend Startup Script

Write-Host "🌐 DharmaMind Frontend Startup" -ForegroundColor Magenta
Write-Host "===============================" -ForegroundColor Magenta

# Check if backend is running
Write-Host "🔍 Checking if backend is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/" -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ Backend is running!" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend not running! Please start backend first." -ForegroundColor Red
    Write-Host "Run: .\start_dharmamind_backend.ps1" -ForegroundColor Yellow
    exit 1
}

# Function to start a frontend app
function Start-Frontend {
    param(
        [string]$AppName,
        [string]$Directory,
        [int]$Port
    )
    
    Write-Host "🚀 Starting $AppName on port $Port..." -ForegroundColor Green
    
    if (Test-Path $Directory) {
        Set-Location $Directory
        
        # Check if node_modules exists
        if (-not (Test-Path "node_modules")) {
            Write-Host "📦 Installing dependencies for $AppName..." -ForegroundColor Yellow
            npm install
        }
        
        # Start the application
        Write-Host "✅ Starting $AppName..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Directory'; npm run dev"
        
        Set-Location ".."
    } else {
        Write-Host "❌ Directory $Directory not found!" -ForegroundColor Red
    }
}

# Start all frontend applications
Start-Frontend "Chat App" "dharmamind-chat" 3001
Start-Sleep -Seconds 2
Start-Frontend "Brand Website" "Brand_Webpage" 3000  
Start-Sleep -Seconds 2
Start-Frontend "Community App" "DhramaMind_Community" 3003

Write-Host ""
Write-Host "🎉 All applications starting!" -ForegroundColor Green
Write-Host "📱 Chat App: http://localhost:3001" -ForegroundColor Cyan
Write-Host "🌐 Brand Website: http://localhost:3000" -ForegroundColor Cyan
Write-Host "👥 Community App: http://localhost:3003" -ForegroundColor Cyan
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
