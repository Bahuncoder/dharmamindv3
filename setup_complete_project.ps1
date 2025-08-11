#!/usr/bin/env powershell
# 🕉️ DharmaMind Complete Project Setup Script
# This script will install everything needed and start the complete project

Write-Host "🕉️ DHARMAMIND COMPLETE PROJECT SETUP 🕉️" -ForegroundColor Magenta
Write-Host "==========================================" -ForegroundColor Magenta

# Step 1: Install Python from Microsoft Store
Write-Host "`n📦 Step 1: Installing Python..." -ForegroundColor Yellow
Write-Host "Opening Microsoft Store to install Python 3.12..." -ForegroundColor Cyan
Start-Process "ms-windows-store://pdp/?ProductId=9NCVDN91XZQP"
Write-Host "Please install Python 3.12 from the Microsoft Store and press Enter to continue..." -ForegroundColor Green
Read-Host

# Step 2: Verify Python Installation
Write-Host "`n✅ Step 2: Verifying Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python installation failed. Please install Python manually and run this script again." -ForegroundColor Red
    exit 1
}

# Step 3: Create Virtual Environment
Write-Host "`n🏗️ Step 3: Creating fresh virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
}
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Step 4: Install Backend Dependencies
Write-Host "`n📚 Step 4: Installing backend dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install fastapi uvicorn[standard] sqlalchemy asyncpg redis fakeredis python-dotenv pydantic

# Step 5: Install Node.js (if not present)
Write-Host "`n🟢 Step 5: Checking Node.js..." -ForegroundColor Yellow
$nodeVersion = & node --version 2>$null
if (-not $nodeVersion) {
    Write-Host "Installing Node.js..." -ForegroundColor Cyan
    # Download and install Node.js
    $nodeUrl = "https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi"
    $nodeInstaller = "$env:TEMP\nodejs.msi"
    Invoke-WebRequest -Uri $nodeUrl -OutFile $nodeInstaller
    Start-Process msiexec.exe -ArgumentList "/i $nodeInstaller /quiet" -Wait
    $env:PATH += ";C:\Program Files\nodejs"
    Write-Host "Node.js installed successfully!" -ForegroundColor Green
} else {
    Write-Host "Node.js already installed: $nodeVersion" -ForegroundColor Green
}

# Step 6: Install Frontend Dependencies
Write-Host "`n🌐 Step 6: Installing frontend dependencies..." -ForegroundColor Yellow

# Chat App
Write-Host "Installing dharmamind-chat dependencies..." -ForegroundColor Cyan
Set-Location "dharmamind-chat"
if (-not (Test-Path "node_modules")) {
    npm install
}
Set-Location ".."

# Brand Webpage
Write-Host "Installing Brand_Webpage dependencies..." -ForegroundColor Cyan
Set-Location "Brand_Webpage"
if (-not (Test-Path "node_modules")) {
    npm install
}
Set-Location ".."

# Community App
Write-Host "Installing DhramaMind_Community dependencies..." -ForegroundColor Cyan
Set-Location "DhramaMind_Community"
if (-not (Test-Path "node_modules")) {
    npm install
}
Set-Location ".."

# Step 7: Create Startup Script
Write-Host "`n🚀 Step 7: Creating startup script..." -ForegroundColor Yellow

$startupScript = @"
#!/usr/bin/env powershell
# DharmaMind Complete System Launcher

Write-Host "🕉️ Starting DharmaMind Complete System..." -ForegroundColor Magenta

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start Backend
Write-Host "🔱 Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\Activate.ps1; python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

# Wait for backend to start
Start-Sleep -Seconds 5

# Start Chat App
Write-Host "💬 Starting Chat Application..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'dharmamind-chat'; npm run dev"

# Start Brand Website
Write-Host "🌐 Starting Brand Website..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Brand_Webpage'; npm run dev"

# Start Community App
Write-Host "👥 Starting Community Application..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'DhramaMind_Community'; npm run dev"

Write-Host "`n🎉 DharmaMind System Starting!" -ForegroundColor Green
Write-Host "🔗 Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "💬 Chat App: http://localhost:3001" -ForegroundColor Cyan  
Write-Host "🌐 Brand Website: http://localhost:3000" -ForegroundColor Cyan
Write-Host "👥 Community: http://localhost:3003" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan

# Wait and then open browser
Start-Sleep -Seconds 10
Start-Process "http://localhost:8000"
"@

$startupScript | Out-File -FilePath "start_dharmamind_complete.ps1" -Encoding UTF8

Write-Host "`n✨ Setup Complete!" -ForegroundColor Green
Write-Host "To start the complete DharmaMind system, run:" -ForegroundColor Yellow
Write-Host "./start_dharmamind_complete.ps1" -ForegroundColor Cyan

Write-Host "`n🕉️ May this system serve all beings with wisdom and compassion! 🕉️" -ForegroundColor Magenta
