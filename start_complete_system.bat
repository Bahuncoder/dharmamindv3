@echo off
setlocal

rem DharmaMind Complete System Startup Script
rem This script starts the backend and opens the demo frontend

echo 🚀 Starting DharmaMind Complete System...
echo.

rem Check if we're in the right directory
if not exist "README.md" (
    echo ❌ Please run this script from the DharmaMind root directory
    pause
    exit /b 1
)

echo 📂 Current directory: %CD%
echo.

rem Check if backend is already running
netstat -an | find "5001" >nul
if errorlevel 1 (
    echo 🔧 Starting Enhanced Enterprise Authentication Backend...
    cd backend\app
    start "DharmaMind Backend" python enhanced_enterprise_auth.py
    cd ..\..
    echo ✅ Backend starting...
    timeout /t 5 >nul
) else (
    echo ✅ Backend already running on port 5001
)

rem Check if frontend is already running
netstat -an | find "3000" >nul
if errorlevel 1 (
    echo 🔧 Starting Next.js Frontend Development Server...
    cd frontend
    start "DharmaMind Frontend" npm run dev
    cd ..
    echo ✅ Frontend starting...
    timeout /t 8 >nul
) else (
    echo ✅ Frontend already running on port 3000
)

echo.
echo 🌟 DharmaMind System Status:
echo    📍 Backend API: http://localhost:5001
echo    📖 API Documentation: http://localhost:5001/docs
echo    🚀 Next.js Frontend: http://localhost:3000
echo    🧪 Demo Frontend: file:///%CD%/frontend/demo.html
echo.

rem Open demo frontend in default browser
echo 🌐 Opening applications in browser...
start "" "http://localhost:3000"
timeout /t 2 >nul
start "" "file:///%CD%/frontend/demo.html"
timeout /t 2 >nul

rem Open API documentation
echo 📖 Opening API documentation...
start "" "http://localhost:5001/docs"

echo.
echo 🎯 What you can do now:
echo    1. Test the API at http://localhost:5001/docs
echo    2. Use the Next.js app at http://localhost:3000
echo    3. Try the demo interface for quick testing
echo    2. Use the demo interface to test registration/login
echo    3. Try the 'Demo Registration' and 'Demo Login' buttons
echo    4. Check the API health status
echo.
echo 💡 Features available:
echo    ✅ User Registration with validation
echo    ✅ User Login with JWT tokens
echo    ✅ Password security requirements
echo    ✅ Profile management endpoints
echo    ✅ Enterprise security features
echo.
echo 🔧 Backend is running in a separate window
echo 🔧 Close that window to stop the backend
echo.

pause
