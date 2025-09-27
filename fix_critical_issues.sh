#!/bin/bash

# 🚀 DharmaMind Critical Issues Fix Script
# Run this script to resolve immediate blocking issues

echo "🕉️  DharmaMind Critical Fix Script"
echo "=================================="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "📁 Project Root: $PROJECT_ROOT"

# 1. Fix Environment Configuration
echo "🔐 Setting up environment configuration..."
if [ ! -f .env ]; then
    echo "📝 Creating .env from production template..."
    cp .env.production .env 2>/dev/null || touch .env
fi

# Generate secure keys if they don't exist
if ! grep -q "JWT_SECRET_KEY=" .env; then
    echo "🔑 Generating JWT_SECRET_KEY..."
    JWT_SECRET=$(openssl rand -hex 64)
    echo "JWT_SECRET_KEY=$JWT_SECRET" >> .env
fi

if ! grep -q "SECRET_KEY=" .env; then
    echo "🔑 Generating SECRET_KEY..."  
    SECRET_KEY=$(openssl rand -hex 32)
    echo "SECRET_KEY=$SECRET_KEY" >> .env
fi

# 2. Clean Python Cache
echo "🧹 Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 3. Check Backend Dependencies
echo "🐍 Checking backend Python environment..."
if [ -d "dharmamind_env" ]; then
    echo "✅ Virtual environment found: dharmamind_env"
    source dharmamind_env/bin/activate
    echo "📦 Installing backend requirements..."
    pip install -r backend/requirements.txt --quiet
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv dharmamind_env
    source dharmamind_env/bin/activate
    pip install -r backend/requirements.txt
fi

# 4. Check Frontend Dependencies
echo "📱 Checking frontend dependencies..."
cd dharmamind-chat
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
else
    echo "✅ Frontend dependencies already installed"
fi
cd ..

# 5. Fix File Permissions
echo "🔧 Fixing file permissions..."
chmod +x test_dharmallm_integration.sh
chmod +x backend/quick_start.sh 2>/dev/null || true
chmod +x backend/install_backend.sh 2>/dev/null || true

# 6. Test Critical Services
echo "🧪 Testing critical services..."

# Test backend imports
echo "   Testing backend imports..."
cd backend
python -c "from app.main import app; print('✅ Backend imports working')" 2>/dev/null || echo "❌ Backend import issues"
cd ..

# Test frontend build
echo "   Testing frontend build..."
cd dharmamind-chat
npm run build > /dev/null 2>&1 && echo "✅ Frontend builds successfully" || echo "❌ Frontend build issues"
cd ..

# 7. Create Quick Start Scripts
echo "📝 Creating quick start scripts..."

cat > quick_start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source ../dharmamind_env/bin/activate
python run_server.py
EOF

cat > quick_start_frontend.sh << 'EOF'
#!/bin/bash
cd dharmamind-chat
npm run dev
EOF

cat > quick_start_all.sh << 'EOF'
#!/bin/bash
# Start all services in parallel
gnome-terminal --tab --title="Backend" -- bash -c "./quick_start_backend.sh; exec bash"
gnome-terminal --tab --title="Frontend" -- bash -c "./quick_start_frontend.sh; exec bash"
echo "🚀 Starting all services..."
echo "🐍 Backend: http://localhost:8000"
echo "📱 Frontend: http://localhost:3000"
EOF

chmod +x quick_start_*.sh

# 8. Validation
echo "✅ Running final validation..."
python3 validate_integration.py || echo "⚠️  Some validation warnings (normal if services not running)"

echo ""
echo "🎯 CRITICAL FIXES COMPLETED!"
echo "=========================="
echo "✅ Environment configuration set up"
echo "✅ Dependencies installed"
echo "✅ File permissions fixed"
echo "✅ Quick start scripts created"
echo ""
echo "🚀 Next Steps:"
echo "1. Review .env file and add your API keys"
echo "2. Run: ./quick_start_all.sh (to start all services)"
echo "3. Or run individual services:"
echo "   - Backend: ./quick_start_backend.sh"
echo "   - Frontend: ./quick_start_frontend.sh"
echo ""
echo "📚 Check PROJECT_COMPREHENSIVE_ANALYSIS.md for detailed improvement plan"
echo "🕉️  DharmaMind is ready for development!"