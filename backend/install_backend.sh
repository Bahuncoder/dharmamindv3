#!/bin/bash

# 🕉️ DharmaMind Backend Installation Script
# ========================================

set -e  # Exit on any error

echo "🕉️ DharmaMind Backend Installation Starting..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if we're in the backend directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Make sure you're in the backend directory."
    exit 1
fi

print_info "Current directory: $(pwd)"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install wheel for better package compilation
print_info "Installing wheel..."
pip install wheel

# Install core requirements
print_info "Installing core requirements..."
pip install -r requirements.txt

# Install development requirements if they exist
if [ -f "requirements-dev.txt" ]; then
    print_info "Installing development requirements..."
    pip install -r requirements-dev.txt
fi

# Install enterprise requirements if they exist
if [ -f "requirements_enterprise.txt" ]; then
    print_info "Installing enterprise requirements..."
    pip install -r requirements_enterprise.txt
fi

# Install additional packages that might be missing
print_info "Installing additional missing packages..."

# Check and install missing packages one by one
packages_to_check=(
    "fakeredis"
    "asyncio"
    "datetime"
    "typing"
    "dataclasses"
    "enum34"
    "logging"
    "pathlib"
    "uuid"
    "json"
    "re"
    "random"
    "time"
    "os"
    "sys"
    "warnings"
    "traceback"
)

additional_packages=(
    "fakeredis"
    "python-dateutil" 
    "pytz"
    "pydantic[email]"
    "bcrypt"
    "python-multipart"
    "aiofiles"
    "websockets"
)

for package in "${additional_packages[@]}"; do
    print_info "Installing $package..."
    pip install "$package" || print_warning "Failed to install $package (might not be needed)"
done

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p data
mkdir -p logs
mkdir -p temp
mkdir -p uploads

# Set permissions
print_info "Setting directory permissions..."
chmod 755 data logs temp uploads

# Check if Redis is available
print_info "Checking Redis availability..."
if command -v redis-server &> /dev/null; then
    print_status "Redis server is available"
    # Try to start Redis if not running
    if ! pgrep -x "redis-server" > /dev/null; then
        print_info "Starting Redis server..."
        redis-server --daemonize yes || print_warning "Could not start Redis server"
    else
        print_status "Redis server is already running"
    fi
else
    print_warning "Redis server not found. FakeRedis will be used for development."
fi

# Check if SQLite directory exists and create if needed
print_info "Setting up database directory..."
mkdir -p data
touch data/dharma_knowledge.db

# Run basic import test
print_info "Testing basic imports..."
python3 -c "
import sys
sys.path.append('.')
try:
    from app.main import app
    print('✅ Main app imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    
try:
    from app.engines.emotional import AdvancedEmotionalEngine
    print('✅ Emotional engine imports successfully')
except Exception as e:
    print(f'❌ Emotional engine import error: {e}')
    
try:
    from app.engines.rishi import AuthenticRishiEngine
    print('✅ Rishi engine imports successfully')
except Exception as e:
    print(f'❌ Rishi engine import error: {e}')
"

print_status "Installation completed!"
print_info "=================================================="
print_info "🕉️ DharmaMind Backend Ready!"
print_info "=================================================="
print_info "To start the server:"
print_info "  source venv/bin/activate"
print_info "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
print_info ""
print_info "To run tests:"
print_info "  python3 test_backend_integration.py"
print_info ""
print_status "May all beings benefit from this technology! 🙏"
