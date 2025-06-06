#!/bin/bash

# OmniBio Biomarker Analysis - Setup Script
# This script sets up the development environment for the biomarker analysis project

set -e  # Exit on any error

echo "🚀 Setting up OmniBio Biomarker Analysis Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check if metabo environment exists
if ! conda env list | grep -q "metabo"; then
    print_error "Conda environment 'metabo' not found!"
    print_status "Please create the metabo environment first with:"
    print_status "conda create -n metabo python=3.11"
    exit 1
fi

print_success "Found conda environment 'metabo'"

# Activate metabo environment and install Python dependencies
print_status "Activating metabo environment and installing Python dependencies..."

# Export conda activate function so it works in this script
eval "$(conda shell.bash hook)"
conda activate metabo

# Verify Python version
PYTHON_VERSION=$(python --version)
print_status "Using $PYTHON_VERSION"

# Install type checker and development tools
print_status "Installing Python development tools..."
pip install --upgrade pip

# Install pyright for type checking
if ! pip list | grep -q "pyright"; then
    print_status "Installing pyright type checker..."
    pip install pyright
else
    print_success "pyright already installed"
fi

# Install Python linting and formatting tools
print_status "Installing Python linting and formatting tools..."
pip install black flake8 isort mypy --upgrade

# Verify core dependencies are installed
print_status "Verifying core Python dependencies..."
REQUIRED_PACKAGES=("fastapi" "uvicorn" "pandas" "numpy" "scikit-learn" "plotly" "matplotlib" "seaborn")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if pip list | grep -q "$package"; then
        VERSION=$(pip list | grep "$package" | awk '{print $2}')
        print_success "$package $VERSION installed"
    else
        print_warning "$package not found, installing..."
        pip install "$package"
    fi
done

# Install additional scientific computing packages if missing
print_status "Installing additional scientific packages..."
pip install mwtab pymzml pyopenms --upgrade --quiet

print_success "Python environment setup complete"

# Setup Frontend (Next.js)
print_status "Setting up frontend dependencies..."

if [ ! -d "omnibio-frontend" ]; then
    print_error "Frontend directory 'omnibio-frontend' not found!"
    exit 1
fi

cd omnibio-frontend

# Check Node.js version
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed"
    exit 1
fi

NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
print_status "Using Node.js $NODE_VERSION with npm $NPM_VERSION"

# Install frontend dependencies
print_status "Installing frontend dependencies..."
npm install

# Install additional development dependencies for linting/formatting
print_status "Installing frontend development tools..."
npm install --save-dev eslint prettier @typescript-eslint/eslint-plugin @typescript-eslint/parser

# Install missing Plotly dependencies if needed
if ! npm list react-plotly.js &> /dev/null; then
    print_status "Installing react-plotly.js..."
    npm install react-plotly.js plotly.js
fi

print_success "Frontend dependencies installed"

# Return to project root
cd ..

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# OmniBio Environment Configuration
ENVIRONMENT=development
API_KEY=omnibio-dev-key-12345
DATABASE_URL=postgresql://localhost:5432/omnibio_dev
UPLOAD_DIR=./biomarker/uploads
RESULTS_DIR=./biomarker/results
EOF
    print_success "Created .env file with default settings"
else
    print_success ".env file already exists"
fi

# Create requirements.txt from current environment
print_status "Generating requirements.txt from current environment..."
conda activate metabo
pip freeze > requirements.txt
print_success "Generated requirements.txt with $(wc -l < requirements.txt) packages"

# Verify setup by testing imports
print_status "Verifying Python setup..."
python -c "
import fastapi
import uvicorn
import pandas as pd
import numpy as np
import sklearn
import plotly
import matplotlib
import seaborn as sns
print('✓ All core Python packages imported successfully')
"

# Test frontend setup
print_status "Verifying frontend setup..."
cd omnibio-frontend
if npm run build --dry-run &> /dev/null; then
    print_success "Frontend build configuration is valid"
else
    print_warning "Frontend build may have issues - check package.json"
fi
cd ..

# Create port cleanup script
print_status "Creating port cleanup script..."
cat > kill_ports.sh << 'EOF'
#!/bin/bash
# Kill processes on development ports
echo "🔧 Clearing development ports..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "✓ Cleared port 8000" || echo "Port 8000 already free"
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "✓ Cleared port 3000" || echo "Port 3000 already free"
lsof -ti:3001 | xargs kill -9 2>/dev/null && echo "✓ Cleared port 3001" || echo "Port 3001 already free"
echo "✅ Port cleanup complete"
EOF

chmod +x kill_ports.sh
print_success "Created kill_ports.sh script"

# Final setup summary
print_success "🎉 Setup Complete!"
echo ""
echo "📋 Setup Summary:"
echo "  ✅ Python environment: metabo (Python 3.11.11)"
echo "  ✅ Backend dependencies: FastAPI, ML libraries, data processing"
echo "  ✅ Frontend dependencies: Next.js, React, Plotly"
echo "  ✅ Development tools: pyright, black, flake8, eslint"
echo "  ✅ Configuration files: .env, requirements.txt"
echo "  ✅ Port cleanup script: kill_ports.sh"
echo ""
echo "🚀 Quick Start Commands:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   conda activate metabo"
echo "   cd biomarker/api"
echo "   python -m uvicorn main:app --reload --port 8000"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd omnibio-frontend"
echo "   npm run dev"
echo ""
echo "3. Clear ports if needed:"
echo "   ./kill_ports.sh"
echo ""
echo "4. Test setup:"
echo "   curl http://localhost:8000/health"
echo "   open http://localhost:3000"
echo ""
print_success "Environment ready for biomarker analysis development! 🧬" 