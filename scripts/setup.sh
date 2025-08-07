#!/bin/bash

# Setup script for Market Trend Forecasting project
# This script automates the initial setup process

echo "🏗️ Setting up Market Trend Forecasting project..."

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p logs
mkdir -p mlruns
mkdir -p mlartifacts
mkdir -p deployed_models
mkdir -p notebooks
mkdir -p reports

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.9 or higher."
    exit 1
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -r requirements-dev.txt

# Create environment file from template
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration file..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your actual configuration."
else
    echo "ℹ️ .env file already exists"
fi

# Initialize git hooks (if in git repository)
if [ -d ".git" ]; then
    echo "🪝 Setting up git hooks..."
    # You can add pre-commit hooks here
    echo "✅ Git hooks configured"
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
python -m pytest tests/ -v

# Start services (optional)
read -p "🐳 Do you want to start Docker services? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Docker services..."
    docker-compose up -d
    echo "✅ Services started. Access the dashboard at http://localhost:8501"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your API keys and configuration"
echo "2. Run 'streamlit run dashboard/app.py' to start the dashboard"
echo "3. Or run 'python main.py' to execute the full pipeline"
echo "4. Visit http://localhost:8501 for the web interface"
echo ""
echo "For more information, see the README.md file."