#!/usr/bin/env bash
set -e

# Setup script for KevRouter virtual environment
VENV_NAME="kevrouter_venv"

echo "=== KevRouter Environment Setup ==="

# 1. Determine available Python interpreter
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither 'python3' nor 'python' was found in PATH."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version) at $(which $PYTHON_CMD)"

# 2. Create virtual environment if it doesn't already exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME'..."
    $PYTHON_CMD -m venv "$VENV_NAME"
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

# 3. Activate virtual environment (support Linux/macOS and Windows Git Bash paths)
if [ -f "$VENV_NAME/bin/activate" ]; then
    echo "Activating virtual environment (POSIX)..."
    source "$VENV_NAME/bin/activate"
elif [ -f "$VENV_NAME/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows/Git Bash)..."
    source "$VENV_NAME/Scripts/activate"
else
    echo "Error: Activation script not found in $VENV_NAME."
    exit 1
fi

# 4. Upgrade core packaging tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 5. Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping dependency installation."
fi

# 6. Register ipykernel for Jupyter notebooks (EDA)
echo "Registering ipykernel for Jupyter..."
python -m ipykernel install --user --name "$VENV_NAME" --display-name "Python ($VENV_NAME)" || echo "Note: ipykernel registration skipped or optional."

echo "=== Setup complete! ==="
echo "To activate this environment later, run:"
echo "  source $VENV_NAME/bin/activate    # Linux / macOS / Delta Cluster"
echo "  source $VENV_NAME/Scripts/activate # Windows Git Bash"
echo "  .\\$VENV_NAME\\Scripts\\Activate.ps1 # Windows PowerShell"
