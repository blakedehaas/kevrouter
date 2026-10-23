# Setup script for KevRouter virtual environment (PowerShell)
$ErrorActionPreference = "Stop"

$VenvName = "kevrouter_venv"

Write-Host "=== KevRouter Environment Setup (PowerShell) ===" -ForegroundColor Cyan

# 1. Determine available Python
$PythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $PythonCmd) {
    $PythonCmd = (Get-Command python3 -ErrorAction SilentlyContinue)
}

if (-not $PythonCmd) {
    Write-Error "Neither 'python' nor 'python3' was found in PATH."
    exit 1
}

Write-Host "Using Python: $(& $PythonCmd.Source --version) at $($PythonCmd.Source)" -ForegroundColor Green

# 2. Create virtual environment if it doesn't already exist
if (-not (Test-Path $VenvName)) {
    Write-Host "Creating virtual environment '$VenvName'..." -ForegroundColor Yellow
    & $PythonCmd.Source -m venv $VenvName
} else {
    Write-Host "Virtual environment '$VenvName' already exists." -ForegroundColor Yellow
}

# 3. Activate virtual environment
$ActivateScript = Join-Path $VenvName "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $ActivateScript
} else {
    Write-Error "Activation script '$ActivateScript' not found."
    exit 1
}

# 4. Upgrade pip and core tools
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# 5. Install requirements
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt
}

# 6. Register ipykernel for Jupyter
Write-Host "Registering ipykernel for Jupyter..." -ForegroundColor Cyan
try {
    python -m ipykernel install --user --name $VenvName --display-name "Python ($VenvName)"
} catch {
    Write-Host "Note: ipykernel registration skipped or optional." -ForegroundColor DarkGray
}

Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host "To activate in the future, run: .\$VenvName\Scripts\Activate.ps1" -ForegroundColor Yellow
