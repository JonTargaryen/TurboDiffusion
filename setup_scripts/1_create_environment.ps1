# ======================================================================
# TurboDiffusion Installation - Phase 1: Create Virtual Environment
# ======================================================================
# This script creates a clean Python virtual environment for TurboDiffusion
# Prerequisites: Python 3.9+ must be installed
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Phase 1: Create Virtual Environment" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
Write-Host "Checking for Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "  [OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [X] Python not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure Python 3.9+ is installed and in PATH" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
Write-Host ""

# Check if virtual environment already exists
$venvPath = "turbodiffusion_env"
Write-Host "Checking if virtual environment '$venvPath' already exists..." -ForegroundColor Yellow

if (Test-Path $venvPath) {
    Write-Host "  [!] Virtual environment '$venvPath' already exists" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Do you want to remove it and create fresh? (yes/no)"
    if ($response -eq "yes" -or $response -eq "y") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [X] Failed to remove environment" -ForegroundColor Red
            exit 1
        }
        Write-Host "  [OK] Environment removed" -ForegroundColor Green
    } else {
        Write-Host "Using existing environment" -ForegroundColor Yellow
        Write-Host "Note: This may cause conflicts. Consider removing and recreating." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To activate the environment, run:" -ForegroundColor Cyan
        Write-Host "  .\turbodiffusion_env\Scripts\Activate.ps1" -ForegroundColor White
        Write-Host ""
        Write-Host "Press any key to exit..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }
}
Write-Host ""

# Create new virtual environment
Write-Host "Creating virtual environment '$venvPath'..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

python -m venv $venvPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [X] Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  [OK] Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Upgrade pip in the new environment
Write-Host "Upgrading pip in virtual environment..." -ForegroundColor Yellow
& ".\$venvPath\Scripts\python.exe" -m pip install --upgrade pip

Write-Host ""
Write-Host "  [OK] pip upgraded" -ForegroundColor Green
Write-Host ""

# Instructions to activate
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Activate the environment:" -ForegroundColor White
Write-Host "   .\turbodiffusion_env\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Then run Phase 2 script:" -ForegroundColor White
Write-Host "   .\setup_scripts\2_install_pytorch.ps1" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
