# ======================================================================
# TurboDiffusion Installation - Phase 2: Install PyTorch
# ======================================================================
# This script installs PyTorch 2.7.0+ with CUDA 12.6 support
# Prerequisites: Virtual environment must be activated
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Phase 2: Install PyTorch" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if in virtual environment
Write-Host "Checking virtual environment..." -ForegroundColor Yellow
$venvActive = $env:VIRTUAL_ENV
if (-not $venvActive) {
    Write-Host "  [X] Not in a virtual environment!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please activate the environment first:" -ForegroundColor Yellow
    Write-Host "  .\turbodiffusion_env\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
Write-Host "  [OK] Virtual environment active: $venvActive" -ForegroundColor Green
Write-Host ""

# Check CUDA availability
Write-Host "Checking CUDA Toolkit..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $cudaVer = $nvccVersion | Select-String -Pattern "release (\d+\.\d+)"
        $cudaVersion = $cudaVer.Matches[0].Groups[1].Value
        Write-Host "  [OK] CUDA Toolkit $cudaVersion detected" -ForegroundColor Green
    } else {
        Write-Host "  [!] CUDA Toolkit not detected (nvcc not in PATH)" -ForegroundColor Yellow
        Write-Host "      Proceeding anyway, but compilation may fail later" -ForegroundColor Gray
    }
} catch {
    Write-Host "  [!] CUDA Toolkit not detected" -ForegroundColor Yellow
}
Write-Host ""

# Check if PyTorch is already installed
Write-Host "Checking for existing PyTorch installation..." -ForegroundColor Yellow
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [!] PyTorch already installed: $torchVersion" -ForegroundColor Yellow
        Write-Host ""
        $response = Read-Host "Reinstall PyTorch? (yes/no)"
        if ($response -ne "yes" -and $response -ne "y") {
            Write-Host "Skipping PyTorch installation" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Next step: .\setup_scripts\3_install_dependencies.ps1" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Press any key to exit..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 0
        }
        Write-Host "Uninstalling current PyTorch..." -ForegroundColor Yellow
        pip uninstall torch torchvision torchaudio -y
        Write-Host ""
    }
} catch {
    Write-Host "  [OK] No existing PyTorch installation" -ForegroundColor Green
}
Write-Host ""

# Install PyTorch with CUDA 12.6 support
Write-Host "Installing PyTorch 2.7.0+ with CUDA 12.6 support..." -ForegroundColor Yellow
Write-Host "This will take 5-15 minutes depending on your connection..." -ForegroundColor Gray
Write-Host ""

# PyTorch installation command for CUDA 12.6
Write-Host "Running: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126" -ForegroundColor Gray
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [X] PyTorch installation failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check your internet connection" -ForegroundColor Gray
    Write-Host "2. Try running the command manually:" -ForegroundColor Gray
    Write-Host "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126" -ForegroundColor White
    Write-Host "3. If using CUDA 12.1-12.5, use cu121 instead:" -ForegroundColor Gray
    Write-Host "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host ""
Write-Host "  [OK] PyTorch installed successfully!" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
$verification = python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>&1

Write-Host $verification -ForegroundColor Gray

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  [OK] PyTorch with CUDA support verified!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  [!] Warning: CUDA support not available!" -ForegroundColor Yellow
    Write-Host "      This may cause issues later. Ensure CUDA Toolkit is installed." -ForegroundColor Gray
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "Run Phase 3 script to install dependencies:" -ForegroundColor White
Write-Host "  .\setup_scripts\3_install_dependencies.ps1" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
