# ======================================================================
# TurboDiffusion Installation - Phase 3: Install Dependencies
# ======================================================================
# This script installs all required Python dependencies
# Prerequisites: PyTorch must be installed
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Phase 3: Install Dependencies" -ForegroundColor Cyan
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

# Verify PyTorch is installed
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
try {
    $torchCheck = python -c "import torch; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PyTorch $torchCheck installed" -ForegroundColor Green
    } else {
        throw "PyTorch not found"
    }
} catch {
    Write-Host "✗ PyTorch not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run Phase 2 first:" -ForegroundColor Yellow
    Write-Host "  .\setup_scripts\2_install_pytorch.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
Write-Host ""

# Install dependencies in batches
$totalSteps = 6
$currentStep = 1

# Step 1: Core build dependencies
Write-Host "[$currentStep/$totalSteps] Installing core build dependencies..." -ForegroundColor Yellow
pip install ninja packaging wheel setuptools
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install build dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build dependencies installed" -ForegroundColor Green
Write-Host ""
$currentStep++

# Step 2: Triton
Write-Host "[$currentStep/$totalSteps] Installing Triton..." -ForegroundColor Yellow
Write-Host "Note: This requires CUDA support" -ForegroundColor Gray
pip install "triton>=3.3.0"
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Triton installation failed (may work without optimization)" -ForegroundColor Yellow
} else {
    Write-Host "✓ Triton installed" -ForegroundColor Green
}
Write-Host ""
$currentStep++

# Step 3: Flash Attention (critical but may fail)
Write-Host "[$currentStep/$totalSteps] Installing Flash Attention..." -ForegroundColor Yellow
Write-Host "This may take 5-15 minutes to compile..." -ForegroundColor Gray
Write-Host "Note: Requires CUDA Toolkit and Visual Studio Build Tools" -ForegroundColor Gray
Write-Host ""

pip install flash-attn --no-build-isolation

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠ Flash Attention installation failed" -ForegroundColor Yellow
    Write-Host "This is common on Windows. TurboDiffusion may still work with reduced performance." -ForegroundColor Gray
    Write-Host ""
    $response = Read-Host "Continue anyway? (yes/no)"
    if ($response -ne "yes" -and $response -ne "y") {
        Write-Host "Installation aborted" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✓ Flash Attention installed" -ForegroundColor Green
}
Write-Host ""
$currentStep++

# Step 4: Core Python dependencies
Write-Host "[$currentStep/$totalSteps] Installing core Python dependencies..." -ForegroundColor Yellow
pip install einops numpy pillow loguru imageio[ffmpeg] pandas PyYAML omegaconf attrs fvcore

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install core dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Core dependencies installed" -ForegroundColor Green
Write-Host ""
$currentStep++

# Step 5: NLP and model dependencies
Write-Host "[$currentStep/$totalSteps] Installing NLP and model dependencies..." -ForegroundColor Yellow
pip install ftfy regex transformers nvidia-ml-py prompt-toolkit rich

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install NLP dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ NLP dependencies installed" -ForegroundColor Green
Write-Host ""
$currentStep++

# Step 6: SpargeAttn (optional but recommended)
Write-Host "[$currentStep/$totalSteps] Installing SpargeAttn (optional optimization)..." -ForegroundColor Yellow
Write-Host "This enables SageSLA for faster attention. May take 5-10 minutes..." -ForegroundColor Gray
Write-Host ""

pip install "git+https://github.com/thu-ml/SpargeAttn.git" --no-build-isolation

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ SpargeAttn installation failed" -ForegroundColor Yellow
    Write-Host "TurboDiffusion will use standard SLA instead of SageSLA" -ForegroundColor Gray
} else {
    Write-Host "✓ SpargeAttn installed" -ForegroundColor Green
}
Write-Host ""

# Verify installation
Write-Host "Verifying package installations..." -ForegroundColor Yellow
Write-Host ""

$packages = @("torch", "triton", "einops", "omegaconf", "transformers")
$allInstalled = $true

foreach ($pkg in $packages) {
    try {
        $check = python -c "import $pkg; print('$pkg:', $pkg.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $check" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $pkg not installed" -ForegroundColor Red
            $allInstalled = $false
        }
    } catch {
        Write-Host "  ✗ $pkg not installed" -ForegroundColor Red
        $allInstalled = $false
    }
}

Write-Host ""

if (-not $allInstalled) {
    Write-Host "⚠ Some critical packages failed to install" -ForegroundColor Yellow
    Write-Host "You may need to install them manually before proceeding" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "✓ Phase 3 Complete - Dependencies Installed" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEP:" -ForegroundColor Yellow
Write-Host "Run: .\setup_scripts\4_compile_turbodiffusion.ps1" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
