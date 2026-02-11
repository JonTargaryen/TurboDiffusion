# ======================================================================
# TurboDiffusion Master Installation Script
# ======================================================================
# This script runs all installation phases in sequence
# ======================================================================

param(
    [switch]$SkipSystemCheck,
    [switch]$SkipEnvironmentCreation,
    [switch]$SkipPyTorch,
    [switch]$SkipDependencies,
    [switch]$SkipCompilation,
    [switch]$SkipModels,
    [switch]$SkipTests
)

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  TurboDiffusion Complete Installation" -ForegroundColor Cyan
Write-Host "  Automated Setup for Windows 11 with RTX 5090" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Check system prerequisites" -ForegroundColor Gray
Write-Host "  2. Create conda environment" -ForegroundColor Gray
Write-Host "  3. Install PyTorch with CUDA" -ForegroundColor Gray
Write-Host "  4. Install all dependencies" -ForegroundColor Gray
Write-Host "  5. Compile TurboDiffusion CUDA extensions" -ForegroundColor Gray
Write-Host "  6. Download model checkpoints (~7-16GB)" -ForegroundColor Gray
Write-Host "  7. Test the installation" -ForegroundColor Gray
Write-Host ""
Write-Host "Total time: 1.5-3 hours (mostly downloads and compilation)" -ForegroundColor Yellow
Write-Host "Total disk space: ~30GB" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Continue with installation? (yes/no)"
if ($response -ne "yes" -and $response -ne "y") {
    Write-Host "Installation cancelled" -ForegroundColor Yellow
    exit 0
}

$scriptDir = Join-Path $PSScriptRoot "setup_scripts"
$logDir = Join-Path $PSScriptRoot"logs"

# Create logs directory
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "installation_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $timestampedMessage = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -Path $logFile -Value $timestampedMessage
    Write-Host $Message
}

function Run-Phase {
    param(
        [string]$ScriptName,
        [string]$PhaseName,
        [switch]$Skip
    )
    
    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host $PhaseName -ForegroundColor Cyan
    Write-Host "======================================================" -ForegroundColor Cyan
    
    if ($Skip) {
        Write-Log "Skipping: $PhaseName"
        Write-Host "SKIPPED (--Skip flag provided)" -ForegroundColor Yellow
        return $true
    }
    
    Write-Log "Starting: $PhaseName"
    
    $scriptPath = Join-Path $scriptDir $ScriptName
    if (-not (Test-Path $scriptPath)) {
        Write-Log "ERROR: Script not found: $scriptPath"
        Write-Host "✗ Script not found: $scriptPath" -ForegroundColor Red
        return $false
    }
    
    # For phases that need user interaction, run in same window
    if $ScriptName -match "0_system_check|1_create_environment|5_download_models|6_test_installation") {
        & $scriptPath
    } else {
        # For long-running compilation/downloads, run with output
        & $scriptPath 2>&1 | Tee-Object -FilePath $logFile -Append
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: $PhaseName failed with exit code $LASTEXITCODE"
        Write-Host ""
        Write-Host "✗ $PhaseName failed!" -ForegroundColor Red
        Write-Host "Check log file: $logFile" -ForegroundColor Yellow
        return $false
    }
    
    Write-Log "Completed: $PhaseName"
    Write-Host "✓ $PhaseName completed" -ForegroundColor Green
    return $true
}

# Start installation
Write-Log "===== TurboDiffusion Installation Started ====="
Write-Log "Log file: $logFile"

$success = $true

# Phase 0: System Check
if (-not $SkipSystemCheck) {
    $success = Run-Phase -ScriptName "0_system_check.ps1" -PhaseName "Phase 0: System Prerequisites Check"
    if (-not $success) {
        Write-Host ""
        Write-Host "Please install missing prerequisites before continuing" -ForegroundColor Yellow
        Write-Host "Press any key to exit..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
}

# Phase 1: Create Environment
if (-not $SkipEnvironmentCreation) {
    $success = Run-Phase -ScriptName "1_create_environment.ps1" -PhaseName "Phase 1: Create Conda Environment"
    if (-not $success) {
        Write-Host "Installation failed at Phase 1" -ForegroundColor Red
        exit 1
    }
    
    # Activate environment for subsequent scripts
    Write-Host ""
    Write-Host "Activating conda environment..." -ForegroundColor Yellow
    # Note: In PowerShell, we need to use conda activate through conda.exe
    Write-Host "Please manually activate the environment in a new terminal:" -ForegroundColor Yellow
    Write-Host "  conda activate turbodiffusion" -ForegroundColor Cyan
    Write-Host "Then run this script again with -SkipEnvironmentCreation flag" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or continue with individual phase scripts" -ForegroundColor Yellow
    exit 0
}

# Phase 2: Install PyTorch
$success = Run-Phase -ScriptName "2_install_pytorch.ps1" -PhaseName "Phase 2: Install PyTorch" -Skip:$SkipPyTorch
if (-not $success) {
    Write-Host "Installation failed at Phase 2" -ForegroundColor Red
    exit 1
}

# Phase 3: Install Dependencies
$success = Run-Phase -ScriptName "3_install_dependencies.ps1" -PhaseName "Phase 3: Install Dependencies" -Skip:$SkipDependencies
if (-not $success) {
    Write-Host "Installation failed at Phase 3" -ForegroundColor Red
    exit 1
}

# Phase 4: Compile TurboDiffusion
$success = Run-Phase -ScriptName "4_compile_turbodiffusion.ps1" -PhaseName "Phase 4: Compile TurboDiffusion" -Skip:$SkipCompilation
if (-not $success) {
    Write-Host "Installation failed at Phase 4" -ForegroundColor Red
    exit 1
}

# Phase 5: Download Models
$success = Run-Phase -ScriptName "5_download_models.ps1" -PhaseName "Phase 5: Download Models" -Skip:$SkipModels
if (-not $success) {
    Write-Host "Installation failed at Phase 5" -ForegroundColor Red
    exit 1
}

# Phase 6: Test Installation
$success = Run-Phase -ScriptName "6_test_installation.ps1" -PhaseName "Phase 6: Test Installation" -Skip:$SkipTests
if (-not $success) {
    Write-Host "Installation failed at Phase 6" -ForegroundColor Red
    exit 1
}

# Complete
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "✓ INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "TurboDiffusion is ready to use!" -ForegroundColor Green
Write-Host ""
Write-Host "Log file saved to: $logFile" -ForegroundColor Gray
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Yellow
Write-Host "  1. Activate environment: conda activate turbodiffusion" -ForegroundColor White
Write-Host '  2. Run inference: python turbodiffusion\inference\wan2.1_t2v_infer.py --help' -ForegroundColor White
Write-Host ""

Write-Log "===== TurboDiffusion Installation Completed Successfully ====="

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
