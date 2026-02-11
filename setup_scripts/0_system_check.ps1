# TurboDiffusion Installation - Phase 1: System Check
# This script verifies all prerequisites are met before installation

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  TurboDiffusion Installation - System Check" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: NVIDIA GPU
Write-Host "[1/7] Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($gpuInfo) {
        $gpuData = $gpuInfo -split ','
        $gpuName = $gpuData[0].Trim()
        $driverVersion = $gpuData[1].Trim()
        $vramGB = [math]::Round([double]($gpuData[2].Trim() -replace '[^0-9.]'), 0)
        
        Write-Host "  [OK] GPU detected: $gpuName" -ForegroundColor Green
        Write-Host "       Driver: $driverVersion" -ForegroundColor Gray
        Write-Host "       VRAM: $vramGB GB" -ForegroundColor Gray
        
        if ($vramGB -lt 12) {
            Write-Host "  [!] Warning: Less than 12GB VRAM may limit functionality" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [X] No NVIDIA GPU detected or nvidia-smi not found" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "  [X] Error checking GPU: $_" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check 2: Python
Write-Host "[2/7] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+\.\d+)") {
        $version = $matches[1]
        Write-Host "  [OK] Python detected: $version" -ForegroundColor Green
        
        $versionParts = $version -split '\.'
        $major = [int]$versionParts[0]
        $minor = [int]$versionParts[1]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
            Write-Host "  [!] Warning: Python 3.9+ recommended. Current: $version" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [X] Python not found in PATH" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "  [X] Python not found or error occurred" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check 3: PyTorch
Write-Host "[3/7] Checking PyTorch installation..." -ForegroundColor Yellow
try {
    $torchCheck = python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 2>$null
    if ($torchCheck) {
        $torchData = $torchCheck -split "`n"
        $torchVersion = $torchData[0].Trim()
        $cudaAvailable = $torchData[1].Trim()
        
        Write-Host "  [OK] PyTorch detected: $torchVersion" -ForegroundColor Green
        if ($cudaAvailable -eq "True") {
            Write-Host "  [OK] CUDA support: Available" -ForegroundColor Green
        } else {
            Write-Host "  [!] CUDA support: Not available (CPU-only version)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [!] PyTorch not installed" -ForegroundColor Yellow
        Write-Host "      Will be installed in later steps" -ForegroundColor Gray
    }
} catch {
    Write-Host "  [!] PyTorch not installed" -ForegroundColor Yellow
    Write-Host "      Will be installed in later steps" -ForegroundColor Gray
}
Write-Host ""

# Check 4: CUDA Toolkit
Write-Host "[4/7] Checking CUDA Toolkit (nvcc)..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>$null
    if ($nvccVersion -match "release (\d+\.\d+)") {
        $cudaVersion = $matches[1]
        Write-Host "  [OK] CUDA Toolkit detected: $cudaVersion" -ForegroundColor Green
    } else {
        Write-Host "  [X] CUDA Toolkit (nvcc) not found" -ForegroundColor Red
        Write-Host "      REQUIRED: Download from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
        Write-Host "      Minimum version: 12.1" -ForegroundColor Gray
        $allGood = $false
    }
} catch {
    Write-Host "  [X] CUDA Toolkit (nvcc) not found" -ForegroundColor Red
    Write-Host "      REQUIRED: Download from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    Write-Host "      Minimum version: 12.1" -ForegroundColor Gray
    $allGood = $false
}
Write-Host ""

# Check 5: Visual Studio Build Tools
Write-Host "[5/7] Checking Visual Studio Build Tools..." -ForegroundColor Yellow
$vsWherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWherePath) {
    $vsInfo = & $vsWherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion 2>$null
    if ($vsInfo) {
        Write-Host "  [OK] Visual Studio Build Tools detected: $vsInfo" -ForegroundColor Green
    } else {
        Write-Host "  [X] Visual Studio C++ Build Tools not found" -ForegroundColor Red
        Write-Host "      REQUIRED: Download from https://visualstudio.microsoft.com/downloads/" -ForegroundColor Red
        Write-Host "      Select 'Desktop development with C++' workload" -ForegroundColor Gray
        $allGood = $false
    }
} else {
    Write-Host "  [X] Visual Studio Build Tools not found" -ForegroundColor Red
    Write-Host "      REQUIRED: Download from https://visualstudio.microsoft.com/downloads/" -ForegroundColor Red
    Write-Host "      Select 'Desktop development with C++' workload" -ForegroundColor Gray
    $allGood = $false
}
Write-Host ""

# Check 6: Virtual Environment
Write-Host "[6/7] Checking for virtual environment..." -ForegroundColor Yellow
if (Test-Path "turbodiffusion_env") {
    Write-Host "  [OK] Virtual environment 'turbodiffusion_env' found" -ForegroundColor Green
} else {
    Write-Host "  [!] Virtual environment not created yet" -ForegroundColor Yellow
    Write-Host "      Will be created in Step 3" -ForegroundColor Gray
}
Write-Host ""

# Check 7: Disk Space
Write-Host "[7/7] Checking available disk space..." -ForegroundColor Yellow
$drive = (Get-Location).Drive.Name
$disk = Get-PSDrive $drive
$freeSpaceGB = [math]::Round($disk.Free / 1GB, 2)
if ($freeSpaceGB -gt 35) {
    Write-Host "  [OK] Sufficient disk space: $freeSpaceGB GB free" -ForegroundColor Green
} elseif ($freeSpaceGB -gt 25) {
    Write-Host "  [!] Disk space tight: $freeSpaceGB GB free (35GB+ recommended)" -ForegroundColor Yellow
} else {
    Write-Host "  [X] Insufficient disk space: $freeSpaceGB GB free (35GB+ required)" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Summary
Write-Host "======================================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "[OK] All critical prerequisites are installed!" -ForegroundColor Green
    Write-Host "You can proceed to Phase 2: Environment Setup" -ForegroundColor Green
    Write-Host "Run: .\setup_scripts\1_create_environment.ps1" -ForegroundColor Cyan
} else {
    Write-Host "[X] Missing critical prerequisites!" -ForegroundColor Red
    Write-Host "" 
    Write-Host "Please install the missing components marked with [X] above." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Quick links:" -ForegroundColor Cyan
    Write-Host "  - CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Gray
    Write-Host "  - VS Build Tools: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "After installation, restart PowerShell and run this script again." -ForegroundColor Yellow
}
Write-Host "======================================================" -ForegroundColor Cyan
