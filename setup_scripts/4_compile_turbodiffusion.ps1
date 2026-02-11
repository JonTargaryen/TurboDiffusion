# ======================================================================
# TurboDiffusion Installation - Phase 4: Compile TurboDiffusion
# ======================================================================
# This script compiles TurboDiffusion with CUDA extensions
# Prerequisites: All dependencies must be installed
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Phase 4: Compile CUDA Extensions" -ForegroundColor Cyan
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

# Check CUDA Toolkit
Write-Host "Checking CUDA Toolkit..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ CUDA Toolkit found" -ForegroundColor Green
        $cudaVer = $nvccVersion | Select-String -Pattern "release (\d+\.\d+)"
        Write-Host "  Version: $($cudaVer.Matches[0].Groups[1].Value)" -ForegroundColor Gray
    } else {
        Write-Host "✗ CUDA Toolkit not found (nvcc not in PATH)" -ForegroundColor Red
        Write-Host ""
        Write-Host "CUDA Toolkit is REQUIRED for compilation!" -ForegroundColor Yellow
        Write-Host "Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Press any key to exit..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
} catch {
    Write-Host "✗ CUDA Toolkit not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
Write-Host ""

# Check Visual Studio Build Tools
Write-Host "Checking Visual Studio Build Tools..." -ForegroundColor Yellow
try {
    $clCheck = cl.exe 2>&1
    if ($LASTEXITCODE -ne 9009) {
        Write-Host "✓ Visual Studio Build Tools found" -ForegroundColor Green
    } else {
        # Try to find and setup VS environment
        Write-Host "⚠ cl.exe not in PATH, searching for Visual Studio..." -ForegroundColor Yellow
        
        $vsLocations = @(
            "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
            "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
            "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
            "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        )
        
        $vsFound = $null
        foreach ($loc in $vsLocations) {
            if (Test-Path $loc) {
                $vsFound = $loc
                break
            }
        }
        
        if ($vsFound) {
            Write-Host "✓ Found Visual Studio at:" -ForegroundColor Green
            Write-Host "  $vsFound" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Setting up Visual Studio environment..." -ForegroundColor Yellow
            
            # Run vcvarsall.bat and capture environment
            cmd /c "`"$vsFound`" x64 && set" | ForEach-Object {
                if ($_ -match "^(.*?)=(.*)$") {
                    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
                }
            }
            
            # Verify CL is now available
            $clCheckAgain = cl.exe 2>&1
            if ($LASTEXITCODE -ne 9009) {
                Write-Host "✓ Visual Studio environment configured" -ForegroundColor Green
            } else {
                Write-Host "✗ Failed to configure Visual Studio environment" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "✗ Visual Studio Build Tools not found!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
            Write-Host "Select 'Desktop development with C++' workload" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Press any key to exit..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 1
        }
    }
} catch {
    Write-Host "✗ Visual Studio Build Tools not configured" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Change to project directory
$projectDir = "c:\Users\soumi\TurboWan\TurboDiffusion"
Write-Host "Changing to project directory..." -ForegroundColor Yellow
Set-Location $projectDir
Write-Host "✓ In directory: $projectDir" -ForegroundColor Green
Write-Host ""

# Check if gitsubmodules are initialized
Write-Host "Checking CUTLASS submodule..." -ForegroundColor Yellow
$cutlassPath = Join-Path $projectDir "turbodiffusion\ops\cutlass"
if (Test-Path (Join-Path $cutlassPath "include\cutlass\cutlass.h")) {
    Write-Host "✓ CUTLASS submodule present" -ForegroundColor Green
} else {
    Write-Host "⚠ CUTLASS submodule not initialized" -ForegroundColor Yellow
    Write-Host "Initializing git submodules..." -ForegroundColor Yellow
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to initialize submodules" -ForegroundColor Red
        Write-Host "Try running manually: git submodule update --init --recursive" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✓ Submodules initialized" -ForegroundColor Green
}
Write-Host ""

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
    Write-Host "✓ Removed build directory" -ForegroundColor Green
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
    Write-Host "✓ Removed dist directory" -ForegroundColor Green
}
if (Test-Path "turbodiffusion.egg-info") {
    Remove-Item -Recurse -Force "turbodiffusion.egg-info"
    Write-Host "✓ Removed egg-info directory" -ForegroundColor Green
}
Write-Host ""

# Compile TurboDiffusion
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Compiling TurboDiffusion with CUDA extensions..." -ForegroundColor Yellow
Write-Host "This may take 10-30 minutes depending on your system" -ForegroundColor Gray
Write-Host "" Write-Host "What's being compiled:" -ForegroundColor Gray
Write-Host "  - Custom CUDA kernels (GEMM, LayerNorm, RMSNorm, Quantization)" -ForegroundColor Gray
Write-Host "  - CUTLASS library integration" -ForegroundColor Gray
Write-Host "  - RTX 5090 optimizations (compute capability 12.0a)" -ForegroundColor Gray
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$env:MAX_JOBS = "4"  # Limit parallel jobs to avoid OOM during compilation

# Run compilation
pip install -e . --no-build-isolation

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [X] Compilation failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues and solutions:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. CUDA Version Mismatch (MOST COMMON):" -ForegroundColor White
    Write-Host "   - See: CUDA_VERSION_ERRORS.md for complete error catalog" -ForegroundColor Cyan
    Write-Host "   - Check CUDA: nvcc --version (must be 12.8+ for RTX 5090)" -ForegroundColor Gray
    Write-Host "   - Check PyTorch: python -c \"import torch; print(torch.version.cuda)\"" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. CUTLASS Submodule Missing:" -ForegroundColor White
    Write-Host "   git submodule update --init --recursive" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. MSVC Compiler Errors:" -ForegroundColor White
    Write-Host "   - Ensure Visual Studio 2022 Build Tools installed" -ForegroundColor Gray
    Write-Host "   - Try running from 'Developer Command Prompt for VS 2022'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Out of Memory During Compilation:" -ForegroundColor White
    Write-Host "   - Close other applications" -ForegroundColor Gray
    Write-Host "   - Set: `$env:MAX_JOBS=2" -ForegroundColor Gray
    Write-Host ""
    Write-Host "For detailed error patterns and solutions:" -ForegroundColor Cyan
    Write-Host "  See: CUDA_VERSION_ERRORS.md" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host ""
Write-Host "✓ Compilation successful!" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Verifying TurboDiffusion installation..." -ForegroundColor Yellow
Write-Host ""

try {
    $verifyImport = python -c "import turbo_diffusion_ops; print('✓ turbo_diffusion_ops module loaded'); from turbodiffusion.ops.core import int8_quant; print('✓ CUDA operations available')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $verifyImport -ForegroundColor Green
        Write-Host ""
        Write-Host "✓ TurboDiffusion CUDA extensions verified!" -ForegroundColor Green
    } else {
        Write-Host "⚠ Module imports but CUDA ops may not be working:" -ForegroundColor Yellow
        Write-Host $verifyImport -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Failed to import TurboDiffusion modules" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "✓ Phase 4 Complete - TurboDiffusion Compiled" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEP:" -ForegroundColor Yellow
Write-Host "Run: .\setup_scripts\5_download_models.ps1" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
