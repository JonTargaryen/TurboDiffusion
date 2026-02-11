# ======================================================================
# TurboDiffusion Debugging and Diagnostics Script
# ======================================================================
# Run this script to diagnose issues and get detailed system information
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Diagnostics & Debugging" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$diagFile = "turbodiffusion_diagnostics_$timestamp.txt"

function Write-Diag {
    param([string]$Message)
    Add-Content -Path $diagFile -Value $Message
    Write-Host $Message
}

Write-Diag "TurboDiffusion Diagnostics Report"
Write-Diag "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Diag ("=" * 70)
Write-Diag ""

# Section 1: System Information
Write-Diag "[1] SYSTEM INFORMATION"
Write-Diag ("=" * 70)
Write-Diag "OS: $([System.Environment]::OSVersion.VersionString)"
Write-Diag "Computer: $env:COMPUTERNAME"
Write-Diag "User: $env:USERNAME"
Write-Diag "PowerShell: $($PSVersionTable.PSVersion)"
Write-Diag ""

# Section 2: GPU Information
Write-Diag "[2] GPU INFORMATION"
Write-Diag ("=" * 70)
try {
    $nvidiaInfo = nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free,memory.used --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Diag "GPU: $nvidiaInfo"
        
        # Get detailed GPU info
        $gpuTopology = nvidia-smi topo -m 2>&1
        Write-Diag "`nGPU Topology:"
        Write-Diag $gpuTopology
    } else {
        Write-Diag "ERROR: nvidia-smi failed"
        Write-Diag $nvidiaInfo
    }
} catch {
    Write-Diag "ERROR: NVIDIA driver not installed or nvidia-smi not found"
}
Write-Diag ""

# Section 3: CUDA Toolkit
Write-Diag "[3] CUDA TOOLKIT"
Write-Diag ("=" * 70)
try {
    $nvccOutput = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Diag $nvccOutput
        
        # Check CUDA environment variables
        Write-Diag "`nCUDA Environment Variables:"
        Write-Diag "CUDA_HOME: $env:CUDA_HOME"
        Write-Diag "CUDA_PATH: $env:CUDA_PATH"
    } else {
        Write-Diag "ERROR: nvcc not found in PATH"
        Write-Diag "CUDA_HOME: $env:CUDA_HOME"
        Write-Diag "CUDA_PATH: $env:CUDA_PATH"
    }
} catch {
    Write-Diag "ERROR: CUDA Toolkit not installed"
}
Write-Diag ""

# Section 4: Visual Studio Build Tools
Write-Diag "[4] VISUAL STUDIO BUILD TOOLS"
Write-Diag ("=" * 70)
try {
    $clOutput = cl.exe 2>&1
    if ($clOutput -match "Microsoft") {
        Write-Diag "Visual Studio C++ Compiler found"
        Write-Diag ($clOutput | Select-String -Pattern "Version")
    } else {
        Write-Diag "WARNING: cl.exe not in PATH"
        
        # Search for VS installations
        $vsLocations = @(
            "C:\Program Files\Microsoft Visual Studio\2022\Community",
            "C:\Program Files\Microsoft Visual Studio\2022\Professional",
            "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
            "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
        )
        
        foreach ($loc in $vsLocations) {
            if (Test-Path $loc) {
                Write-Diag "Found Visual Studio at: $loc"
            }
        }
    }
} catch {
    Write-Diag "ERROR: Visual Studio Build Tools not found"
}
Write-Diag ""

# Section 5: Python Environment
Write-Diag "[5] PYTHON ENVIRONMENT"
Write-Diag ("=" * 70)
try {
    $pythonVersion = python --version 2>&1
    Write-Diag "Python: $pythonVersion"
    Write-Diag "Python Path: $(where.exe python)"
    
    $condaEnv = $env:CONDA_DEFAULT_ENV
    if ($condaEnv) {
        Write-Diag "Conda Environment: $condaEnv"
    } else {
        Write-Diag "Conda Environment: None (not activated)"
    }
    
    # Check conda
    try {
        $condaVersion = conda --version 2>&1
        Write-Diag "Conda: $condaVersion"
    } catch {
        Write-Diag "Conda: Not installed"
    }
} catch {
    Write-Diag "ERROR: Python not found"
}
Write-Diag ""

# Section 6: Installed Python Packages
Write-Diag "[6] INSTALLED PYTHON PACKAGES"
Write-Diag ("=" * 70)
try {
    $pipList = python -m pip list 2>&1
    Write-Diag $pipList
} catch {
    Write-Diag "ERROR: Could not list Python packages"
}
Write-Diag ""

# Section 7: PyTorch CUDA Test
Write-Diag "[7] PYTORCH CUDA TEST"
Write-Diag ("=" * 70)
try {
    $torchTest = python -c @"
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'Device Count: {torch.cuda.device_count()}')
    print(f'Current Device: {torch.cuda.current_device()}')
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device Capability: {torch.cuda.get_device_capability(0)}')
    print(f'Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Test CUDA operation
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    z = x @ y
    print(f'CUDA Operation Test: OK (result shape: {z.shape})')
else:
    print('CUDA not available!')
"@ 2>&1
    Write-Diag $torchTest
} catch {
    Write-Diag "ERROR: PyTorch not installed or import failed"
    Write-Diag $_
}
Write-Diag ""

# Section 8: TurboDiffusion Installation
Write-Diag "[8] TURBODIFFUSION INSTALLATION"
Write-Diag ("=" * 70)
try {
    $turboTest = python -c @"
try:
    import turbo_diffusion_ops
    print('✓ turbo_diffusion_ops module found')
except ImportError as e:
    print(f'✗ turbo_diffusion_ops not found: {e}')

try:
    from turbodiffusion.ops.core import int8_quant, int8_linear
    print( '✓ TurboDiffusion core operations available')
except ImportError as e:
    print(f'✗ TurboDiffusion core operations not available: {e}')

try:
    import triton
    print(f'✓ Triton version: {triton.__version__}')
except ImportError:
    print('✗ Triton not installed')

try:
    import einops
    print(f'✓ Einops version: {einops.__version__}')
except ImportError:
    print('✗ Einops not installed')

try:
    import transformers
    print(f'✓ Transformers version: {transformers.__version__}')
except ImportError:
    print('✗ Transformers not installed')
"@ 2>&1
    Write-Diag $turboTest
} catch {
    Write-Diag "ERROR: Could not test TurboDiffusion installation"
    Write-Diag $_
}
Write-Diag ""

# Section 9: Model Checkpoints
Write-Diag "[9] MODEL CHECKPOINTS"
Write-Diag ("=" * 70)
$projectDir = "c:\Users\soumi\TurboWan\TurboDiffusion"
$checkpointsDir = Join-Path $projectDir "checkpoints"

if (Test-Path $checkpointsDir) {
    $files = Get-ChildItem -Path $checkpointsDir -File
    if ($files.Count -gt 0) {
        Write-Diag "Found $($files.Count) checkpoint file(s):"
        foreach ($file in $files) {
            $sizeGB = [math]::Round($file.Length / 1GB, 2)
            Write-Diag "  - $($file.Name) ($sizeGB GB)"
        }
        $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
        $totalSizeGB = [math]::Round($totalSize / 1GB, 2)
        Write-Diag "`nTotal size: $totalSizeGB GB"
    } else {
        Write-Diag "Checkpoints directory exists but is empty"
    }
} else {
    Write-Diag "Checkpoints directory not found: $checkpointsDir"
}
Write-Diag ""

# Section 10: Disk Space
Write-Diag "[10] DISK SPACE"
Write-Diag ("=" * 70)
$drives = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -gt 0 }
foreach ($drive in $drives) {
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    $usedGB = [math]::Round($drive.Used / 1GB, 2)
    $totalGB = [math]::Round(($drive.Free + $drive.Used) / 1GB, 2)
    $percentUsed = [math]::Round(($drive.Used / ($drive.Free + $drive.Used)) * 100, 1)
    Write-Diag "Drive $($drive.Name): $usedGB GB used / $totalGB GB total ($freeGB GB free, $percentUsed% used)"
}
Write-Diag ""

# Section 11: Environment Variables
Write-Diag "[11] RELEVANT ENVIRONMENT VARIABLES"
Write-Diag ("=" * 70)
$envVars = @("PATH", "PYTHONPATH", "CUDA_HOME", "CUDA_PATH", "CONDA_DEFAULT_ENV", "MAX_JOBS")
foreach ($var in $envVars) {
    $value = [System.Environment]::GetEnvironmentVariable($var)
    if ($value) {
        if ($var -eq "PATH") {
            Write-Diag "PATH:"
            $value -split ';' | ForEach-Object { Write-Diag "  $_" }
        } else {
            Write-Diag "$var: $value"
        }
    } else {
        Write-Diag "$var: (not set)"
    }
}
Write-Diag ""

# Section 12: Common Issues Check
Write-Diag "[12] COMMON ISSUES CHECK"
Write-Diag ("=" * 70)

$issues = @()

# Check CUDA
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    $issues += "CUDA Toolkit not in PATH (nvcc not found)"
}

# Check VS Build Tools
try {
    $clCheckResult = cl.exe 2>&1
    if (-not ($clCheckResult -match "Microsoft")) {
        $issues += "Visual Studio Build Tools not in PATH (cl.exe not found)"
    }
} catch {
    $issues += "Visual Studio Build Tools not in PATH (cl.exe not accessible)"
}

# Check Python packages
$requiredPackages = @("torch", "triton", "einops", "transformers")
foreach ($pkg in $requiredPackages) {
    try {
        $checkPkg = python -c "import $pkg" 2>&1
        if ($LASTEXITCODE -ne 0) {
            $issues += "Python package '$pkg' not installed"
        }
    } catch {
        $issues += "Python package '$pkg' not installed"
    }
}

# Check disk space
$drive = (Get-Location).Drive.Name
$disk = Get-PSDrive $drive
$freeSpaceGB = [math]::Round($disk.Free / 1GB, 2)
if ($freeSpaceGB -lt 25) {
    $issues += "Low disk space on drive $drive ($freeSpaceGB GB free, 25GB+ recommended)"
}

if ($issues.Count -eq 0) {
    Write-Diag "No common issues detected"
} else {
    Write-Diag "Found $($issues.Count) potential issue(s):"
    foreach ($issue in $issues) {
        Write-Diag "  ✗ $issue"
    }
}
Write-Diag ""

# Final summary
Write-Diag ("=" * 70)
Write-Diag "Diagnostics complete"
Write-Diag "Report saved to: $diagFile"
Write-Diag ("=" * 70)

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Diagnostics report saved to:" -ForegroundColor Green
Write-Host "  $diagFile" -ForegroundColor White
Write-Host ""
Write-Host "Share this file when asking for help!" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
