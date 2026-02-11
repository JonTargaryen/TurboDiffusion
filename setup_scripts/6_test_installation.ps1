# ======================================================================
# TurboDiffusion Installation - Phase 6: Test Installation
# ======================================================================
# This script tests the TurboDiffusion installation with a quick inference
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Phase 6: Test Installation" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
$projectDir = "c:\Users\soumi\TurboWan\TurboDiffusion"
Set-Location $projectDir

# Check virtual environment
Write-Host "Checking environment..." -ForegroundColor Yellow
$venvActive = $env:VIRTUAL_ENV
if (-not $venvActive) {
    Write-Host "  [X] Not in virtual environment!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Activate with: .\turbodiffusion_env\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
Write-Host "  [OK] Virtual environment active: $venvActive" -ForegroundColor Green
Write-Host ""

# Test 1: Basic imports
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Test 1/5: Verifying Python imports" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$imports = @(
    @{Module="torch"; Display="PyTorch"},
    @{Module="turbo_diffusion_ops"; Display="TurboDiffusion CUDA ops"},
    @{Module="turbodiffusion.ops.core"; Display="TurboDiffusion Core"},
    @{Module="triton"; Display="Triton"},
    @{Module="einops"; Display="Einops"},
    @{Module="transformers"; Display="Transformers"}
)

$allImportsOk = $true
foreach ($import in [$ | H"  Testing $($import.Display)..." -ForegroundColor Gray
    try {
        $result = python -c "import $($import.Module); print('OK')" 2>&1
        if ($LASTEXITCODE -eq 0 -and $result -match "OK") {
            Write-Host "  ✓ $($import.Display)" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $($import.Display) - $result" -ForegroundColor Red
            $allImportsOk = $false
        }
    } catch {
        Write-Host "  ✗ $($import.Display) - $_" -ForegroundColor Red
        $allImportsOk = $false
    }
}
Write-Host ""

if (-not $allImportsOk) {
    Write-Host "⚠ Some imports failed - installation may be incomplete" -ForegroundColor Yellow
    $response = Read-Host "Continue with tests anyway? (yes/no)"
    if ($response -ne "yes" -and $response -ne "y") {
        exit 1
    }
    Write-Host ""
}

# Test 2: CUDA availability
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Test 2/5: Verifying CUDA setup" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$cudaTest = python -c @"
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'Device Count: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device Capability: {torch.cuda.get_device_capability(0)}')
    print(f'Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print('ERROR: CUDA not available!')
"@ 2>&1

Write-Host $cudaTest -ForegroundColor Gray
Write-Host ""

if ($cudaTest -match "CUDA Available: False") {
    Write-Host "✗ CUDA not available - GPU acceleration disabled!" -ForegroundColor Red
    Write-Host "This will prevent TurboDiffusion from running." -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
} else {
    Write-Host "✓ CUDA setup verified" -ForegroundColor Green
}
Write-Host ""

# Test 3: Check model files
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Test 3/5: Verifying model checkpoints" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$requiredFiles = @(
    "checkpoints\Wan2.1_VAE.pth",
    "checkpoints\models_t5_umt5-xxl-enc-bf16.pth"
)

$optionalFiles = @(
    "checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth",
    "checkpoints\TurboWan2.1-T2V-14B-480P-quant.pth",
    "checkpoints\TurboWan2.1-T2V-14B-720P-quant.pth"
)

$allRequiredPresent = $true
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $projectDir $file
    if (Test-Path $fullPath) {
        $sizeGB = [math]::Round((Get-Item $fullPath).Length / 1GB, 2)
        Write-Host "  ✓ $file ($sizeGB GB)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (REQUIRED)" -ForegroundColor Red
        $allRequiredPresent = $false
    }
}

$modelFound = $false
foreach ($file in $optionalFiles) {
    $fullPath = Join-Path $projectDir $file
    if (Test-Path $fullPath) {
        $sizeGB = [math]::Round((Get-Item $fullPath).Length / 1GB, 2)
        Write-Host "  ✓ $file ($sizeGB GB)" -ForegroundColor Green
        $modelFound = $true
    }
}

Write-Host ""

if (-not $allRequiredPresent) {
    Write-Host "✗ Required model files missing!" -ForegroundColor Red
    Write-Host "Run: .\setup_scripts\5_download_models.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

if (-not $modelFound) {
    Write-Host "⚠ No TurboDiffusion model found!" -ForegroundColor Yellow
    Write-Host "At least one model is required for inference." -ForegroundColor Yellow
    Write-Host "Run: .\setup_scripts\5_download_models.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "✓ All required model files present" -ForegroundColor Green
Write-Host ""

# Test 4: Quick CUDA ops test
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Test 4/5: Testing CUDA operations" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$cudaOpsTest = python -c @"
import torch
from turbodiffusion.ops.core import int8_quant, int8_linear

# Test quantization
x = torch.randn(16, 128, device='cuda', dtype=torch.bfloat16)
x_q, x_s = int8_quant(x)
print(f'✓ Quantization: {x.shape} -> {x_q.shape} (dtype: {x_q.dtype})')

# Test int8 linear
w = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16)
w_q, w_s = int8_quant(w)
y = int8_linear(x, w_q, w_s)
print(f'✓ Int8 Linear: ({x.shape[0]}, {x.shape[1]}) @ ({w_q.shape[1]}, {w_q.shape[0]}) -> {y.shape}')

print('✓ CUDA operations working correctly')
"@ 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $cudaOpsTest -ForegroundColor Gray
    Write-Host ""
    Write-Host "✓ CUDA operations test passed" -ForegroundColor Green
} else {
    Write-Host $cudaOpsTest -ForegroundColor Red
    Write-Host ""
    Write-Host "✗ CUDA operations test failed" -ForegroundColor Red
    Write-Host "The CUDA extensions may not be compiled correctly" -ForegroundColor Yellow
    Write-Host ""
}
Write-Host ""

# Test 5: Quick inference test (optional)
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Test 5/5: Quick Inference Test (Optional)" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Run a quick video generation test? This will take 2-5 seconds. (yes/no)"

if ($response -eq "yes" -or $response -eq "y") {
    Write-Host ""
    Write-Host "Generating test video..." -ForegroundColor Yellow
    Write-Host "This will test end-to-end functionality" -ForegroundColor Gray
    Write-Host ""
    
    # Create output directory
    $outputDir = Join-Path $projectDir "output"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }
    
    # Set environment
    $env:PYTHONPATH = "turbodiffusion"
    
    # Determine which model to use
    $modelPath = ""
    if (Test-Path "checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth") {
        $modelPath = "checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth"
        $modelName = "Wan2.1-1.3B"
    } elseif (Test-Path "checkpoints\TurboWan2.1-T2V-14B-480P-quant.pth") {
        $modelPath = "checkpoints\TurboWan2.1-T2V-14B-480P-quant.pth"
        $modelName = "Wan2.1-14B"
    } elseif (Test-Path "checkpoints\TurboWan2.1-T2V-14B-720P-quant.pth") {
        $modelPath = "checkpoints\TurboWan2.1-T2V-14B-720P-quant.pth"
        $modelName = "Wan2.1-14B"
    } else {
        Write-Host "✗ No model found for testing!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Press any key to exit..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
    
    Write-Host "Using model: $modelName" -ForegroundColor Cyan
    Write-Host "Model path: $modelPath" -ForegroundColor Gray
    Write-Host ""
    
    $startTime = Get-Date
    
    python turbodiffusion\inference\wan2.1_t2v_infer.py `
        --model $modelName `
        --dit_path $modelPath `
        --resolution 480p `
        --prompt "A beautiful sunset over mountains with birds flying" `
        --num_samples 1 `
        --num_steps 4 `
        --num_frames 33 `
        --quant_linear `
        --attention_type sagesla `
        --sla_topk 0.1 `
        --save_path output\test_video.mp4
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host ""
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Video generation successful!" -ForegroundColor Green
        Write-Host "  Time taken: $([math]::Round($duration, 2)) seconds" -ForegroundColor Gray
        Write-Host "  Output: output\test_video.mp4" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Opening video..." -ForegroundColor Yellow
        Start-Process "output\test_video.mp4"
    } else {
        Write-Host "✗ Video generation failed" -ForegroundColor Red
        Write-Host "Check the errors above for troubleshooting" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping inference test" -ForegroundColor Yellow
}

Write-Host ""

# Final summary
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Installation Test Summary" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

if ($allImportsOk -and $allRequiredPresent) {
    Write-Host "✓ TurboDiffusion installation is complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now generate videos with:" -ForegroundColor White
    Write-Host "  python turbodiffusion\inference\wan2.1_t2v_infer.py --help" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Quick start example:" -ForegroundColor White
    Write-Host '  python turbodiffusion\inference\wan2.1_t2v_infer.py \' -ForegroundColor Cyan
    Write-Host '    --model Wan2.1-1.3B \' -ForegroundColor Cyan
    Write-Host '    --dit_path checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth \' -ForegroundColor Cyan
    Write-Host '    --prompt "Your creative prompt here" \' -ForegroundColor Cyan
    Write-Host '    --quant_linear \' -ForegroundColor Cyan
    Write-Host '    --attention_type sagesla' -ForegroundColor Cyan
} else {
    Write-Host "⚠ Installation has some issues" -ForegroundColor Yellow
    Write-Host "Review the test results above and fix any problems" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
