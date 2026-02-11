# ======================================================================
# TurboDiffusion Installation - Phase 5: Download Model Checkpoints
# ======================================================================
# This script downloads required model checkpoints
# Warning: Large downloads (7-16GB total)
# ======================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "TurboDiffusion Phase 5: Download Model Checkpoints" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
$projectDir = "c:\Users\soumi\TurboWan\TurboDiffusion"
Set-Location $projectDir

# Create checkpoints directory
$checkpointsDir = Join-Path $projectDir "checkpoints"
if (-not (Test-Path $checkpointsDir)) {
    Write-Host "Creating checkpoints directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $checkpointsDir | Out-Null
    Write-Host "✓ Created: $checkpointsDir" -ForegroundColor Green
} else {
    Write-Host "✓ Checkpoints directory exists" -ForegroundColor Green
}
Write-Host ""

Set-Location $checkpointsDir

# Function to download file with progress
function Download-FileWithProgress {
    param(
        [string]$Url,
        [string]$OutputFile,
        [string]$Description
    )
    
    Write-Host "Downloading $Description..." -ForegroundColor Yellow
    Write-Host "  URL: $Url" -ForegroundColor Gray
    Write-Host "  Output: $OutputFile" -ForegroundColor Gray
    Write-Host ""
    
    if (Test-Path $OutputFile) {
        $fileInfo = Get-Item $OutputFile
        $sizeGB = [math]::Round($fileInfo.Length / 1GB, 2)
        Write-Host "  ℹ File already exists ($sizeGB GB)" -ForegroundColor Cyan
        $response = Read-Host "  Re-download? (yes/no)"
        if ($response -ne "yes" -and $response -ne "y") {
            Write-Host "  ✓ Skipping download" -ForegroundColor Green
            return $true
        }
        Remove-Item $OutputFile -Force
    }
    
    try {
        # Use curl (built into Windows 10+) for better progress display
        $curlPath = "C:\Windows\System32\curl.exe"
        if (Test-Path $curlPath) {
            & $curlPath -L -o $OutputFile --progress-bar $Url
            if ($LASTEXITCODE -eq 0) {
                $fileInfo = Get-Item $OutputFile
                $sizeGB = [math]::Round($fileInfo.Length / 1GB, 2)
                Write-Host "  ✓ Downloaded successfully ($sizeGB GB)" -ForegroundColor Green
                return $true
            } else {
                Write-Host "  ✗ Download failed" -ForegroundColor Red
                return $false
            }
        } else {
            # Fallback to Invoke-WebRequest
            Invoke-WebRequest -Uri $Url -OutFile $OutputFile
            $fileInfo = Get-Item $OutputFile
            $sizeGB = [math]::Round($fileInfo.Length / 1GB, 2)
            Write-Host "  ✓ Downloaded successfully ($sizeGB GB)" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "  ✗ Download failed: $_" -ForegroundColor Red
        return $false
    }
}

# Base models (REQUIRED for all inference)
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Step 1/3: Downloading Base Models (REQUIRED)" -ForegroundColor Yellow
Write-Host "These are needed for all model variants" -ForegroundColor Gray
Write-Host "Total size: ~6GB" -ForegroundColor Gray
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$success = $true

# VAE model
$success = $success -and (Download-FileWithProgress `
    -Url "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth" `
    -OutputFile "Wan2.1_VAE.pth" `
    -Description "VAE Model (~1.5GB)")

Write-Host ""

# Text encoder
$success = $success -and (Download-FileWithProgress `
    -Url "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth" `
    -OutputFile "models_t5_umt5-xxl-enc-bf16.pth" `
    -Description "umT5 Text Encoder (~4.5GB)")

Write-Host ""

if (-not $success) {
    Write-Host "✗ Failed to download base models" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check internet connection" -ForegroundColor White
    Write-Host "2. Try downloading manually from Hugging Face" -ForegroundColor White
    Write-Host "3. Ensure you have enough disk space" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# TurboDiffusion models (Choose one)
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Step 2/3: Choose TurboDiffusion Model" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available models:" -ForegroundColor White
Write-Host ""
Write-Host "1. TurboWan2.1-T2V-1.3B-480P-quant (~1.5GB)" -ForegroundColor Cyan
Write-Host "   - RECOMMENDED for RTX 5090" -ForegroundColor Gray
Write-Host "   - Fastest generation (~2 seconds)" -ForegroundColor Gray
Write-Host "   - 480p resolution" -ForegroundColor Gray
Write-Host "   - Good quality for most use cases" -ForegroundColor Gray
Write-Host ""
Write-Host "2. TurboWan2.1-T2V-14B-480P-quant (~10GB)" -ForegroundColor Cyan
Write-Host "   - Higher quality at 480p" -ForegroundColor Gray
Write-Host "   - Slower (~5-10 seconds)" -ForegroundColor Gray
Write-Host "   - Requires more VRAM" -ForegroundColor Gray
Write-Host ""
Write-Host "3. TurboWan2.1-T2V-14B-720P-quant (~10GB)" -ForegroundColor Cyan
Write-Host "   - Best quality at 720p" -ForegroundColor Gray
Write-Host "   - Slowest (~10-15 seconds)" -ForegroundColor Gray
Write-Host "   - Maximum VRAM usage" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Download all models (~22GB total)" -ForegroundColor Cyan
Write-Host ""
Write-Host "5. Skip model download (use existing)" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "Enter choice (1-5)"

$modelDownloads = @()

switch ($choice) {
    "1" {
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth"
            File = "TurboWan2.1-T2V-1.3B-480P-quant.pth"
            Desc = "TurboWan2.1 1.3B 480P Quantized"
        }
    }
    "2" {
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-480P/resolve/main/TurboWan2.1-T2V-14B-480P-quant.pth"
            File = "TurboWan2.1-T2V-14B-480P-quant.pth"
            Desc = "TurboWan2.1 14B 480P Quantized"
        }
    }
    "3" {
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-720P/resolve/main/TurboWan2.1-T2V-14B-720P-quant.pth"
            File = "TurboWan2.1-T2V-14B-720P-quant.pth"
            Desc = "TurboWan2.1 14B 720P Quantized"
        }
    }
    "4" {
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth"
            File = "TurboWan2.1-T2V-1.3B-480P-quant.pth"
            Desc = "TurboWan2.1 1.3B 480P Quantized"
        }
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-480P/resolve/main/TurboWan2.1-T2V-14B-480P-quant.pth"
            File = "TurboWan2.1-T2V-14B-480P-quant.pth"
            Desc = "TurboWan2.1 14B 480P Quantized"
        }
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-720P/resolve/main/TurboWan2.1-T2V-14B-720P-quant.pth"
            File = "TurboWan2.1-T2V-14B-720P-quant.pth"
            Desc = "TurboWan2.1 14B 720P Quantized"
        }
    }
    "5" {
        Write-Host "Skipping model downloads" -ForegroundColor Yellow
    }
    default {
        Write-Host "Invalid choice. Defaulting to option 1 (1.3B model)" -ForegroundColor Yellow
        $modelDownloads += @{
            Url = "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth"
            File = "TurboWan2.1-T2V-1.3B-480P-quant.pth"
            Desc = "TurboWan2.1 1.3B 480P Quantized"
        }
    }
}

Write-Host ""

foreach ($model in $modelDownloads) {
    $success = $success -and (Download-FileWithProgress `
        -Url $model.Url `
        -OutputFile $model.File `
        -Description $model.Desc)
    Write-Host ""
}

# Summary
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Step 3/3: Download Summary" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$files = Get-ChildItem -Path $checkpointsDir -File
$totalSize = ($files | Measure-Object -Property Length -Sum).Sum
$totalSizeGB = [math]::Round($totalSize / 1GB, 2)

Write-Host "Downloaded files:" -ForegroundColor White
foreach ($file in $files) {
    $sizeGB = [math]::Round($file.Length / 1GB, 2)
    Write-Host "  ✓ $($file.Name) ($sizeGB GB)" -ForegroundColor Green
}
Write-Host ""
Write-Host "Total size: $totalSizeGB GB" -ForegroundColor Cyan
Write-Host ""

if ($success) {
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host "✓ Phase 5 Complete - Models Downloaded" -ForegroundColor Green
    Write-Host ""
    Write-Host "NEXT STEP:" -ForegroundColor Yellow
    Write-Host "Run: .\setup_scripts\6_test_installation.ps1" -ForegroundColor Cyan
    Write-Host "======================================================" -ForegroundColor Cyan
} else {
    Write-Host "⚠ Some downloads may have failed" -ForegroundColor Yellow
    Write-Host "Check the files above and re-run if needed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
