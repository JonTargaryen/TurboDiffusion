# TurboDiffusion Installation & Setup Guide

## Quick Start (For Experienced Users)

```powershell
# 1. Run system check
.\setup_scripts\0_system_check.ps1

# 2. Install prerequisites (if missing):
#    - Visual Studio 2022 Build Tools (C++ workload)
#    - CUDA Toolkit 12.6+

# 3. Create environment
.\setup_scripts\1_create_environment.ps1
.\turbodiffusion_env\Scripts\Activate.ps1

# 4. Install PyTorch
.\setup_scripts\2_install_pytorch.ps1

# 5. Install dependencies
.\setup_scripts\3_install_dependencies.ps1

# 6. Compile CUDA extensions
.\setup_scripts\4_compile_turbodiffusion.ps1

# 7. Download models
.\setup_scripts\5_download_models.ps1

# 8. Test installation
.\setup_scripts\6_test_installation.ps1
```

## Step-by-Step Installation Guide

### Phase 0: Prerequisites (Manual Installation Required)

#### 1. Visual Studio 2022 Build Tools
- **Download**: [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
- Select "Desktop development with C++"
- Required components:
  - MSVC v143 - VS 2022 C++ x64/x86 build tools
  - Windows 10/11 SDK
  - CMake tools for Windows
- **Size**: ~7GB
- **Time**: 15-30 minutes

#### 2. CUDA Toolkit 12.6 or Later
- **Download**: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Choose: Windows → x86_64 → 11 → exe (network)
- **IMPORTANT**: Check "Add to PATH" during installation
- **Your RTX 5090 needs**: CUDA 12.6+ (minimum 12.1)
- **Size**: ~3.5GB
- **Time**: 20-40 minutes
- After installation, verify:
  ```powershell
  nvcc --version
  ```

#### 3. Python 3.9-3.12
- **Already installed**: Python 3.12.10 ✓
- Virtual environment will be created automatically
- No additional installation needed

### Phase 1-6: Automated Installation

Once prerequisites are installed, run the automated scripts in order:

#### Run System Check
```powershell
cd c:\Users\soumi\TurboWan\TurboDiffusion
.\setup_scripts\0_system_check.ps1
```

This will verify all prerequisites are installed correctly.

#### Create Virtual Environment
```powershell
.\setup_scripts\1_create_environment.ps1
.\turbodiffusion_env\Scripts\Activate.ps1
```

#### Install PyTorch (in activated environment)
```powershell
.\setup_scripts\2_install_pytorch.ps1
```

#### Install Dependencies
```powershell
.\setup_scripts\3_install_dependencies.ps1
```

#### Compile TurboDiffusion
```powershell
.\setup_scripts\4_compile_turbodiffusion.ps1
```
⏱️ This step takes 10-30 minutes

#### Download Models
```powershell
.\setup_scripts\5_download_models.ps1
```
Choose which model to download (recommended: 1.3B for fastest generation)

#### Test Installation
```powershell
.\setup_scripts\6_test_installation.ps1
```

## Usage

### Basic Text-to-Video Generation

```powershell
.\turbodiffusion_env\Scripts\Activate.ps1
$env:PYTHONPATH="turbodiffusion"

python turbodiffusion\inference\wan2.1_t2v_infer.py `
    --model Wan2.1-1.3B `
    --dit_path checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth `
    --prompt "A beautiful sunset over mountains" `
    --num_steps 4 `
    --quant_linear `
    --attention_type sagesla `
    --save_path output\my_video.mp4
```

### Command Line Arguments

- `--model`: Model size (`Wan2.1-1.3B` or `Wan2.1-14B`)
- `--dit_path`: Path to model checkpoint
- `--prompt`: Text description of video (use long, detailed prompts)
- `--num_steps`: Sampling steps (1-4, default: 4)
- `--num_frames`: Number of frames (default: 81 for 5 seconds at 16fps)
- `--resolution`: Output resolution (`480p` or `720p`)
- `--aspect_ratio`: Aspect ratio (`16:9`, `9:16`, `4:3`, etc.)
- `--quant_linear`: Enable quantization (required for quantized models)
- `--attention_type`: Attention mechanism (`sagesla`, `sla`, or `original`)
- `--sla_topk`: Top-k ratio for SLA (default: 0.1, try 0.15 for better quality)
- `--seed`: Random seed for reproducibility
- `--save_path`: Output video path

### Performance Tips

1. **For Fastest Generation** (~2 seconds on RTX 5090):
   - Use `--model Wan2.1-1.3B`
   - Use quantized model with `--quant_linear`
   - Use `--attention_type sagesla`
   - Use `--resolution 480p`
   - Use `--num_steps 4`

2. **For Best Quality**:
   - Use `--model Wan2.1-14B`
   - Use `--resolution 720p`
   - Use `--sla_topk 0.15`
   - Use longer, more detailed prompts

3. **If Running Out of Memory**:
   - Use smaller model (1.3B instead of 14B)
   - Use lower resolution (480p instead of 720p)
   - Close other GPU-using applications
   - Reduce `--num_frames`

## Troubleshooting

### Run Diagnostics
If you encounter issues, run the diagnostics script:
```powershell
.\diagnose.ps1
```

This generates a comprehensive report of your system configuration.

### Common Issues

#### "nvcc not found"
**Solution**: 
1. Restart PowerShell after CUDA installation
2. Manually add CUDA to PATH:
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
   ```

#### "cl.exe not found" (Visual Studio)
**Solution**:
1. Ensure Visual Studio 2022 Build Tools are installed with C++ workload
2. Run Visual Studio Developer PowerShell
3. Or manually run:
   ```powershell
   & "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
   ```

#### Compilation fails with CUTLASS errors
**Solution**:
1. Ensure git submodules are initialized:
   ```powershell
   git submodule update --init --recursive
   ```
2. Clean and retry:
   ```powershell
   Remove-Item -Recurse -Force build, dist, *.egg-info
   pip install -e . --no-build-isolation
   ```

#### Out of Memory (OOM) during generation
**Solution**:
1. Use quantized models with `--quant_linear`
2. Use 1.3B model instead of 14B
3. Lower resolution (480p instead of 720p)
4. Close other GPU applications
5. Monitor GPU memory:
   ```powershell
   nvidia-smi
   ```

#### Slow generation (>10 seconds)
**Solution**:
1. Ensure using `--quant_linear` flag
2. Ensure using `--attention_type sagesla`
3. Install SpargeAttn if not installed:
   ```powershell
   pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
   ```
4. Check GPU utilization during generation

#### Flash Attention fails to compile
**Solution**:
Flash Attention is optional. If it fails:
1. Skip with: Continue installation without it
2. TurboDiffusion will still work with slightly reduced performance
3. The custom CUDA ops are more important than flash-attn

## Project Structure

```
TurboDiffusion/
├── setup_scripts/          # Installation scripts
│   ├── 0_system_check.ps1
│   ├── 1_create_environment.ps1
│   ├── 2_install_pytorch.ps1
│   ├── 3_install_dependencies.ps1
│   ├── 4_compile_turbodiffusion.ps1
│   ├── 5_download_models.ps1
│   └── 6_test_installation.ps1
├── turbodiffusion/         # Main package
│   ├── ops/               # Custom CUDA operations
│   ├── inference/         # Inference scripts
│   ├── rcm/              # rCM components
│   └── imaginaire/       # Training components
├── checkpoints/           # Model checkpoints (downloaded)
├── scripts/              # Inference bash scripts
├── install.ps1           # Master installation script
├── diagnose.ps1          # Diagnostic tool
└── INSTALLATION_ANALYSIS.md  # This file

```

## System Requirements

### Minimum
- Windows 11 (Windows 10 may work)
- NVIDIA GPU with 16GB+ VRAM (RTX 4090, RTX 5090, etc.)
- 32GB system RAM
- 30GB free disk space
- Python 3.9+
- CUDA 12.6+
- Visual Studio 2022 Build Tools

### Recommended (for RTX 5090)
- Windows 11
- NVIDIA RTX 5090 (32GB VRAM)
- 64GB system RAM
- 50GB free SSD space
- Python 3.12
- CUDA 12.6 or 12.8
- Visual Studio 2022

## Model Information

### Available Models

| Model | Size | Best Resolution | Speed (RTX 5090) | VRAM |
|-------|------|----------------|------------------|------|
| TurboWan2.1-T2V-1.3B-480P-quant | ~1.5GB | 480p | ~2 seconds | ~8GB |
| TurboWan2.1-T2V-14B-480P-quant | ~10GB | 480p | ~5-8 seconds | ~16GB |
| TurboWan2.1-T2V-14B-720P-quant | ~10GB | 720p | ~10-15 seconds | ~24GB |
| TurboWan2.2-I2V-A14B-720P-quant | ~10GB × 2 | 720p (I2V) | ~10-15 seconds | ~20GB |

### Model Files (All Required)
- `Wan2.1_VAE.pth` (~1.5GB) - VAE for encoding/decoding
- `models_t5_umt5-xxl-enc-bf16.pth` (~4.5GB) - Text encoder
- At least one TurboDiffusion model checkpoint

## Advanced Usage

### Interactive TUI Mode
```powershell
python turbodiffusion\inference\wan2.1_t2v_infer.py --serve
```

This launches an interactive terminal UI for generating videos.

### Batch Generation
Create a file `prompts.txt` with one prompt per line, then:
```powershell
Get-Content prompts.txt | ForEach-Object {
    python turbodiffusion\inference\wan2.1_t2v_infer.py `
        --model Wan2.1-1.3B `
        --dit_path checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth `
        --prompt $_ `
        --quant_linear `
        --save_path "output\$(Get-Date -Format 'yyyyMMddHHmmss').mp4"
}
```

### Custom Model Quantization
If you have unquantized models:
```powershell
python turbodiffusion\inference\modify_model.py `
    --input_path path\to\original_model.pth `
    --output_path path\to\quantized_model.pth `
    --model Wan2.1-1.3B `
    --attention_type sla `
    --quant_linear
```

## Additional Resources

- **Paper**: [TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times](https://arxiv.org/pdf/2512.16093)
- **GitHub**: https://github.com/thu-ml/TurboDiffusion
- **SageAttention**: https://github.com/thu-ml/SageAttention
- **CUTLASS**: https://github.com/NVIDIA/cutlass

## Getting Help

1. **Run diagnostics**: `.\diagnose.ps1`
2. **Check logs**: Look in `logs\installation_*.log`
3. **Review installation analysis**: Read `INSTALLATION_ANALYSIS.md`
4. **GitHub Issues**: Report issues at https://github.com/thu-ml/TurboDiffusion/issues

## License

TurboDiffusion is licensed under Apache 2.0. See LICENSE file for details.
