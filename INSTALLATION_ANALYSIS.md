# TurboDiffusion Installation Analysis & Plan

## Project Overview
**TurboDiffusion** is a video generation acceleration framework that speeds up diffusion models by 100-200x on RTX 5090.

- **Text-to-Video (T2V)**: Generate videos from text prompts
- **Image-to-Video (I2V)**: Generate videos from images
- **Key Technologies**: SageAttention, SLA (Sparse-Linear Attention), rCM distillation, CUTLASS (CUDA kernels)

## Current System Status

### ✅ Working
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **Driver**: 591.74 (supports CUDA 13.1)
- **Python**: 3.12.10 (matches requirements)
- **PyTorch**: 2.11.0.dev20260125+cu128 (development version)

### ❌ Missing Critical Components
1. **CUDA Toolkit** (nvcc compiler not found) - REQUIRED for compiling CUDA extensions
2. **Visual Studio Build Tools** - REQUIRED for CUDA compilation on Windows
3. **Conda** - Highly recommended for environment management
4. **Custom CUDA Extensions** - Not compiled (turbo_diffusion_ops)
5. **Missing Python Packages**:
   - triton >= 3.3.0
   - flash-attn
   - einops
   - loguru
   - omegaconf
   - fvcore
   - transformers
   - Many others...

## Installation Challenges

### 1. Windows + CUDA Compilation
- Requires Visual Studio 2022 with C++ tools
- CUDA Toolkit must match PyTorch CUDA version
- Environment variables must be configured correctly

### 2. Custom CUDA Operations
- CUTLASS library (v4.3.0) embedded in project
- Custom kernels for: quantization, GEMM, LayerNorm, RMSNorm
- Requires proper compilation flags for RTX 5090 (compute capability 12.0a)

### 3. Dependency Chain
- Some packages depend on compiled CUDA extensions
- Installation order matters
- Some packages need --no-build-isolation flag

## Comprehensive Installation Plan

### Phase 1: System Prerequisites (Manual Installation Required)

#### Step 1.1: Install Visual Studio 2022 Build Tools
**Download**: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++"
- Include: MSVC v143, Windows 10/11 SDK, CMake tools
- **Size**: ~7GB
- **Time**: 15-30 minutes

#### Step 1.2: Install CUDA Toolkit 12.6 or 12.8
**Download**: https://developer.nvidia.com/cuda-downloads
- Choose: Windows → x86_64 → 11 → exe (network)
- CUDA 12.6.2 recommended (matches PyTorch)
- **Size**: ~3.5GB
- **Time**: 20-40 minutes
- **Important**: Add to PATH during installation

#### Step 1.3: Install Miniconda3
**Download**: https://docs.conda.io/en/latest/miniconda.html
- Windows 64-bit installer
- **Size**: ~100MB
- **Time**: 5 minutes
- Select "Add to PATH" option

### Phase 2: Environment Setup (Automated)

#### Step 2.1: Create Conda Environment
```powershell
conda create -n turbodiffusion python=3.12 -y
conda activate turbodiffusion
```

#### Step 2.2: Install PyTorch with CUDA
```powershell
# Install PyTorch 2.7.0 with CUDA 12.6
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
```

Note: Avoid PyTorch 2.8+ due to potential OOM issues (per README)

### Phase 3: Install Dependencies (Automated)

#### Step 3.1: Install Core Dependencies
```powershell
pip install triton>=3.3.0
pip install ninja packaging wheel setuptools
```

#### Step 3.2: Install Flash Attention
```powershell
pip install flash-attn --no-build-isolation
```

#### Step 3.3: Install Other Dependencies
```powershell
pip install einops numpy pillow loguru imageio[ffmpeg] pandas PyYAML omegaconf attrs fvcore ftfy regex transformers nvidia-ml-py prompt-toolkit rich
```

### Phase 4: Compile TurboDiffusion (Automated)

#### Step 4.1: Install SpargeAttn (Optional but Recommended)
```powershell
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
```

#### Step 4.2: Compile TurboDiffusion
```powershell
cd c:\Users\soumi\TurboWan\TurboDiffusion
pip install -e . --no-build-isolation
```

### Phase 5: Download Model Checkpoints (Large Downloads)

#### Step 5.1: Create Checkpoints Directory
```powershell
mkdir checkpoints
cd checkpoints
```

#### Step 5.2: Download Base Models (REQUIRED)
```powershell
# VAE model (~1.5GB)
curl -L -o Wan2.1_VAE.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth

# Text encoder (~4.5GB)
curl -L -o models_t5_umt5-xxl-enc-bf16.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
```

#### Step 5.3: Download TurboDiffusion Model (Choose One)
```powershell
# Option A: 1.3B model (quantized, ~1.5GB) - FASTEST, RECOMMENDED FOR RTX 5090
curl -L -o TurboWan2.1-T2V-1.3B-480P-quant.pth https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth

# Option B: 14B model (quantized, ~10GB) - HIGHER QUALITY
# curl -L -o TurboWan2.1-T2V-14B-720P-quant.pth https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-720P/resolve/main/TurboWan2.1-T2V-14B-720P-quant.pth
```

**Total Download Size**: ~7.5GB (1.3B model) or ~16GB (14B model)

### Phase 6: Testing & Verification

#### Step 6.1: Verify CUDA Compilation
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

#### Step 6.2: Verify TurboDiffusion Import
```powershell
python -c "import turbo_diffusion_ops; print('CUDA ops loaded successfully!')"
```

#### Step 6.3: Quick Test Inference (5-second video)
```powershell
$env:PYTHONPATH="turbodiffusion"
python turbodiffusion/inference/wan2.1_t2v_infer.py --model Wan2.1-1.3B --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth --resolution 480p --prompt "A cat playing with a ball of yarn" --num_samples 1 --num_steps 4 --quant_linear --attention_type sagesla --sla_topk 0.1 --save_path output/test_video.mp4
```

Expected time: ~2-5 seconds for generation on RTX 5090

## Potential Issues & Solutions

**⚠️ CRITICAL REFERENCE**: See **[CUDA_VERSION_ERRORS.md](CUDA_VERSION_ERRORS.md)** for complete CUDA version mismatch error catalog (30+ error patterns with solutions)

### Issue 1: CUDA Compilation Fails
**Symptoms**: "nvcc not found", "Unsupported gpu architecture", compilation errors
**Solutions**:
1. **CUDA Version**: Must be 12.8+ for RTX 5090 - `nvcc --version`
2. **VS Build Tools**: Verify `cl.exe` accessible
3. **PyTorch Match**: Check `python -c "import torch; print(torch.version.cuda)"`
4. **Restart**: PowerShell after installations
5. **Detailed Errors**: See [CUDA_VERSION_ERRORS.md](CUDA_VERSION_ERRORS.md) for specific error patterns
6. Set environment variables:
   ```powershell
   $env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
   $env:PATH="$env:CUDA_HOME\bin;$env:PATH"
   ```

### Issue 2: Flash-Attention Compilation Fails
**Symptoms**: Build errors when installing flash-attn
**Solutions**:
1. Ensure CUDA 12.x is installed
2. Use `--no-build-isolation` flag
3. If still fails, skip flash-attn (may reduce performance)

### Issue 3: Out of Memory (OOM)
**Symptoms**: CUDA OOM during generation
**Solutions**:
1. Use quantized models (`--quant_linear` flag)
2. Use 1.3B model instead of 14B
3. Reduce `--num_samples`
4. Lower resolution (480p instead of 720p)

### Issue 4: Slow Generation
**Symptoms**: Generation takes >10 seconds
**Solutions**:
1. Verify using quantized model
2. Ensure `--attention_type sagesla` is set
3. Install SpargeAttn: `pip install git+https://github.com/thu-ml/SpargeAttn.git`
4. Check GPU utilization during inference

## Destructive Options (If All Else Fails)

### Option A: Clean Python Environment
```powershell
# Remove all Python packages
pip freeze > uninstall.txt
pip uninstall -r uninstall.txt -y
# Then restart from Phase 2
```

### Option B: Fresh Conda Install
```powershell
# Remove conda environment
conda deactivate
conda env remove -n turbodiffusion
# Then restart from Phase 2
```

### Option C: WSL2 + Linux (Nuclear Option)
If Windows compilation proves impossible:
1. Install WSL2: `wsl --install`
2. Install Ubuntu 22.04 from Microsoft Store
3. Install CUDA in WSL: Follow NVIDIA WSL-CUDA guide
4. Compile TurboDiffusion in Linux environment
5. Run inference in WSL

**Pros**: Linux compilation is more straightforward
**Cons**: Need to reinstall everything, potential GPU passthrough issues

### Option D: Docker Container (Alternative)
Use pre-built CUDA development container:
```powershell
docker pull nvcr.io/nvidia/pytorch:25.01-py3
docker run --gpus all -it -v c:\Users\soumi\TurboWan\TurboDiffusion:/workspace nvcr.io/nvidia/pytorch:25.01-py3
```

## Resource Requirements

| Component | Size | Time |
|-----------|------|------|
| Visual Studio Build Tools | ~7GB | 15-30 min |
| CUDA Toolkit 12.6 | ~3.5GB | 20-40 min |
| Miniconda | ~100MB | 5 min |
| Python packages | ~5GB | 10-20 min |
| Model checkpoints | 7.5-16GB | 30-60 min |
| **Total** | **~23-32GB** | **1.5-3 hours** |

## Next Steps

1. **Manual installations first**: VS Build Tools → CUDA Toolkit → Miniconda
2. **Restart PowerShell** to refresh environment variables
3. **Run automated setup scripts** (will be provided)
4. **Test with small model** (1.3B) before trying 14B
5. **Monitor GPU usage** during inference

## References

- TurboDiffusion Paper: https://arxiv.org/pdf/2512.16093
- CUTLASS Library: https://github.com/NVIDIA/cutlass
- SageAttention: https://github.com/thu-ml/SageAttention
- PyTorch CUDA: https://pytorch.org/get-started/locally/
