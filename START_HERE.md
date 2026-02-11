# TurboDiffusion Installation - START HERE

## 🚀 What is This?

You have **TurboDiffusion** - a cutting-edge video generation framework that generates videos **100-200x faster** than traditional diffusion models. On your RTX 5090, you can generate a 5-second video in **~2 seconds!**

## ⚡ Quick Status Check

Run this first to see what you need:
```powershell
.\setup_scripts\0_system_check.ps1
```

## 📋 Installation Options

### Option 1: Guided Step-by-Step (RECOMMENDED)
Perfect if you want to understand each step:

1. **Check prerequisites** → `.\setup_scripts\0_system_check.ps1`
2. **Install missing components** (Visual Studio, CUDA) - Links provided by script
3. **Create environment** → `.\setup_scripts\1_create_environment.ps1`
4. **Activate environment** → `.\turbodiffusion_env\Scripts\Activate.ps1`  
5. **Install PyTorch** → `.\setup_scripts\2_install_pytorch.ps1`
6. **Install dependencies** → `.\setup_scripts\3_install_dependencies.ps1`
7. **Compile CUDA extensions** → `.\setup_scripts\4_compile_turbodiffusion.ps1` (10-30 min)
8. **Download models** → `.\setup_scripts\5_download_models.ps1` (7-16GB)
9. **Test installation** → `.\setup_scripts\6_test_installation.ps1`

### Option 2: Automated (For Experienced Users)
After installing prerequisites:
```powershell
.\install.ps1
```

### Option 3: Docker (Coming Soon)
Linux container with everything pre-configured

## 🔧 Prerequisites (Install These First)

### 1. Visual Studio 2022 Build Tools (~7GB, 20-30 min)
- Download: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++"
- **Why**: Needed to compile CUDA code on Windows

### 2. CUDA Toolkit 12.6+ (~3.5GB, 20-40 min)
- Download: https://developer.nvidia.com/cuda-downloads
- **Your RTX 5090 needs**: CUDA 12.6 or 13.0 (minimum 12.1)
- **IMPORTANT**: Check "Add to PATH" during installation
- **Why**: Required for GPU acceleration and CUDA compilation

### 3. Python 3.9-3.12
- Already installed: Python 3.12.10 ✓
- Virtual environment will be created automatically
- **Why**: TurboDiffusion requires Python 3.9+

**After installing**, restart PowerShell!

## 🎯 What You'll Get

| Feature | What It Does |
|---------|-------------|
| **Text-to-Video** | Generate videos from text descriptions |
| **Image-to-Video** | Animate static images |
| **Lightning Fast** | 2-15 seconds per 5-second video (depending on model) |
| **High Quality** | 480p and 720p output |
| **RTX 5090 Optimized** | Uses custom CUDA kernels for your GPU |

## 📊 Model Options

Choose during download (Step 8):

| Model | Size | Speed | VRAM | Best For |
|-------|------|-------|------|----------|
| 1.3B-480P-quant ✨ | 1.5GB | ~2 sec | 8GB | **Daily use, experimentation** |
| 14B-480P-quant | 10GB | ~5 sec | 16GB | Higher quality at 480p |
| 14B-720P-quant | 10GB | ~10 sec | 24GB | Best quality, HD output |

✨ = Recommended for RTX 5090

## 🎬 Usage Example

After installation:
```powershell
.\turbodiffusion_env\Scripts\Activate.ps1
$env:PYTHONPATH="turbodiffusion"

python turbodiffusion\inference\wan2.1_t2v_infer.py `
    --model Wan2.1-1.3B `
    --dit_path checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth `
    --prompt "A cinematic shot of a knight riding through a misty forest at dawn" `
    --num_steps 4 `
    --quant_linear `
    --attention_type sagesla `
    --save_path output\my_video.mp4
```

**Result**: A 5-second video generated in ~2 seconds! 🎉

## 🆘 Need Help?

### Something not working?
```powershell
.\diagnose.ps1
```
This creates a detailed report of your system.

### Common Issues

| Problem | Solution |
|---------|----------|
| "nvcc not found" | Install CUDA Toolkit 12.8+, restart PowerShell |
| "Unsupported gpu architecture" | **CUDA too old** - See [CUDA_VERSION_ERRORS.md](CUDA_VERSION_ERRORS.md) |
| torch.cuda.is_available() = False | PyTorch-CUDA mismatch - See [CUDA_VERSION_ERRORS.md](CUDA_VERSION_ERRORS.md) |
| "cl.exe not found" | Install Visual Studio 2022 Build Tools |
| Compilation fails | Run `git submodule update --init --recursive` |
| Import turbo_diffusion_ops fails | Runtime mismatch - See [CUDA_VERSION_ERRORS.md](CUDA_VERSION_ERRORS.md) |
| Out of memory | Use 1.3B model, lower resolution to 480p |
| Slow generation | Ensure using `--quant_linear` and `--attention_type sagesla` |

## 📚 Documentation

- **[CUDA_VERSION_ERRORS.md](CUDA_VERSION_ERRORS.md)** - **CRITICAL**: Complete CUDA version mismatch error reference
- **[INSTALLATION_ANALYSIS.md](INSTALLATION_ANALYSIS.md)** - Deep dive into requirements and issues
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete reference guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
- **Research Paper**: https://arxiv.org/pdf/2512.16093

## ⏱️ Time & Space Requirements

| Phase | Time | Space |
|-------|------|-------|
| Visual Studio | 20-30 min | 7GB |
| CUDA Toolkit | 20-40 min | 3.5GB |
| Python venv | 2 min | 500MB |
| Python packages | 10-20 min | 5GB |
| Model download | 30-60 min | 7.5-16GB |
| Compilation | 10-30 min | - |
| **TOTAL** | **1.5-3 hours** | **23-32GB** |

## 🎓 Pro Tips

1. **Use detailed prompts** - The models perform better with long, descriptive prompts
2. **Try different seeds** - Use `--seed` parameter for variations
3. **Experiment with steps** - `--num_steps 2` is faster, `4` is better quality
4. **Monitor GPU memory** - Run `nvidia-smi` in another terminal
5. **Start with 1.3B model** - It's fast and great for testing

## 🔄 Updating TurboDiffusion

```powershell
cd c:\Users\soumi\TurboWan\TurboDiffusion
git pull
git submodule update --init --recursive
.\turbodiffusion_env\Scripts\Activate.ps1
pip install -e . --no-build-isolation
```

## 🚨 Nuclear Options (If Everything Fails)

### Option A: Fresh Start
```powershell
deactivate
Remove-Item -Recurse -Force turbodiffusion_env
python -m venv turbodiffusion_env
# Then re-run installation from Step 3
```

### Option B: WSL2 Linux
If Windows compilation is impossible:
1. Install WSL2: `wsl --install`
2. Install Ubuntu 22.04 from Microsoft Store
3. Follow Linux installation guide in WSL

### Option C: Cloud GPU
- Google Colab (Free GPU)
- RunPod, Vast.ai, Lambda Labs (Paid)

## 📞 Support

- **GitHub Issues**: https://github.com/thu-ml/TurboDiffusion/issues
- **Run diagnostics**: `.\diagnose.ps1` (share the output)
- **Check logs**: Look in `logs\installation_*.log`

---

## 🎯 Ready to Start?

**STEP 1**: Run system check
```powershell
.\setup_scripts\0_system_check.ps1
```

**STEP 2**: Follow the prompts!

The scripts will guide you through everything. Each step takes time, but you'll have an incredibly fast video generation system when you're done!

---

💡 **Tip**: You can run steps 4-8 in the background while you work on other things. The compilation and downloads are the long parts.

Good luck! 🚀🎬
