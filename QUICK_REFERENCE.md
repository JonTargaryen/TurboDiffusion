# TurboDiffusion Quick Reference Card

## 🚦 Installation Status Checklist

```
Prerequisites:
[ ] Visual Studio 2022 Build Tools (C++ workload)
[ ] CUDA Toolkit 12.6 or later  
[ ] Python 3.9+ (already installed: 3.12.10)
[ ] Restart PowerShell after installations

Installation Steps:
[ ] 0. System check: .\setup_scripts\0_system_check.ps1
[ ] 1. Create env: .\setup_scripts\1_create_environment.ps1
[ ] 2. Activate: .\turbodiffusion_env\Scripts\Activate.ps1
[ ] 3. PyTorch: .\setup_scripts\2_install_pytorch.ps1
[ ] 4. Dependencies: .\setup_scripts\3_install_dependencies.ps1
[ ] 5. Compile: .\setup_scripts\4_compile_turbodiffusion.ps1
[ ] 6. Models: .\setup_scripts\5_download_models.ps1
[ ] 7. Test: .\setup_scripts\6_test_installation.ps1
```

## 📝 Essential Commands

### Environment
```powershell
# Activate (REQUIRED before running)
.\turbodiffusion_env\Scripts\Activate.ps1

# Deactivate
deactivate

# Remove and recreate
deactivate
Remove-Item -Recurse -Force turbodiffusion_env
python -m venv turbodiffusion_env
```

### Generation (1.3B Model - Fastest)
```powershell
$env:PYTHONPATH="turbodiffusion"

python turbodiffusion\inference\wan2.1_t2v_infer.py `
    --model Wan2.1-1.3B `
    --dit_path checkpoints\TurboWan2.1-T2V-1.3B-480P-quant.pth `
    --prompt "YOUR PROMPT HERE" `
    --num_steps 4 `
    --quant_linear `
    --attention_type sagesla `
    --sla_topk 0.1 `
    --save_path output\video.mp4
```

### Generation (14B Model - Best Quality)
```powershell
python turbodiffusion\inference\wan2.1_t2v_infer.py `
    --model Wan2.1-14B `
    --dit_path checkpoints\TurboWan2.1-T2V-14B-720P-quant.pth `
    --resolution 720p `
    --prompt "YOUR PROMPT HERE" `
    --num_steps 4 `
    --quant_linear `
    --attention_type sagesla `
    --sla_topk 0.15 `
    --save_path output\video.mp4
```

### Interactive Mode
```powershell
python turbodiffusion\inference\wan2.1_t2v_infer.py --serve
```

## 🔧 Troubleshooting

### Check GPU Status
```powershell
nvidia-smi
```

### Run Diagnostics
```powershell
.\diagnose.ps1
```

### Verify CUDA Operations
```powershell
python -c "import torch; print(torch.cuda.is_available())"
python -c "import turbo_diffusion_ops; print('OK')"
```

### Clean and Recompile
```powershell
.\turbodiffusion_env\Scripts\Activate.ps1
Remove-Item -Recurse -Force build, dist, *.egg-info
pip install -e . --no-build-isolation
```

## 🎯 Common Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `--model` | `Wan2.1-1.3B`, `Wan2.1-14B` | Model size |
| `--num_steps` | `1-4` | Higher = better quality |
| `--resolution` | `480p`, `720p` | Output resolution |
| `--num_frames` | `33`, `81` (default) | Video length |
| `--aspect_ratio` | `16:9`, `9:16`, `4:3` | Video aspect ratio |
| `--sla_topk` | `0.05-0.2` | Lower=faster, Higher=better |
| `--seed` | `0-999999` | For reproducibility |

## 🚨 Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Import errors | `.\turbodiffusion_env\Scripts\Activate.ps1` |
| CUDA not found | Restart PowerShell |
| OOM errors | Use 1.3B model, 480p resolution |
| Slow generation | Add `--quant_linear --attention_type sagesla` |
| Compilation fails | `git submodule update --init --recursive` |

## 📂 Important Paths

```
c:\Users\soumi\TurboWan\TurboDiffusion\
├── setup_scripts\         # Installation scripts
├── checkpoints\           # Models (after download)
├── output\                # Generated videos
├── logs\                  # Installation logs
├── START_HERE.md          # Start here!
├── SETUP_GUIDE.md         # Complete guide
└── diagnose.ps1           # Diagnostic tool
```

## 💾 Model Files

Required (6GB):
- `checkpoints\Wan2.1_VAE.pth`
- `checkpoints\models_t5_umt5-xxl-enc-bf16.pth`

Model Checkpoints (choose one or more):
- `TurboWan2.1-T2V-1.3B-480P-quant.pth` (1.5GB) ⭐ Recommended
- `TurboWan2.1-T2V-14B-480P-quant.pth` (10GB)
- `TurboWan2.1-T2V-14B-720P-quant.pth` (10GB)

## 🎬 Prompt Tips

✅ Good Prompts (Long and detailed):
```
"A wide-angle cinematic shot of a majestic eagle soaring over 
snow-capped mountains at golden hour, with dramatic clouds and 
sunbeams breaking through, camera slowly panning and following 
the eagle's graceful flight"
```

❌ Bad Prompts (Too short):
```
"eagle flying"
```

## ⚡ Performance Targets

| Model | Resolution | Expected Time (RTX 5090) |
|-------|-----------|--------------------------|
| 1.3B | 480p | 2-3 seconds |
| 14B | 480p | 5-8 seconds |
| 14B | 720p | 10-15 seconds |

## 📞 Getting Help

1. Run diagnostics: `.\diagnose.ps1`
2. Check logs: `logs\installation_*.log`
3. Read guides: `SETUP_GUIDE.md`, `INSTALLATION_ANALYSIS.md`
4. GitHub Issues: https://github.com/thu-ml/TurboDiffusion/issues

## 🔄 Updating

```powershell
cd c:\Users\soumi\TurboWan\TurboDiffusion
git pull
git submodule update --init --recursive
.\turbodiffusion_env\Scripts\Activate.ps1
pip install -e . --no-build-isolation
```

---

**Keep this file open during installation and usage!** 📌
