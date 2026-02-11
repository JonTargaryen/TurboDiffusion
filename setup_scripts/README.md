# TurboDiffusion Setup Scripts

## Overview
This directory contains automated PowerShell scripts to install and configure TurboDiffusion on Windows 11 with RTX 5090.

## Scripts Execution Order

### 0. System Check
```powershell
.\0_system_check.ps1
```
- Verifies all prerequisites
- Checks GPU, Python, CUDA, Visual Studio
- Shows what's missing

### 1. Create Environment
```powershell
.\1_create_environment.ps1
```
- Creates conda environment with Python 3.12
- Environment name: `turbodiffusion`

**After this step**: Activate the environment
```powershell
conda activate turbodiffusion
```

### 2. Install PyTorch
```powershell
.\2_install_pytorch.ps1
```
- Installs PyTorch 2.7.0 with CUDA 12.6
- Verifies CUDA support
- ⏱️ Time: 5-10 minutes

### 3. Install Dependencies
```powershell
.\3_install_dependencies.ps1
```
- Installs Triton, Flash Attention, Einops, etc.
- Total ~30 packages
- ⏱️ Time: 10-20 minutes

### 4. Compile TurboDiffusion
```powershell
.\4_compile_turbodiffusion.ps1
```
- Compiles custom CUDA extensions
- CUTLASS integration
- RTX 5090 optimizations
- ⏱️ Time: 10-30 minutes

### 5. Download Models
```powershell
.\5_download_models.ps1
```
- Downloads VAE and text encoder (required)
- Choose TurboDiffusion model (1.3B recommended)
- 💾 Size: 7.5-16GB
- ⏱️ Time: 30-60 minutes

### 6. Test Installation
```powershell
.\6_test_installation.ps1
```
- Verifies all imports
- Tests CUDA operations
- Optional: Quick video generation test

## Running All Steps

See parent directory's `install.ps1` for master installation script.

## Troubleshooting

If any script fails:
1. Read the error message carefully
2. Run `diagnose.ps1` in parent directory
3. Check logs in `logs\` directory
4. Review error messages and solutions in SETUP_GUIDE.md

## Requirements

- Windows 11
- PowerShell 5.1+
- Administrator privileges (for some installations)

## Notes

- Scripts create logs automatically
- Safe to re-run if they fail
- Environment activation required before steps 2-6
- Internet connection required for downloads
