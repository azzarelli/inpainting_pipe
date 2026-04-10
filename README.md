# (Local) GUI for SD 1.5 - Inpainting
Basic drawing functionality and +/- prompt inputs for mask-based inpainting.

## Models
- SD v1.5 - Inpainting
- VAE `vae-ft-mse-840000-ema-pruned`

# Installation: PyTorch 2.6 with CUDA 12.4
```
conda create -n sd-inpaint python=3.11
conda activate sd-inpaint

pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate peft omegaconf safetensors Pillow numpy tqdm dearpygui
```

# Folder Management
- Add `checkpoint`/`outputs` folders
- Add a folder for `vae` and `inpainting` in the checkpoints and place the checkpoint there
- Update paths in `modules/inpainting.py` if needed