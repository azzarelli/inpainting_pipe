import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
import os

class SDInpaintingModel:
    """
    Loads SD 1.5 inpainting from a single .ckpt file with a separate VAE .safetensors.

    Checkpoint layout:
        checkpoints/inpainting/sd-v1-5-inpainting.ckpt
        checkpoints/vae/vae-ft-mse-840000-ema-pruned.safetensors
    """

    CKPT_PATH = Path("checkpoints/inpainting")
    VAE_PATH  = Path("checkpoints/vae")

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, cfg=None):
        self.device = torch.device(device)
        self.dtype  = dtype

        self.cfg = cfg
                
        # Load everything via the pipeline converter, then unpack components
        pipeline = self._load_pipeline()

        self.vae          = pipeline.vae
        self.unet         = pipeline.unet
        self.text_encoder = pipeline.text_encoder
        self.tokenizer    = pipeline.tokenizer
        self.scheduler    = pipeline.scheduler

        del pipeline  # free the wrapper, keep the components

        self._freeze_for_training()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> StableDiffusionInpaintPipeline:
        """
        from_single_file handles .ckpt → diffusers component conversion internally.
        We then swap in the finetuned VAE on top.
        """
        ckpt = os.path.join(self.CKPT_PATH, self.cfg["main-name"], ".ckpt")
        
        pipeline = StableDiffusionInpaintPipeline.from_single_file(
            ckpt,
            torch_dtype=self.dtype,
            load_safety_checker=False,
        )
        
        pipeline.vae = self._load_vae()
        pipeline = pipeline.to(self.device)
        return pipeline

    def _load_vae(self) -> AutoencoderKL:
        """
        Load the finetuned MSE VAE from .safetensors.
        from_single_file works for both .ckpt and .safetensors here.
        """
        ckpt = os.path.join(self.VAE_PATH, self.cfg["vae-name"], ".safetensors")
        
        vae = AutoencoderKL.from_single_file(
            ckpt,
            torch_dtype=self.dtype,
        )
        
        return vae

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------

    def _freeze_for_training(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

    def train_mode(self):
        self.unet.train()

    def eval_mode(self):
        self.unet.eval()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.unet.parameters())

    def __repr__(self):
        n_total     = sum(p.numel() for p in self.unet.parameters())
        n_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        return (
            f"SDInpaintingModel(\n"
            f"  ckpt      = {self.CKPT_PATH}\n"
            f"  vae       = {self.VAE_PATH}\n"
            f"  device    = {self.device}\n"
            f"  dtype     = {self.dtype}\n"
            f"  unet      = {n_total:,} params ({n_trainable:,} trainable)\n"
            f")"
        )
        
        
def build_pipeline(state) -> StableDiffusionInpaintPipeline:
    """Reconstruct a pipeline from the loaded model components for inference."""
    model = state["pipeline"]
    pipe = StableDiffusionInpaintPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=model.unet,
        scheduler=model.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    return pipe

