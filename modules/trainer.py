import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_cosine_schedule_with_warmup

import json

class InpaintingDataset(Dataset):
    """
    Loads from data/data.json + data/cropped/ images.
    Mask is generated from the AABB boxes in the json.
    """

    def __init__(self, data_dir: str, size: int = 512, cfg=None):
        self.cfg = cfg
        
        self.size = size
        self.data_dir = Path(data_dir)
        self.size = size
        

        json_path = self.data_dir / "data.json"
        with open(json_path) as f:
            self.samples = json.load(f)

        print(f"[dataset] found {len(self.samples)} samples in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        img_path = self.data_dir / entry["image"]
        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Build mask from boxes at original resolution, then resize
        mask = Image.new("L", (orig_w, orig_h), 0)
        mask_np = np.array(mask)
        for box in entry["box"]:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            mask_np[y:y+h, x:x+w] = 255
        mask = Image.fromarray(mask_np)

        # Resize both to model input size
        image = image.resize((self.size, self.size), Image.LANCZOS)
        mask  = mask.resize((self.size, self.size), Image.NEAREST)

        trigger   = self.cfg["trigger"]
        category  = self.cfg["category-info"]
        quality   = self.cfg["quality-info"]
        
        detail    = entry["text"].replace(" and ", ", ")
        prompt    = f"{trigger}, {detail}, {category}, {quality}"

        image_t = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        mask_t  = torch.tensor(np.array(mask),  dtype=torch.float32).unsqueeze(0) / 255.0
        mask_t  = (mask_t > 0.5).float()
        masked_image_t = image_t * (1 - mask_t)

        return {
            "image":        image_t,
            "mask":         mask_t,
            "masked_image": masked_image_t,
            "prompt":       prompt,
        }


class SDInpaintingTrainer:

    def __init__(self, model, output_dir: str = "./lora_output", rank: int = 4, cfg=None):
        self.cfg = cfg

        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            lora_dropout=0.1,
        )
        self.model.unet = get_peft_model(self.model.unet, lora_config)
        self.model.unet.print_trainable_parameters()

    def _encode_prompt(self, prompts: list) -> torch.Tensor:
        tokens = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.model.device)
        with torch.no_grad():
            return self.model.text_encoder(tokens)[0]

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image = image.to(device=self.model.device, dtype=self.model.dtype)
            dist = self.model.vae.encode(image).latent_dist
            return dist.mode() * self.model.vae.config.scaling_factor

    def train(
        self,
        data_dir:      str,
        epochs:        int   = 10,
        batch_size:    int   = 1,
        lr:            float = 2e-4,
        save_every:    int   = 1,
        loss_callback  = None,   # fn(epoch, step, total_steps, loss)
        stop_flag      = None,   # fn() → bool, True = stop
        grad_accum:    int   = 2,   # effective batch = batch_size * grad_accum
    ):
        dataset    = InpaintingDataset(data_dir, cfg=self.cfg['dataset'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.unet.parameters()),
            lr=lr,
            weight_decay=1e-2,
        )

        total_steps   = epochs * len(dataloader)
        warmup_steps  = max(50, total_steps // 10)  # 10% warmup, minimum 50
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler()

        self.model.train_mode()
        optimizer.zero_grad()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for step, batch in enumerate(dataloader, 1):
                if stop_flag and stop_flag():
                    print("[trainer] stopped by user")
                    self.model.eval_mode()
                    return

                image_latents        = self._encode_image(batch["image"])
                masked_image_latents = self._encode_image(batch["masked_image"])

                mask_latents = F.interpolate(
                    batch["mask"].to(self.model.device),
                    size=image_latents.shape[-2:],
                )

                noise     = torch.randn_like(image_latents)
                timesteps = torch.randint(
                    0, 700,
                    (image_latents.shape[0],),
                    device=self.model.device,
                ).long()

                noisy_latents = self.model.scheduler.add_noise(image_latents, noise, timesteps)

                unet_input            = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)
                encoder_hidden_states = self._encode_prompt(batch["prompt"])

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16):
                    noise_pred = self.model.unet(
                        unet_input.to(torch.float16),
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states.to(torch.float16),
                    ).sample
                    loss = F.mse_loss(noise_pred, noise.to(torch.float16)) / grad_accum

                scaler.scale(loss).backward()

                if step % grad_accum == 0 or step == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.unet.parameters()),
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    self.lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * grad_accum  # unscale for logging

                if loss_callback:
                    loss_callback(epoch, step, len(dataloader), loss.item() * grad_accum)

            avg_loss = total_loss / len(dataloader)
            print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}")

            if epoch % save_every == 0:
                self._save_lora(epoch)

        self.model.eval_mode()

    def _save_lora(self, epoch: int):
        out = self.output_dir / f"lora_epoch{epoch}"
        self.model.unet.save_pretrained(str(out))
        print(f"[saved] LoRA → {out}")