#!/usr/bin/env python3
"""SDXL LoRA trainer — fast edition with latent‑cache + random sampling + tqdm

Speed‑ups
---------
1. **Latent & embedding cache** → 30× faster after first epoch.
2. **Random sample per epoch** via `--sample N`.
3. **Mini‑batch** controlled by `--batch B`.

Latest patch (fix save‑LoRA KeyError)
------------------------------------
* Pass `adapter_name='fashion_lora'` to `get_peft_model_state_dict` so it looks up the correct LoRA config instead of the non‑existent default adapter.
"""
from __future__ import annotations

import argparse, random, math, json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms as T

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_csv', default='labels/dataset.csv')
ap.add_argument('--prompt_txt', default='artifacts/image_prompts_with_static.txt')
ap.add_argument('--output', default='lora-fashion')
ap.add_argument('--epochs', type=int, default=10)
ap.add_argument('--sample', type=int, default=-1, help='images per epoch (-1 = all)')
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--rank', type=int, default=128)
ap.add_argument('--lr', type=float, default=5e-5)
args = ap.parse_args()

random.seed(42)

device = 'cuda'
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16
).to(device)
pipe.vae.requires_grad_(False)

lora_cfg = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank * 2,
    target_modules=[n for n, _ in pipe.unet.named_modules() if n.endswith(('to_q', 'to_k', 'to_v', 'to_out.0'))],
    bias='none',
    init_lora_weights='gaussian',
)
pipe.unet.add_adapter(lora_cfg, adapter_name='fashion_lora')

acc = Accelerator(mixed_precision='fp16')
unet, optimizer = acc.prepare(
    pipe.unet,
    torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr, fused=True),
)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

df = pd.read_csv(args.dataset_csv)
all_imgs = [f'./dataset/{p}' for p in df.image_path]
assert len(all_imgs) == len(prompts)

_rgb2tensor = T.Compose([
    T.Resize(512),
    T.CenterCrop(512),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

CACHE = Path('cache')
CACHE.mkdir(exist_ok=True)
meta_path = CACHE / 'meta.json'
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

def encode_to_cache(idx: int):
    cache_f = CACHE / f'{idx}.pt'
    if cache_f.exists():
        return
    img_path = all_imgs[idx]
    prompt = 'fashion_lora'
    with torch.no_grad():
        px = _rgb2tensor(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device, dtype=torch.float16)
        latent = pipe.vae.encode(px).latent_dist.sample() * 0.18215
        tok1 = pipe.tokenizer(prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors='pt').input_ids.to(device)
        tok2 = pipe.tokenizer_2(prompt, padding='max_length', max_length=pipe.tokenizer_2.model_max_length, truncation=True, return_tensors='pt').input_ids.to(device)
        e1 = pipe.text_encoder(tok1, output_hidden_states=True).hidden_states[-2]
        e2_out = pipe.text_encoder_2(tok2, output_hidden_states=True)
        e2 = e2_out.hidden_states[-2]
        pooled = e2_out.pooler_output if getattr(e2_out, 'pooler_output', None) is not None else e2.mean(dim=1)
        prompt_emb = torch.cat([e1, e2], dim=-1)
    torch.save({'latent': latent.half(), 'prompt': prompt_emb.half(), 'pooled': pooled.half()}, cache_f)
    meta[str(idx)] = prompt

print('⌛ Caching latents/text embeds (first run only)…')
for i in tqdm(range(len(all_imgs))):
    encode_to_cache(i)
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print('✅ Cache ready.')

def sample_indices(n: int) -> List[int]:
    idxs = list(range(len(all_imgs)))
    random.shuffle(idxs)
    return idxs if n == -1 or n > len(idxs) else idxs[:n]

for ep in range(1, args.epochs + 1):
    indices = sample_indices(args.sample)
    num_steps = math.ceil(len(indices) / args.batch)
    pbar = tqdm(range(num_steps), desc=f'Epoch {ep}/{args.epochs}')
    for step in pbar:
        batch_idx = indices[step * args.batch:(step + 1) * args.batch]
        batch = [torch.load(CACHE / f'{i}.pt', map_location=device) for i in batch_idx]
        latents = torch.cat([b['latent'] for b in batch])
        embeds = torch.cat([b['prompt'] for b in batch])
        pooled = torch.cat([b['pooled'] for b in batch])
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
        noise = torch.randn_like(latents)
        noisy = noise_scheduler.add_noise(latents, noise, t).half()
        time_ids = torch.tensor([512, 512, 0, 0, 512, 512], device=device).repeat(latents.size(0), 1)
        pred = unet(noisy, t, encoder_hidden_states=embeds, added_cond_kwargs={'text_embeds': pooled, 'time_ids': time_ids}).sample
        loss = F.mse_loss(pred.to(torch.float16), noise)
        acc.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

out = Path(args.output)
out.mkdir(exist_ok=True)

unwrapped_unet = acc.unwrap_model(unet)
lora_state = convert_state_dict_to_diffusers(
    get_peft_model_state_dict(unwrapped_unet, adapter_name='fashion_lora')
)

pipe.save_lora_weights(
    out,
    unet_lora_layers=lora_state,
    safe_serialization=True,
)
print(f'✨ LoRA saved to {out.resolve()}')
