from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import List, Optional

import torch, numpy as np
from PIL import Image
from diffusers import (StableDiffusionXLPipeline, LCMScheduler,
                       ControlNetModel, StableDiffusionXLControlNetPipeline)
from transformers import AutoTokenizer, GPT2LMHeadModel


def build_sdxl_pipeline(device: str = 'cuda', *, lora: Optional[str] = None,
                         controlnet: Optional[str] = None) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        variant='fp16' if device.startswith('cuda') else None).to(device)

    # Replace scheduler with LCM for fast 4‑step generation
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Optional LoRA
    if lora:
        pipe.load_lora_weights(lora, adapter_name='fashion_lora')
        pipe.fuse_lora()

    # Optional ControlNet (currently supports openpose or seg)
    # if controlnet:
    #     model_id = {
    #         'openpose': 'diffusers/controlnet-openpose-sdxl-1.0',
    #         'seg':      'diffusers/controlnet-seg-sdxl-1.0',
    #     }[controlnet]
    #     cn = ControlNetModel.from_pretrained(model_id,
    #                                          torch_dtype=pipe.dtype).to(device)
    #     pipe = StableDiffusionXLControlNetPipeline(**pipe.components, controlnet=cn).to(device)

    return pipe


def generate_outfit(pipe, prompt: str, *, negative: str = '', steps: int = 20,
                     seed: int | None = None, pose_image: Optional[str] = None,
                     height: int = 1024, width: int = 1024) -> Image.Image:
    g = None if seed is None else torch.Generator(pipe.device).manual_seed(seed)

    kwargs = {
        'prompt': prompt,
        'num_inference_steps': steps,
        'guidance_scale': 7.0,
        'negative_prompt': negative,
        'generator': g,
    }
    if pose_image is not None and hasattr(pipe, 'controlnet'):
        import cv2, numpy as np
        img = cv2.imread(pose_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('Cannot read pose image')
        kwargs['controlnet_conditioning_image'] = np.array(img)[None, ...]

    return pipe(**kwargs).images[0]

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Test image generator')
    ap.add_argument('--out', default='outfit.png')
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = build_sdxl_pipeline(device)
    img = generate_outfit(pipe, 'a man standing in front of a wall, full-body, long-sleeve, neckline round neckline round neck, upper covers navel, upper covers navel, lower fabric is denim',
                          negative='ugly, blurry, bad anatomy, bad hands, deformed')
    img.save(args.out)
    print(f'Image saved to {args.out}')
