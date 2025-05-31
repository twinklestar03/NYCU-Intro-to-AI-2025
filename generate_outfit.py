import argparse, json, sys
from pathlib import Path
from typing import List, Optional

import torch, numpy as np
from PIL import Image
from diffusers import (StableDiffusionXLPipeline, LCMScheduler, DiffusionPipeline,
                       ControlNetModel, StableDiffusionXLControlNetPipeline)
from transformers import AutoTokenizer, GPT2LMHeadModel
from embedding_predictor import predict_embedding
from models.embed_2_prompt import Embed2Prompt, Embed2PromptModel 

ART_DIR    = Path('artifacts')
BRIDGE_PT  = 'checkpoints/embed2prompt_2/bridge.pt' 
BASE_MODEL = 'gpt2'
def load_prompt_generator(device: str = 'cuda'):
    ck = torch.load(BRIDGE_PT, map_location=device)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    lm  = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
    bridge = Embed2Prompt(ck['config']['dim_in'], ck['config']['dim_out'],
                          ck['config']['prefix_len'], ck['config'].get('dropout',0.0)).to(device)
    bridge.load_state_dict(ck['state_dict'])
    model = Embed2PromptModel(lm, bridge, tok).eval()

    def gen_prompt(emb: np.ndarray, max_len: int = 64) -> str:
        with torch.no_grad():
            vec = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            prefix = bridge(vec)
            attn   = torch.ones(prefix.size()[:-1], dtype=torch.long, device=device)
            ids = lm.generate(inputs_embeds=prefix,
                              attention_mask=attn,
                              max_new_tokens=max_len,
                              do_sample=True, top_p=0.9, temperature=0.8)
            return tok.decode(ids[0], skip_special_tokens=True)
    return gen_prompt


def build_sdxl_pipelines(
    device: str = 'cuda',
    *,
    lora: Optional[str] = None,
    controlnet: Optional[str] = None,
    use_lcm: bool = True,
):
    base = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        variant='fp16' if device.startswith('cuda') else None,
        use_safetensors=True,
    ).to(device)

    if use_lcm:                                  # 4-step LCM (可選)
        base.scheduler = LCMScheduler.from_config(base.scheduler.config)

    if lora:                                     # LoRA 只掛在 base 端
        base.load_lora_weights(lora, adapter_name='fashion_lora')
        base.fuse_lora()

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    return base, refiner


def generate_outfit(
    base, refiner,
    prompt: str,
    *,
    negative: str = '',
    steps: int = 30,              
    high_noise_frac: float = 0.8,    
    seed: int | None = None,
    pose_image: Optional[str] = None,
):
    g = None if seed is None else torch.Generator(base.device).manual_seed(seed)

    base_kwargs = dict(
        prompt           = prompt,
        negative_prompt  = negative,
        num_inference_steps = steps,
        denoising_end    = high_noise_frac,
        output_type      = 'latent',   
        generator        = g,
    )
    if pose_image and hasattr(base, 'controlnet'):

        img = cv2.imread(pose_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('Cannot read pose image')
        base_kwargs['controlnet_conditioning_image'] = np.array(img)[None, ...]

    latents = base(**base_kwargs).images 

    image = refiner(
        prompt          = prompt,
        negative_prompt = negative,
        num_inference_steps = 8,
        denoising_start = high_noise_frac,
        image           = latents,
        generator       = g,
    ).images[0]

    return image


# 3. main ────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser('Outfit image generator')
    ap.add_argument('--gender', type=int, choices=[0,1], required=True)
    ap.add_argument('--temperature', type=int, choices=range(5), required=True)
    ap.add_argument('--dresscode', type=int, required=True)
    ap.add_argument('--out', default='outfit.png')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--lora', default=None, help='Path to LoRA weights (optional)')
    ap.add_argument('--controlnet', choices=['openpose','seg'])
    ap.add_argument('--pose-img', help='Pose/seg image path for ControlNet')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1. embedding to prompt
    emb2prompt = load_prompt_generator(device)
    emb = predict_embedding(gender=args.gender,
                            temperature=args.temperature,
                            dresscode=args.dresscode)

    DRESSCODE = ['business_formal', 'business_casual', 'smart_casual', 'sport', 'vacation', 'urban']
    TEMPERATURE = ['cold', 'cool', 'neutral', 'warm', 'hot']
    prompt = (
        'ultra-realistic,4k,fasion,single person,outfit,full-body,(standing:1.5),' + DRESSCODE[args.dresscode] +',' + TEMPERATURE[args.temperature] + ' weather,'
        + emb2prompt(emb)
        + ',best quality'
    )
    prompt = ','.join(list(set(prompt.split(','))))
    print('[Prompt]', prompt)

    # 2. SDXL Pipelines
    base, refiner = build_sdxl_pipelines(
        device,
        lora      = args.lora,
        controlnet = args.controlnet,
        use_lcm   = True,   
    )

    # 3. Generate outfit image
    img = generate_outfit(
        base, refiner,
        prompt,
        seed        = args.seed,
        pose_image  = args.pose_img,
        negative    = 'ugly, blurry, bad anatomy, bad hands, deformed',
        steps       = 20,
        high_noise_frac = 0.8,
    )
    img.save(args.out)
    print(f'Image saved to {args.out}')
