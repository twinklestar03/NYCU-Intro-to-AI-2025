# test_prompt_gen.py
import numpy as np, torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from models.embed_2_prompt import Embed2Prompt, Embed2PromptModel  

base_model  = 'gpt2'
tokenizer   = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token   
lm          = GPT2LMHeadModel.from_pretrained(base_model).eval()

ckpt = torch.load('checkpoints/embed2prompt/bridge.pt', map_location='cpu')
bridge = Embed2Prompt(**ckpt['config'])
bridge.load_state_dict(ckpt['state_dict'])
model  = Embed2PromptModel(lm, bridge, tokenizer).eval()

vec = torch.tensor(
        np.load('artifacts/final_emb_gender_onehot.npy')[0],
        dtype=torch.float32 
      ).unsqueeze(0)                                        # (1, D)

with torch.no_grad():
    prefix = bridge(vec)                         # (1, prefix_len, hidden)
    attn   = torch.ones(prefix.size()[:-1], dtype=torch.long)  # (1, prefix_len)

    out = model.lm.generate(
        inputs_embeds   = prefix,
        attention_mask  = attn,
        max_new_tokens  = 60,     
        do_sample       = True,      
        top_p           = 0.9,
        temperature     = 0.7,
        pad_token_id    = tokenizer.pad_token_id,
        eos_token_id    = tokenizer.eos_token_id,
    )

print('🖼️ prompt:', tokenizer.decode(out[0], skip_special_tokens=True))
