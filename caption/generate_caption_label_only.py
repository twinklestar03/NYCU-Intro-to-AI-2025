import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load dataset and embeddings
df = pd.read_csv('labels/dataset.csv')
image_paths = df['image_path'].apply(lambda p: f'./dataset/{p}' if not p.startswith('./dataset/') else p).tolist()
final_emb = np.load('artifacts/final_emb.npy')
shape_attrs = df[[f's{i}' for i in range(12)]].to_numpy()
texture_attrs = df[['t0', 't1', 't2']].to_numpy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CaptionDataset(Dataset):
    SHAPE_MAPPING = [
        ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "NA"],
        ["three-point shorts", "medium short", "three-quarter", "long pants", "NA"],
        ["no socks", "socks", "leggings", "NA"],
        ["no hat", "wearing hat", "NA"],
        ["no glasses", "eyeglasses", "sunglasses", "glasses in hand or on clothes", "NA"],
        ["no neckwear", "wearing neckwear", "NA"],
        ["no wristwear", "wearing wristwear", "NA"],
        ["no ring", "wearing ring", "NA"],
        ["no waist accessories", "belt", "clothing accessory at waist", "hidden", "NA"],
        ["V-neck", "square neck", "round neck", "standing collar", "lapel", "suspenders", "NA"],
        ["wearing cardigan", "not wearing cardigan", "NA"],
        ["upper does not cover navel", "upper covers navel", "NA"]
    ]

    TEXTURE_MAPPING = [
        "denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", "NA"
    ]
    TEXTURE_POS = ['upper fabric', 'lower fabric', 'outer fabric']

    def __init__(self, image_paths, embeddings, shape_attrs, texture_attrs):
        self.image_paths = image_paths
        self.embeddings = embeddings
        self.shape_attrs = shape_attrs
        self.texture_attrs = texture_attrs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float)
        shapes = self.shape_attrs[idx]
        descs = []
        for i, code in enumerate(shapes):
            if not np.isnan(code) and 0 <= int(code) < len(self.SHAPE_MAPPING[i]):
                desc = self.SHAPE_MAPPING[i][int(code)]
                if desc != "NA":
                    descs.append(desc)
        caption = ", ".join(descs)

        textures = self.texture_attrs[idx]
        texture_descs = []
        for i, code in enumerate(textures):
            if not np.isnan(code) and 0 <= int(code) < len(self.TEXTURE_MAPPING):
                desc = self.TEXTURE_MAPPING[int(code)]
                pos_desc = self.TEXTURE_POS[i] if i < len(self.TEXTURE_POS) else "unknown"
                if desc != "NA":
                    texture_descs.append(f'{pos_desc} is {desc}')
        caption += ". " + ", ".join(texture_descs) if texture_descs else caption

        print(caption)
        return emb, caption

# Example: DataLoader check
dataset = CaptionDataset(image_paths, final_emb, shape_attrs, texture_attrs)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for emb_batch, cap_batch in loader:
    # print(cap_batch)
    break