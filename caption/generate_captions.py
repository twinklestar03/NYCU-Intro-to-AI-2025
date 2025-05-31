import os
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

class ImageCaptionDataset(Dataset):
    def __init__(self, df, img_col):
        self.paths = df[img_col].tolist()
        self.df = df
        self.shape_attrs = df[[f's{i}' for i in range(12)]].to_numpy()
        self.texture_attrs = df[['t0', 't1', 't2']].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open('./dataset/' + self.paths[idx]).convert('RGB')
        shape_descs = []
        for i, code in enumerate(self.shape_attrs[idx]):
            if not np.isnan(code):
                code = int(code)
                # guard index out of range
                if code < len(StaticMaps.SHAPE_MAPPING[i]):
                    desc = StaticMaps.SHAPE_MAPPING[i][code]
                else:
                    desc = 'NA'
                if desc != 'NA': shape_descs.append(desc)
        texture_descs = []
        for i, code in enumerate(self.texture_attrs[idx]):
            if not np.isnan(code):
                code = int(code)
                tex = StaticMaps.TEXTURE_MAPPING[code]
                pos = StaticMaps.TEXTURE_POS[i] if i < len(StaticMaps.TEXTURE_POS) else f'fabric{i}'
                if tex != 'NA': texture_descs.append(f'{pos} is {tex}')

        all_descs = []
        all_descs.append(StaticMaps.DRESSCODE_MAP[self.df['dresscode'][idx]])
        all_descs.append(StaticMaps.GENDER_MAP[self.df['sex'][idx]])
        all_descs += shape_descs + texture_descs

        random.shuffle(all_descs)
        static_caption = ', '.join(all_descs)
        return img, idx, static_caption

# 靜態映射表
class StaticMaps:
    SHAPE_MAPPING = [
        ['sleeveless','short-sleeve','medium-sleeve','long-sleeve','not long-sleeve','NA'],
        ['three-point shorts','medium short','three-quarter','long pants','NA'],
        ['NA','socks','leggings','NA'],
        ['NA','wearing hat','NA'],
        ['NA','wearing eyeglasses','wearing sunglasses','glasses in hand or on clothes','NA'],
        ['NA','wearing neckwear','NA'],
        ['NA','wearing wristwear','NA'],
        ['NA','wearing ring','NA'],
        ['NA','wearing belt','clothing accessory at waist','NA','NA'],
        ['neckline V-neck','neckline square neck','neckline round neck','neckline standing collar','neckline lapel','neckline suspenders','NA'],
        ['wearing cardigan','NA','NA'],
        ['upper cloth does not cover navel','upper covers navel','NA']
    ]
    TEXTURE_MAPPING = ['denim','cotton','leather','furry','knitted','chiffon','other','NA']
    TEXTURE_POS = ['upper fabric','lower fabric','outer fabric']

    GENDER_MAP = ['male', 'female']
    DRESSCODE_MAP = ['business_formal', 'business_casual', 'smart_casual', 'sport', 'vacation', 'urban']

def collate_fn(batch):
    images, indices, statics = zip(*batch)
    return list(images), torch.tensor(indices, dtype=torch.long), list(statics)


def main():
    df = pd.read_csv('labels/dataset.csv')
    img_col = 'image_path'
    model_name = 'nlpconnect/vit-gpt2-image-captioning'

    # Dataset & DataLoader
    dataset = ImageCaptionDataset(df, img_col)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Initialization
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    model.config.use_cache = False

    prompts = [''] * len(dataset)

    for images, indices, statics in loader:
        pixel_values = feature_extractor(images=images, return_tensors='pt').pixel_values.to(device)
        outputs = model.generate(pixel_values=pixel_values, max_length=30, num_beams=5, early_stopping=True, use_cache=False)
        gen_caps = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for idx, static_cap, gen_cap in zip(indices.tolist(), statics, gen_caps):
            if static_cap:
                full = f'{gen_cap}, {static_cap}'
            else:
                full = gen_cap
            prompts[idx] = full

    # 存檔
    os.makedirs('artifacts', exist_ok=True)
    out_path = os.path.join('artifacts', 'image_prompts_with_static_1.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(p + '\n')
    print(f'Generated captions saved to {out_path}')

if __name__ == '__main__':
    main()
