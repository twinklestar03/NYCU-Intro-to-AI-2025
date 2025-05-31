import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from models.weather_2_style import Weather2StyleMLP, train_model


TEMP_MAP   = {0: 'very cold', 1: 'cold', 2: 'mild', 3: 'warm', 4: 'hot'}
GENDER_MAP = {0: 'male', 1: 'female'}
CAT_COLS   = [f's{i}' for i in range(12)] + ['t0', 't1', 't2', 'sex', 'dresscode']

ART_DIR = Path('artifacts')
SBERT_NAME = 'all-MiniLM-L6-v2'


def fit_pipeline(csv_path: str | Path, out_dir: str | Path = ART_DIR):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f'[fit] Loaded dataset: {len(df)} rows')

    # 1) SBERT text embedding
    sbert = SentenceTransformer(SBERT_NAME)
    texts = (df['dresscode'].astype(str) + ' | ' +
             df['sex'].map(GENDER_MAP) + ' | ' +
             df['temp'].map(TEMP_MAP))
    text_emb = sbert.encode(texts.tolist(), batch_size=32)          # (N, 384)

    # 2) One‑Hot categorical
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_emb = ohe.fit_transform(df[CAT_COLS])                       # (N, d_cat)

    # 3) Normalised temperature
    scaler = MinMaxScaler()
    num_emb = scaler.fit_transform(df[['temp']])                   # (N, 1)

    final_emb = np.concatenate([text_emb, cat_emb, num_emb], axis=1)
    np.save(out_dir / 'final_emb.npy', final_emb)
    joblib.dump(ohe,     out_dir / 'ohe.joblib')
    joblib.dump(scaler,  out_dir / 'scaler.joblib')

    print('✔ Pipeline fitted & saved:')
    print('  text_emb :', text_emb.shape)
    print('  cat_emb  :', cat_emb.shape)
    print('  num_emb  :', num_emb.shape)
    print('→ final_emb:', final_emb.shape)


def load_components(dir_: str | Path = ART_DIR):
    dir_ = Path(dir_)
    sbert   = SentenceTransformer(SBERT_NAME)
    ohe     = joblib.load(dir_ / 'ohe.joblib')
    scaler  = joblib.load(dir_ / 'scaler.joblib')
    return sbert, ohe, scaler


_DEF_SHAPES  = [0]*12
_DEF_TEXTURE = [0,0,0]

def predict_embedding(*,
                      gender: int,
                      temperature: int,
                      dresscode: int,
                      shapes: List[int] | None = None,
                      textures: List[int] | None = None,
                      comp = None,
                      embedding_path:str ='artifacts/final_emb_gender_onehot.npy', 
                      dataset_path: str = 'labels/dataset.csv',
                      model_path: str = 'artifacts/weather2style.pth')  -> np.ndarray:
    '''Return (d_total,) embedding using *exactly* the same formula as training.'''
    preloaded = train_model(embedding_path, dataset_path, model_path)
    df = preloaded['df']
    scaler_temp = preloaded['scaler_temp']
    oh = preloaded['oh_encoder']
    model = preloaded['model']
    Y = preloaded['Y']
    device = next(model.parameters()).device
    model.eval()

    # Prepare new input sample
    temp_arr = np.array([[temperature]])
    gender_arr = np.array([[gender]])
    dress_arr = np.array([[dresscode]])
    t_feat = scaler_temp.transform(temp_arr)
    cat_feat = oh.transform(np.concatenate([gender_arr, dress_arr], axis=1))
    x_new = np.concatenate([t_feat, cat_feat], axis=1)

    # Predict style embedding
    with torch.no_grad():
        style_emb_new = model(torch.from_numpy(x_new).float().to(device)).cpu().numpy()

    return style_emb_new


if __name__ == '__main__':
    p = argparse.ArgumentParser('Deterministic embedding predictor')
    sub = p.add_subparsers(dest='cmd', required=True)

    fit_p = sub.add_parser('fit')
    fit_p.add_argument('--csv', required=True)
    fit_p.add_argument('--out-dir', default='artifacts')

    pred_p = sub.add_parser('predict')
    pred_p.add_argument('--gender', type=int, choices=[0,1], required=True)
    pred_p.add_argument('--temperature', type=int, choices=range(5), required=True)
    pred_p.add_argument('--dresscode', type=int, required=True)

    args = p.parse_args()

    if args.cmd == 'fit':
        fit_pipeline(args.csv, args.out_dir)
    else:
        emb = predict_embedding(gender=args.gender,
                                 temperature=args.temperature,
                                 dresscode=args.dresscode)
        print(json.dumps(emb.tolist()[:10]), '...', f'dim={emb.shape[0]}')
