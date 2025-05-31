import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from models.weather_2_style import Weather2StyleMLP, train_model
import math


def predict_and_retrieve(_temp: float, _gender: int, _dress_code: int, K: int = 5,
                         embedding_path:str ='artifacts/final_emb_gender_onehot.npy', 
                         dataset_path: str = 'labels/dataset.csv',
                         model_path: str = 'artifacts/weather2style.pth'):
    preloaded = train_model(embedding_path, dataset_path, model_path)
    df = preloaded['df']
    scaler_temp = preloaded['scaler_temp']
    oh = preloaded['oh_encoder']
    model = preloaded['model']
    Y = preloaded['Y']
    device = next(model.parameters()).device
    model.eval()

    # Prepare new input sample
    temp_arr = np.array([[_temp]])
    gender_arr = np.array([[_gender]])
    dress_arr = np.array([[_dress_code]])
    t_feat = scaler_temp.transform(temp_arr)
    cat_feat = oh.transform(np.concatenate([gender_arr, dress_arr], axis=1))
    x_new = np.concatenate([t_feat, cat_feat], axis=1)

    # Predict style embedding
    with torch.no_grad():
        style_emb_new = model(torch.from_numpy(x_new).float().to(device)).cpu().numpy()

    # K-NN find top-K similar samples
    nn_model = NearestNeighbors(n_neighbors=K).fit(Y)
    dists, idxs = nn_model.kneighbors(style_emb_new)
    similar_keys = df['image_path'].iloc[idxs[0]].tolist()
    print(f'Top-{K} similar samples keys:', similar_keys, dists)

    return similar_keys, dists[0]


def baseline(_temp, _gender, _dress_code, K=5):
    return predict_and_retrieve(_temp, _gender, _dress_code, K)
