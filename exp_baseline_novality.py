import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from baseline import baseline

# Show the images using matplotlib, also plot the distances
import matplotlib.pyplot as plt
from PIL import Image


temp = 4  # 0: very cold, 1: cold, 2: mild, 3: warm, 4: hot
gender = 1
dress_code = 3
paths, dists = baseline(temp, gender, dress_code, K=5)

def show_images(image_paths, dists):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    average_dist = np.mean(dists)
    diff_top_avg = np.abs(dists[0] - average_dist)
    fig.suptitle(f'Top-5 similar images ({temp}, {gender}, {dress_code}) (Avg. Dist: {average_dist:.4f}, Top diff: {diff_top_avg:.4f})', fontsize=16)

    for ax, img_path, dist in zip(axes, image_paths, dists):
        img = Image.open('./dataset/'+img_path).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Dist: {dist:.4f}')

    plt.show()
# Show the top-5 similar images
show_images(paths, dists)