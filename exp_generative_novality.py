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
from image_distances import compute_lpips

# temp = 4  # 0: very cold, 1: cold, 2: mild, 3: warm, 4: hot
# gender = 1
# dress_code = 3
# paths, dists = baseline(temp, gender, dress_code, K=1)
temp = 0  # 0: very cold, 1: cold, 2: mild, 3: warm, 4: hot
gender = 0
dress_code = 4
paths, dists = baseline(temp, gender, dress_code, K=1)

baseline_imagepath = './dataset/' + paths[0]

# GENERATED_IMAGES = [
#     'result-4-1-3_0.png',
#     'result-4-1-3_1.png',
#     'result-4-1-3_2.png',
#     'result-4-1-3_3.png',
#     'result-4-1-3_4.png'
# ]
GENERATED_IMAGES = [
    'result-0-0-4_0.png',
    'result-0-0-4_1.png',
    'result-0-0-4_2.png',
    'result-0-0-4_3.png',
    'result-0-0-4_4.png'
]

baseline_img = Image.open(baseline_imagepath).convert('RGB')
dists = []
for gen in GENERATED_IMAGES:
    img2 = Image.open(gen).convert('RGB')

    dists.append(compute_lpips(baseline_img, img2))

# show generated images and distances
def show_images(image_paths, dists):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    average_dist = np.mean(dists)
    diff_top_avg = np.abs(dists[0] - average_dist)
    fig.suptitle(f'Generated images (Avg. Dist: {average_dist:.4f}, Top diff: {diff_top_avg:.4f})', fontsize=16)

    # also show baseline image
    axes[0].imshow(baseline_img)
    axes[0].axis('off')
    axes[0].set_title(f'Baseline Image')
    for ax, img_path, dist in zip(axes[1:], image_paths, dists):
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Dist: {dist:.4f}')

    plt.show()

# Show the generated images
show_images(GENERATED_IMAGES, dists)


