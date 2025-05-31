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


# m_images = [
#     'result-0-0-4_0.png',
#     'result-0-0-4_1.png',
#     'result-0-0-4_2.png',
#     'result-0-0-4_3.png',
#     'result-0-0-4_4.png'
# ]
m_images = [
    'result-4-1-3_0.png',
    'result-4-1-3_1.png',
    'result-4-1-3_2.png',
    'result-4-1-3_3.png',
    'result-4-1-3_4.png'
]

print("Sampled image paths:", m_images)


# Compute Pairwise LPIPS distances and store pairwise distances in a matrix
M = len(m_images)
dist_matrix = np.zeros((M, M), dtype=np.float32)
dist_list = []

for i in range(M):
    for j in range(i+1, M):
        img1 = Image.open(m_images[i]).convert('RGB')
        img2 = Image.open(m_images[j]).convert('RGB')
        dist = compute_lpips(img1, img2)
        print(f'Distance between {m_images[i]} and {m_images[j]}: {dist:.4f}')

        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
        dist_list.append((m_images[i], m_images[j], dist))


fig, axes = plt.subplots(2, M, figsize=(3*M, 6))
for idx, img_name in enumerate(m_images):
    ax = axes[0, idx]
    img = Image.open(img_name).convert('RGB')
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'#{idx}', fontsize=10)

ax = axes[1, 0]
im = ax.imshow(dist_matrix, cmap='viridis')
ax.set_xticks(np.arange(M))
ax.set_yticks(np.arange(M))
ax.set_xticklabels([f'{i}' for i in range(M)])
ax.set_yticklabels([f'{i}' for i in range(M)])
ax.set_xlabel('Image index', fontsize=10)
ax.set_ylabel('Image index', fontsize=10)
ax.set_title('Pairwise LPIPS Distance Matrix', fontsize=10)

for i in range(M):
    for j in range(M):
        text = ax.text(j, i, f'{dist_matrix[i, j]:.3f}',
                       ha="center", va="center", color="w", fontsize=8)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('LPIPS Distance', rotation=270, labelpad=10, fontsize=10)

for k in range(1, M):
    axes[1, k].remove()

plt.tight_layout()
plt.show()


pairs = [f'{i}-{j}' for (i,j) in [(idx_i, idx_j) for idx_i in range(M) for idx_j in range(idx_i+1, M)]]
values = [d for (_, _, d) in dist_list]

plt.figure(figsize=(8, 4))
plt.bar(pairs, values, color='C1')
plt.xlabel('Image pairs (i-j)')
plt.ylabel('LPIPS distance')
plt.title('Pairwise LPIPS Distances')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# print average distance
average_distance = np.mean(dist_matrix[np.triu_indices(M, k=1)])
print(f'Average LPIPS distance: {average_distance:.4f}')
