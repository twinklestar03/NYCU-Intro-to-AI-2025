import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv('labels/dataset.csv')  
try:
    from sentence_transformers import SentenceTransformer
    temp_map = {0:'very cold',1:'cold',2:'mild',3:'warm',4:'hot'}
    sex_map  = {0:'male',1:'female'}
    texts = (
        df['dresscode'].astype(str) + ' | ' +
        df['sex'].map(sex_map)       + ' | ' +
        df['temp'].map(temp_map)
    )
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embs = model.encode(texts.tolist(), batch_size=32)
    print('Text embeddings shape:', text_embs.shape)
except ImportError:
    print('No sentence-transformers → using zeros.')
    text_embs = np.zeros((len(df), 1))

# One-hot encode shapes s0–s11 與 textures t0–t2
cat_cols = [f's{i}' for i in range(12)] + ['t0','t1','t2']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_embs = ohe.fit_transform(df[cat_cols])
print('Categorical one-hot shape:', cat_embs.shape)

# Normalize temperature
scaler = MinMaxScaler()
num_embs = scaler.fit_transform(df[['temp']])
print('Numeric embeddings shape:', num_embs.shape)

# Concatenate all embeddings
final_emb = np.concatenate([text_embs, cat_embs, num_embs], axis=1)
print('Final embedding shape:', final_emb.shape)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(final_emb)
df["cluster"] = labels

n_samples = 4
for c in sorted(df["cluster"].unique()):
    print(f"\n=== Cluster {c} (Total Entries { (df['cluster']==c).sum() }) ===")
    sample = df[df["cluster"]==c].sample(n=min(n_samples, (df['cluster']==c).sum()), random_state=42)
    print(sample[["key","temp","sex","dresscode"]])

for c in sorted(df["cluster"].unique()):
    dist = df[df["cluster"]==c]["dresscode"].value_counts(normalize=True)
    print(f"\nCluster {c} dresscode distribution:")
    print(dist)

# 2. PCA 降到 3 維
pca3 = PCA(n_components=3, random_state=42)
coords3 = pca3.fit_transform(final_emb)

# 3. 畫 3D 散點圖
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    coords3[:,0], coords3[:,1], coords3[:,2],
    c=labels,                # 讓不同群有不同顏色（使用預設 colormap）
    alpha=0.8
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(f"3D PCA of Outfit Embeddings (K={n_clusters})")
plt.show()