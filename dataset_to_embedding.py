import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# Load the dataset
df = pd.read_csv('labels/dataset.csv') 

# Semetic Embedding for text features
temp_map = {0:'very cold', 1:'cold', 2:'mild', 3:'warm', 4:'hot'}
gender_map  = {0:'male',  1:'female'}

texts = (
    df['dresscode'].astype(str) + ' | ' +
    df['sex'].map(gender_map)       + ' | ' +
    df['temp'].map(temp_map)
)

sbert = SentenceTransformer('all-MiniLM-L6-v2')
text_embs = sbert.encode(texts.tolist(), batch_size=32)   # (N, d_text)

# Turn shapes and textures into categorical features
cat_cols = [f's{i}' for i in range(12)] + ['t0','t1','t2'] + ['sex', 'dresscode']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_embs = ohe.fit_transform(df[cat_cols])               # (N, d_cat)

# Normalize temperature feature
scaler = MinMaxScaler()
num_embs = scaler.fit_transform(df[['temp']])            # (N, 1)

print('Numerical embeddings shape:', num_embs.shape)
print('text_embs ndim, shape:', text_embs.ndim, text_embs.shape)
print('cat_embs  ndim, shape:', cat_embs.ndim,  cat_embs.shape)
print('num_embs  ndim, shape:', num_embs.ndim,  num_embs.shape)

# Concatenate all embeddings
final_emb = np.concatenate([text_embs, cat_embs, num_embs], axis=1)  # (N, d_total)

print('Final embedding shape:', final_emb.shape)

# Save the final embedding to a file
np.save('artifacts/final_emb.npy', final_emb)
