import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import math

class Weather2StyleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        style_dim: int,
        hidden: list[int] | None = None,
        dropout: float = 0.2,
        act: nn.Module | None = None,
    ):
        super().__init__()
        hidden = hidden or [128, 256, 256, 128]
        act = act or nn.ReLU(inplace=True)

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                act,
                nn.Dropout(dropout),
            ]
            prev = h

        # output head
        layers.append(nn.Linear(prev, style_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """
        xavier-uniform + small bias init
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(embedding_path:str, dataset_path:str , model_path: str):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Prepare input features
    scaler_temp = MinMaxScaler()
    temp_feat = scaler_temp.fit_transform(df[['temp']])   # (N,1)

    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_feats = oh.fit_transform(df[['sex','dresscode']])  # (N, n_sex+n_dresscode)

    X = np.concatenate([temp_feat, cat_feats], axis=1)
    input_dim = X.shape[1]      # = 1 + (sex_onehot + dresscode_onehot)

    # Load final_emb from artifacts/final_emb.npy
    final_emb = np.load(embedding_path)  # (N, style_dim)
    Y = final_emb               # (N, style_dim)

    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float()
    dataset_path = TensorDataset(X_tensor, Y_tensor)
    loader  = DataLoader(dataset_path, batch_size=32, shuffle=True)

    # Setup model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Weather2StyleMLP(input_dim=input_dim, style_dim=Y.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Check if saved model exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return {
            'scaler_temp': scaler_temp,
            'oh_encoder': oh,
            'model': model,
            'Y': Y,
            'df': df
        }

    # Training loop
    n_epochs = 100
    for epoch in range(1, n_epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:2d} | MSE Loss: {avg_loss:.4f}')

    # Save model after training
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return {
        'scaler_temp': scaler_temp,
        'oh_encoder': oh,
        'model': model,
        'Y': Y,
        'df': df
    }
