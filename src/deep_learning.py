"""
src/deep_learning.py
====================
MLP Tabular con PyTorch para regresión de precios Ames Housing.

CRISP-DM: Modeling + Evaluation
- Arquitectura: [input → 512 → 256 → 128 → 1]
- Cada bloque: Linear → BatchNorm1d → ReLU → Dropout
- Capa latente de 128D reutilizable como embeddings (Fase 4)
- Adam + ReduceLROnPlateau + Early Stopping
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import compute_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Arquitectura ─────────────────────────────────────────────────────────────

class MLPTabular(nn.Module):
    """
    Perceptrón Multicapa para regresión tabular.
    Arquitectura: input → 512 → 256 → 128 → 1
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()

        def block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.layer1    = block(input_dim, 512)
        self.layer2    = block(512, 256)
        self.embedding = block(256, 128)
        self.output    = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.embedding(x)
        return self.output(x).squeeze(1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Representación latente de 128D sin gradientes."""
        with torch.no_grad():
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.embedding(x)
        return x


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    cfg:     dict,
    save_path: str = "models/mlp_tabular.pt",
) -> tuple[MLPTabular, list, list, int]:
    """
    Entrena el MLP con Adam + ReduceLROnPlateau + Early Stopping.

    Retorna
    -------
    model        : MLPTabular con los mejores pesos cargados
    train_losses : RMSE por época en train (log scale)
    val_losses   : RMSE por época en val (log scale)
    best_epoch   : época donde se alcanzó el mínimo de val loss
    """
    dl_cfg   = cfg["models"]["mlp"]
    seed     = cfg["random_state"]
    torch.manual_seed(seed)

    batch_size = dl_cfg["batch_size"]
    epochs     = dl_cfg["epochs"]
    patience   = dl_cfg["patience"]
    lr         = dl_cfg["lr"]
    dropout    = dl_cfg["dropout"]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = MLPTabular(input_dim=X_train.shape[1], dropout=dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_epoch    = 0
    train_losses, val_losses = [], []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Entrenando MLP — máx {epochs} épocas | early stopping paciencia={patience} | device={DEVICE}")
    print(f"{'Época':>6} | {'Train RMSE':>12} | {'Val RMSE':>10} | {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(Xb)
        train_rmse = np.sqrt(running / len(train_ds))

        # — Validación —
        model.eval()
        running_v = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                running_v += criterion(model(Xb), yb).item() * len(Xb)
        val_rmse = np.sqrt(running_v / len(val_ds))
        val_loss = (running_v / len(val_ds))

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_rmse:>12.6f} | {val_rmse:>10.6f} | {lr_now:>10.6f}")

        # — Early stopping —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_cnt  = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\nEarly stopping en época {epoch} | mejor: época {best_epoch}")
                break

    # Cargar mejores pesos
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"Mejor época: {best_epoch} | Mejor Val RMSE (log): {np.sqrt(best_val_loss):.6f}")
    print(f"Modelo guardado: {save_path}")
    return model, train_losses, val_losses, best_epoch


# ─── Evaluación ───────────────────────────────────────────────────────────────

def evaluate_mlp(model: MLPTabular, X: np.ndarray, y_log: np.ndarray) -> dict:
    """Evalúa el MLP sobre cualquier split. Retorna métricas en USD."""
    model.eval()
    with torch.no_grad():
        y_pred_log = model(
            torch.tensor(X, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
    return compute_metrics(y_log, y_pred_log, log_scale=True)


# ─── Embeddings ───────────────────────────────────────────────────────────────

def extract_embeddings(
    model:     MLPTabular,
    X_all:     np.ndarray,
    save_path: str = "models/mlp_embeddings.npy",
) -> np.ndarray:
    """
    Extrae la representación latente de 128D para todo el dataset.
    Guarda el array en save_path.
    """
    model.eval()
    X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
    embeddings = model.get_embeddings(X_t).cpu().numpy()
    np.save(save_path, embeddings)
    print(f"Embeddings guardados: {save_path}  shape={embeddings.shape}")
    return embeddings


def load_mlp(input_dim: int, save_path: str = "models/mlp_tabular.pt") -> MLPTabular:
    """Carga un MLP entrenado desde disco."""
    model = MLPTabular(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    return model


# ─── Autoencoder Tabular ──────────────────────────────────────────────────────

class TabularAutoencoder(nn.Module):
    """
    Autoencoder Tabular para generar embeddings comprimidos (32D).

    Arquitectura:
        Encoder: input → 128 → 64 → 32  (espacio latente)
        Decoder: 32 → 64 → 128 → input  (reconstrucción)

    El espacio latente 32D se usa como representación compacta para
    búsqueda de similitud (Mini Proyecto 3 / RAG Tabular).
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Solo el encoder: retorna el vector latente 32D."""
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    cfg:     dict,
    latent_dim:  int = 32,
    save_path:   str = "models/autoencoder.pt",
) -> tuple:
    """
    Entrena el Autoencoder Tabular con Adam + Early Stopping.

    La tarea de entrenamiento es reconstruir la entrada (loss = MSE de reconstrucción).
    No usa Sale_Price — es aprendizaje no supervisado sobre las features.

    Retorna (model, train_losses, val_losses, best_epoch).
    """
    ae_cfg    = cfg["models"].get("autoencoder", {})
    epochs    = ae_cfg.get("epochs", 150)
    patience  = ae_cfg.get("patience", 20)
    lr        = ae_cfg.get("lr", 1e-3)
    batch_sz  = ae_cfg.get("batch_size", 256)
    dropout   = ae_cfg.get("dropout", 0.2)
    seed      = cfg["random_state"]
    torch.manual_seed(seed)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Xt),
        batch_size=batch_sz, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xv, Xv),
        batch_size=batch_sz, shuffle=False,
    )

    model     = TabularAutoencoder(X_train.shape[1], latent_dim=latent_dim, dropout=dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    patience_cnt = 0
    best_epoch   = 0
    train_losses, val_losses = [], []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Autoencoder — {X_train.shape[1]}D → {latent_dim}D latente | params={total_params:,}")
    print(f"{'Época':>6} | {'Train MSE':>12} | {'Val MSE':>10} | {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xb, _ in train_loader:
            Xb = Xb.to(DEVICE)
            optimizer.zero_grad()
            x_hat, _ = model(Xb)
            loss = criterion(x_hat, Xb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(Xb)
        t_loss = running / len(Xt)

        model.eval()
        running_v = 0.0
        with torch.no_grad():
            for Xb, _ in val_loader:
                Xb = Xb.to(DEVICE)
                x_hat, _ = model(Xb)
                running_v += criterion(x_hat, Xb).item() * len(Xb)
        v_loss = running_v / len(Xv)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} | {t_loss:>12.6f} | {v_loss:>10.6f} | {lr_now:>10.6f}")

        if v_loss < best_val:
            best_val = v_loss; best_epoch = epoch; patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\nEarly stopping en época {epoch} | mejor: época {best_epoch}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"Mejor época: {best_epoch} | Mejor Val MSE: {best_val:.6f}")
    print(f"Modelo guardado: {save_path}")
    return model, train_losses, val_losses, best_epoch


def extract_ae_embeddings(
    model:     "TabularAutoencoder",
    X_all:     np.ndarray,
    save_path: str = "models/ae_embeddings.npy",
) -> np.ndarray:
    """Extrae embeddings 32D del Autoencoder para todo el dataset."""
    model.eval()
    X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
    embs = model.encode(X_t).cpu().numpy()
    np.save(save_path, embs)
    print(f"AE Embeddings guardados: {save_path}  shape={embs.shape}")
    return embs


def load_autoencoder(input_dim: int, latent_dim: int = 32,
                     save_path: str = "models/autoencoder.pt") -> "TabularAutoencoder":
    """Carga un Autoencoder entrenado desde disco."""
    model = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    return model
