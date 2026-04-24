"""
src/clustering.py
=================
Componente no supervisado: K-Means sobre propiedades del dataset Ames Housing.

CRISP-DM: Modeling (componente no supervisado)
- Selección de K con Elbow Method + Silhouette Score
- Entrenamiento y asignación de clusters
- Análisis e interpretación de centroides
- Visualización 2D con PCA/UMAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Variables que usaremos para clustering (numéricas, semánticamente ricas)
CLUSTER_FEATURES = [
    "Gr_Liv_Area",
    "Overall_Qual",
    "Year_Built",
    "Total_Bsmt_SF",
    "Garage_Cars",
    "Sale_Price",
    "Total_SF",        # feature engineered
    "Total_Baths",     # feature engineered
    "House_Age",       # feature engineered
]


def prepare_cluster_data(df: pd.DataFrame) -> tuple:
    """
    Prepara la matriz de features para clustering:
    - Selecciona columnas disponibles de CLUSTER_FEATURES
    - Imputa NaNs con mediana
    - Escala con StandardScaler

    Retorna: (X_scaled, X_raw, scaler, used_features)
    """
    available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X_raw = df[available_features].copy()

    # Imputar con mediana (robusta a outliers)
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    # Overall_Qual puede venir como category → convertir
    if X_raw["Overall_Qual"].dtype.name == "category":
        X_raw["Overall_Qual"] = X_raw["Overall_Qual"].astype(str).str.extract(r'(\d+)')[0].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, X_raw, scaler, available_features


def elbow_and_silhouette(X_scaled: np.ndarray, k_range: list, random_state: int = 42) -> tuple:
    """
    Calcula inertia y silhouette score para cada K.
    Útil para elegir el número óptimo de clusters.
    
    Retorna: (inertias, silhouettes)
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k >= 2:
            sil = silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled)))
            silhouettes.append(sil)
        else:
            silhouettes.append(None)

    return inertias, silhouettes


def plot_elbow_silhouette(k_range: list, inertias: list, silhouettes: list,
                          save_path: str = None):
    """
    Gráfico doble: Elbow Method (inertia) + Silhouette Score por K.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Elbow ──────────────────────────────────
    ax1.plot(k_range, inertias, "o-", color="#2E74B5", linewidth=2, markersize=7)
    ax1.set_xlabel("Número de Clusters (K)", fontsize=12)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=12)
    ax1.set_title("Elbow Method — Selección de K óptimo", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_range)

    # ── Silhouette ─────────────────────────────
    sil_k    = [k for k, s in zip(k_range, silhouettes) if s is not None]
    sil_vals = [s for s in silhouettes if s is not None]
    ax2.bar(sil_k, sil_vals, color="#2E74B5", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Número de Clusters (K)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score por K\n(más alto = mejor separación)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(sil_k)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def train_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    """Entrena K-Means con el K seleccionado."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=15, max_iter=500)
    km.fit(X_scaled)
    print(f"[KMeans] K={k} | Inertia: {km.inertia_:.2f}")
    return km


def interpret_clusters(df: pd.DataFrame, labels: np.ndarray,
                        features: list) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas por cluster para interpretar
    qué tipo de propiedades agrupa cada cluster.
    
    Retorna un DataFrame con la mediana de cada feature por cluster.
    """
    df_cluster = df[features].copy()
    df_cluster["Cluster"] = labels

    # Si Overall_Qual es category
    if df_cluster["Overall_Qual"].dtype.name == "category":
        df_cluster["Overall_Qual"] = df_cluster["Overall_Qual"].astype(str).str.extract(r'(\d+)')[0].astype(float)

    summary = df_cluster.groupby("Cluster").agg(
        N=("Cluster", "count"),
        **{f: (f, "median") for f in features if f in df_cluster.columns}
    ).round(1)

    # Calcular % del total
    summary["% Total"] = (summary["N"] / summary["N"].sum() * 100).round(1)

    return summary


def label_clusters(summary: pd.DataFrame) -> dict:
    """
    Asigna etiquetas semánticas a cada cluster basándose en
    el precio mediano y la calidad general.
    """
    if "Sale_Price" not in summary.columns:
        return {i: f"Cluster {i}" for i in summary.index}

    sorted_by_price = summary["Sale_Price"].sort_values()
    n = len(sorted_by_price)
    labels = {}
    for rank, (cluster_id, _) in enumerate(sorted_by_price.items()):
        if rank == 0:
            labels[cluster_id] = "Económico"
        elif rank == n - 1:
            labels[cluster_id] = "Lujo"
        elif rank == 1:
            labels[cluster_id] = "Gama Media"
        else:
            labels[cluster_id] = "Premium"
    return labels


def plot_cluster_pca(X_scaled: np.ndarray, labels: np.ndarray,
                     cluster_labels: dict, save_path: str = None):
    """
    Reducción PCA a 2D para visualizar los clusters.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    var_exp = pca.explained_variance_ratio_
    palette = sns.color_palette("Set2", n_colors=len(np.unique(labels)))

    plt.figure(figsize=(10, 7))
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        label_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=f"Cluster {cluster_id}: {label_name}",
            alpha=0.55, s=25, color=palette[cluster_id]
        )

    plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}% varianza)", fontsize=11)
    plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}% varianza)", fontsize=11)
    plt.title("Clusters de Propiedades — Visualización PCA 2D\n(Ames Housing Dataset)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return X_2d, pca


def plot_cluster_boxplots(df: pd.DataFrame, labels: np.ndarray,
                          features_to_plot: list, cluster_labels: dict,
                          save_path: str = None):
    """
    Boxplots de variables clave por cluster para facilitar la interpretación.
    """
    df_plot = df[features_to_plot].copy()
    df_plot["Cluster"] = [f"C{c}: {cluster_labels.get(c, '')}" for c in labels]

    if "Overall_Qual" in df_plot.columns and df_plot["Overall_Qual"].dtype.name == "category":
        df_plot["Overall_Qual"] = df_plot["Overall_Qual"].astype(str).str.extract(r'(\d+)')[0].astype(float)

    n_features = len(features_to_plot)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", n_colors=len(np.unique(labels)))

    for i, feat in enumerate(features_to_plot):
        if feat in df_plot.columns:
            sns.boxplot(
                data=df_plot, x="Cluster", y=feat,
                palette=palette, ax=axes[i], linewidth=1.2
            )
            axes[i].set_title(feat, fontsize=11, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=15)

    # Apagar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribución de Variables por Cluster", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cluster_price_distribution(df: pd.DataFrame, labels: np.ndarray,
                                    cluster_labels: dict, save_path: str = None):
    """
    Distribución de Sale_Price por cluster con KDE + histograma.
    """
    if "Sale_Price" not in df.columns:
        return

    palette = sns.color_palette("Set2", n_colors=len(np.unique(labels)))
    plt.figure(figsize=(12, 6))

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        prices = df.loc[mask, "Sale_Price"]
        label_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        sns.kdeplot(prices, label=f"Cluster {cluster_id}: {label_name} (n={mask.sum()})",
                    color=palette[cluster_id], linewidth=2)

    plt.xlabel("Sale Price (USD)", fontsize=12)
    plt.ylabel("Densidad", fontsize=12)
    plt.title("Distribución de Precio por Cluster de Mercado", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
