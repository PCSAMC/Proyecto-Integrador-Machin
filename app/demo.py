"""
app/demo.py
===========
Demo CLI: predicción de precio + propiedades similares.

Uso:
    python app/demo.py                         # propiedad aleatoria del test set
    python app/demo.py --idx 42                # propiedad con índice 42
    python app/demo.py --idx 42 --k 5          # top-5 similares
    python app/demo.py --idx 42 --model xgboost

CRISP-DM: Deployment — integra predict.py + similarity.py
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.predict import load_pipeline
from src.similarity import PropertySimilarity
from src.preprocessing import load_config


SEPARATOR = "=" * 62


def run_demo(idx: int, k: int, model_name: str):
    cfg = load_config("config/params.yaml")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    X_test  = np.load("data/processed/X_test.npy")
    y_test  = np.load("data/processed/y_test.npy")
    raw_df  = pd.read_csv("data/raw/ames_housing_raw.csv")

    # Índice en el dataset completo (embeddings cubren train+val+test)
    X_train = np.load("data/processed/X_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    n_train_val = len(X_train) + len(X_val)

    # idx en el test set → idx global en embeddings
    global_idx = n_train_val + idx
    real_price = float(np.expm1(y_test[idx]))

    # ── Predicción de precio ──────────────────────────────────────────────────
    pipeline    = load_pipeline(model_name)
    pred_log    = pipeline.model.predict(X_test[idx].reshape(1, -1))
    pred_price  = float(np.expm1(pred_log))
    error_pct   = abs(real_price - pred_price) / real_price * 100

    print(f"\n{SEPARATOR}")
    print(f"  DEMO — Ames Housing | Grupo 4 SIS-351")
    print(f"{SEPARATOR}")
    print(f"  Propiedad #{idx} (test set) | Modelo: {model_name}")
    print(f"{SEPARATOR}")
    print(f"  Precio real      : ${real_price:>10,.0f}")
    print(f"  Precio predicho  : ${pred_price:>10,.0f}")
    print(f"  Error            : {error_pct:>9.2f}%")
    print(f"{SEPARATOR}")

    # ── Propiedades similares (RAG) ───────────────────────────────────────────
    sim = PropertySimilarity(metric=cfg["similarity"]["metric"])
    sim.fit(
        embeddings_path="models/mlp_embeddings.npy",
        labels_path="models/mlp_embeddings_labels.npy",
        raw_df=raw_df,
    )

    neighbors = sim.search(query_idx=global_idx, k=k)

    print(f"\n  TOP-{k} PROPIEDADES MÁS SIMILARES (por embeddings MLP):")
    print(f"  {'#':<4} {'Similitud':>10} {'Precio USD':>12} {'Segmento':>15}")
    print(f"  {'-'*50}")
    for _, row in neighbors.iterrows():
        print(f"  {int(row['idx']):<4} {row['similarity']:>10.4f} ${int(row['price_usd']):>11,} {row['segment']:>15}")

    # Mostrar features clave si están disponibles
    key_cols = [c for c in ["Gr_Liv_Area", "Overall_Qual", "Year_Built", "Neighborhood"]
                if c in neighbors.columns]
    if key_cols:
        print(f"\n  Features clave de los similares:")
        print(neighbors[key_cols].to_string(index=False))

    print(f"\n{SEPARATOR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: predicción + similitud Ames Housing")
    parser.add_argument("--idx",   type=int, default=0,                help="Índice en test set (default: 0)")
    parser.add_argument("--k",     type=int, default=5,                help="Top-K similares (default: 5)")
    parser.add_argument("--model", type=str, default="ridge_baseline", help="Modelo a usar (default: ridge_baseline)")
    args = parser.parse_args()

    run_demo(idx=args.idx, k=args.k, model_name=args.model)
