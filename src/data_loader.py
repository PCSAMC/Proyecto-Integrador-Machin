"""
src/data_loader.py
==================
Descarga, carga y trazabilidad del dataset Ames Housing desde OpenML.

CRISP-DM: Data Understanding
"""

import json
import os
from datetime import datetime

import openml
import pandas as pd
import yaml


def load_config(config_path: str = "config/params.yaml") -> dict:
    """Carga el archivo de configuración centralizado."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_ames_housing(openml_id: int = 41211, save_dir: str = "data/raw") -> pd.DataFrame:
    """
    Descarga el dataset Ames Housing desde OpenML y lo guarda localmente.
    
    Parámetros
    ----------
    openml_id : int
        ID del dataset en OpenML (41211 = Ames Housing procesado por make_ames()).
    save_dir : str
        Carpeta donde guardar el CSV descargado.

    Retorna
    -------
    pd.DataFrame
        Dataset completo con Sale_Price como última columna.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "ames_housing_raw.csv")

    # Si ya existe, cargar directo (no re-descargar)
    if os.path.exists(csv_path):
        print(f"[INFO] Dataset ya descargado. Cargando desde: {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    print(f"[INFO] Descargando dataset OpenML ID={openml_id} ...")
    dataset = openml.datasets.get_dataset(openml_id)

    X, y, cat_indicator, attr_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe"
    )

    # Unir features + target
    df = pd.concat([X, y.rename("Sale_Price")], axis=1)

    # Guardar CSV
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Dataset guardado en: {csv_path}  ({df.shape[0]} filas × {df.shape[1]} columnas)")

    # Guardar metadata de trazabilidad
    metadata = {
        "dataset_name": "Ames Housing",
        "openml_id": openml_id,
        "source_url": "https://www.openml.org/search?type=data&status=active&id=41211",
        "author": "Dean De Cock (2011)",
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "target_col": "Sale_Price",
        "local_path": csv_path,
    }
    meta_path = os.path.join("data", "metadata.json")
    os.makedirs("data", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Metadata guardada en: {meta_path}")

    return df


def load_ames_housing(csv_path: str = "data/raw/ames_housing_raw.csv") -> pd.DataFrame:
    """
    Carga el dataset desde CSV local.
    Llama a download_ames_housing() si no existe el archivo.
    """
    if not os.path.exists(csv_path):
        print("[INFO] CSV no encontrado. Iniciando descarga...")
        return download_ames_housing()
    df = pd.read_csv(csv_path)
    # Restaurar dtypes de categorías conocidas (se pierden al guardar CSV)
    # Se re-aplican en preprocessing.py
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


if __name__ == "__main__":
    df = download_ames_housing()
    print(df.head())
    print(df.dtypes.value_counts())
