"""
src/preprocessing.py
=====================
Pipeline reproducible de preprocesamiento para el dataset Ames Housing.

CRISP-DM: Data Preparation
- Limpieza y tratamiento de outliers
- Partición estratificada 70/15/15
- Pipeline sklearn (fit solo en train → transform en val/test)
- Encodings, imputación, escalado
"""

import os
import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer


# ─── Mapeos ordinales ────────────────────────────────────────────────────────
# Muchas variables de calidad usan la escala: No_Basement/No_Garage/... → 0
# y luego Po < Fa < TA < Gd < Ex

QUALITY_MAP = {
    "No_Basement": 0, "No_Garage": 0, "No_Pool": 0, "No_Fence": 0,
    "No_Fireplace": 0, "None": 0,
    "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
}

OVERALL_MAP = {str(i): i for i in range(1, 11)}
# Overall_Qual y Overall_Cond vienen como category con valores "1".."10"

EXPOSURE_MAP = {"No": 0, "No_Basement": 0, "Mn": 1, "Av": 2, "Gd": 3}

FUNCTIONAL_MAP = {
    "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
    "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8
}

FINISH_MAP = {
    "No_Basement": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6
}

LAND_SLOPE_MAP = {"Gtl": 0, "Mod": 1, "Sev": 2}

PAVED_MAP = {"N": 0, "P": 1, "Y": 2}

GARAGE_FINISH_MAP = {"No_Garage": 0, "Unf": 1, "RFn": 2, "Fin": 3}


def load_config(config_path: str = "config/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def remove_outliers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Elimina propiedades atípicas documentadas por De Cock (2011):
    GrLivArea > 4000 ft² con precio bajo (ventas parciales, no residenciales).
    """
    threshold = cfg["dataset"]["outlier_threshold_grlivarea"]
    before = len(df)
    df = df[~((df["Gr_Liv_Area"] > threshold) & (df["Sale_Price"] < 300_000))]
    after = len(df)
    print(f"[Outliers] Eliminadas {before - after} filas atípicas (Gr_Liv_Area > {threshold} ft² con precio bajo)")
    return df.reset_index(drop=True)


def apply_ordinal_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables ordinales categóricas a numéricas
    preservando el orden semántico.
    """
    df = df.copy()

    # Convertir a str para el mapeo (vienen como category)
    quality_cols = [
        "Exter_Qual", "Exter_Cond", "Bsmt_Qual", "Bsmt_Cond",
        "Heating_QC", "Kitchen_Qual", "Fireplace_Qu",
        "Garage_Qual", "Garage_Cond", "Pool_QC"
    ]
    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).map(QUALITY_MAP).fillna(0).astype(int)

    # Overall Qual / Cond: son category con valores "1".."10"
    for col in ["Overall_Qual", "Overall_Cond"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+)')[0].astype(float)

    if "Bsmt_Exposure" in df.columns:
        df["Bsmt_Exposure"] = df["Bsmt_Exposure"].astype(str).map(EXPOSURE_MAP).fillna(0)
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].astype(str).map(FUNCTIONAL_MAP).fillna(5)
    if "BsmtFin_Type_1" in df.columns:
        df["BsmtFin_Type_1"] = df["BsmtFin_Type_1"].astype(str).map(FINISH_MAP).fillna(0)
    if "BsmtFin_Type_2" in df.columns:
        df["BsmtFin_Type_2"] = df["BsmtFin_Type_2"].astype(str).map(FINISH_MAP).fillna(0)
    if "Land_Slope" in df.columns:
        df["Land_Slope"] = df["Land_Slope"].astype(str).map(LAND_SLOPE_MAP).fillna(0)
    if "Paved_Drive" in df.columns:
        df["Paved_Drive"] = df["Paved_Drive"].astype(str).map(PAVED_MAP).fillna(0)
    if "Garage_Finish" in df.columns:
        df["Garage_Finish"] = df["Garage_Finish"].astype(str).map(GARAGE_FINISH_MAP).fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering: variables derivadas útiles para el problema.
    """
    df = df.copy()

    # Superficie total de la casa (subterráneo + pisos)
    df["Total_SF"] = df["Total_Bsmt_SF"] + df["First_Flr_SF"] + df["Second_Flr_SF"]

    # Baños totales ponderados
    df["Total_Baths"] = (
        df["Full_Bath"] + 0.5 * df["Half_Bath"] +
        df["Bsmt_Full_Bath"] + 0.5 * df["Bsmt_Half_Bath"]
    )

    # Antigüedad de la propiedad al momento de la venta
    df["House_Age"] = df["Year_Sold"] - df["Year_Built"]

    # Años desde la última remodelación
    df["Remod_Age"] = df["Year_Sold"] - df["Year_Remod_Add"]

    # Indicador: fue remodelado alguna vez
    df["Was_Remodeled"] = (df["Year_Built"] != df["Year_Remod_Add"]).astype(int)

    # Porch total
    df["Total_Porch_SF"] = (
        df["Open_Porch_SF"] + df["Enclosed_Porch"] +
        df["Three_season_porch"] + df["Screen_Porch"]
    )

    # Tiene piscina (flag)
    df["Has_Pool"] = (df["Pool_Area"] > 0).astype(int)

    return df


def get_feature_groups(df: pd.DataFrame, target_col: str = "Sale_Price"):
    """
    Clasifica las columnas del DataFrame en:
    - numéricas (para StandardScaler + mediana imputer)
    - categóricas de baja cardinalidad (para OHE)
    
    Retorna: (num_cols, cat_cols)
    """
    exclude = [target_col]
    num_cols = df.select_dtypes(include=["int64", "float64", "int32", "uint8"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude]
    cat_cols = [c for c in cat_cols if c not in exclude]
    return num_cols, cat_cols


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Construye el ColumnTransformer de sklearn:
    - Numéricos: imputar con mediana → StandardScaler
    - Categóricos: imputar con 'missing' → OneHotEncoder (handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def stratified_split(df: pd.DataFrame, cfg: dict):
    """
    Partición estratificada 70/15/15 por precio en bins.
    
    Retorna: (train_df, val_df, test_df)
    """
    rs  = cfg["random_state"]
    n_bins = cfg["split"]["n_price_bins"]

    # Crear bins de precio para estratificación
    df["_price_bin"] = pd.qcut(df["Sale_Price"], q=n_bins, labels=False, duplicates="drop")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=rs, stratify=df["_price_bin"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=rs, stratify=temp_df["_price_bin"]
    )

    # Eliminar columna auxiliar
    for split in [train_df, val_df, test_df]:
        split.drop(columns=["_price_bin"], inplace=True)

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_data(df: pd.DataFrame, cfg: dict, save_dir: str = "data/processed"):
    """
    Pipeline completo: limpieza → feature engineering → split → preprocessor.

    Retorna
    -------
    X_train, X_val, X_test : np.ndarray  (ya transformados)
    y_train, y_val, y_test : np.ndarray  (log1p o crudo según config)
    preprocessor           : ColumnTransformer ajustado
    feature_names          : list[str]
    train_df, val_df, test_df : pd.DataFrame originales sin transformar
    """
    os.makedirs(save_dir, exist_ok=True)
    target_col = cfg["dataset"]["target_col"]

    # 1. Outliers
    df = remove_outliers(df, cfg)

    # 2. Mapeos ordinales
    df = apply_ordinal_mappings(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Partición
    train_df, val_df, test_df = stratified_split(df, cfg)

    # 5. Separar X / y
    y_train_raw = train_df[target_col].values
    y_val_raw   = val_df[target_col].values
    y_test_raw  = test_df[target_col].values

    # Transformación logarítmica del target (si está habilitada)
    if cfg["preprocessing"]["log_transform_target"]:
        y_train = np.log1p(y_train_raw)
        y_val   = np.log1p(y_val_raw)
        y_test  = np.log1p(y_test_raw)
    else:
        y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw

    X_train_df = train_df.drop(columns=[target_col])
    X_val_df   = val_df.drop(columns=[target_col])
    X_test_df  = test_df.drop(columns=[target_col])

    # 6. Identificar columnas por tipo
    num_cols, cat_cols = get_feature_groups(X_train_df)

    # 7. Construir y ajustar preprocessor SOLO en train
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val   = preprocessor.transform(X_val_df)
    X_test  = preprocessor.transform(X_test_df)

    # 8. Recuperar nombres de features transformadas
    cat_feature_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names

    print(f"[Preprocessing] Features totales tras OHE: {len(feature_names)}")

    # 9. Guardar artefactos
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
    print(f"[Preprocessing] Artefactos guardados en: {save_dir}/")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocessor, feature_names,
        train_df, val_df, test_df
    )
