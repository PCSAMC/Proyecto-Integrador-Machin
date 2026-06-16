"""
src/predict.py
==============
Pipeline de inferencia para el modelo Ames Housing.

Uso:
    from src.predict import predict_price, load_pipeline

    # Predicción con un dict de features
    result = predict_price({'Gr_Liv_Area': 1500, 'Overall_Qual': 'Good', ...})
    print(result['price_usd'])

    # Predicción sobre un DataFrame completo
    pipeline = load_pipeline()
    predictions = pipeline.predict_df(df)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from src.preprocessing import apply_ordinal_mappings, engineer_features


MODEL_DIR   = "models"
DATA_DIR    = "data/processed"
DEFAULT_MODEL = "ridge_baseline"   # mejor R² en test set


class AmesPipeline:
    """
    Encapsula preprocesador + modelo para hacer predicciones sobre datos crudos.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name   = model_name
        self.preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
        self.model        = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))

        with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
            self.feature_names = json.load(f)

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df = apply_ordinal_mappings(df)
        df = engineer_features(df)
        # Eliminar Sale_Price si viene incluido
        df = df.drop(columns=["Sale_Price"], errors="ignore")
        return self.preprocessor.transform(df)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice el precio para cada fila de df (datos crudos sin preprocesar).
        Retorna un DataFrame con columnas: Sale_Price_Pred, Sale_Price_Pred_Log.
        """
        X = self._preprocess(df)
        y_log = self.model.predict(X)
        y_usd = np.expm1(y_log)
        return pd.DataFrame({
            "Sale_Price_Pred":     y_usd.round(2),
            "Sale_Price_Pred_Log": y_log.round(6),
        })

    def predict_single(self, features: dict) -> dict:
        """
        Predice el precio de una sola propiedad dado un dict de features.

        Parámetros
        ----------
        features : dict  — valores crudos tal como vienen del dataset original

        Retorna
        -------
        dict con price_usd, price_log y model_name
        """
        df = pd.DataFrame([features])
        result = self.predict_df(df)
        return {
            "price_usd":   float(result["Sale_Price_Pred"].iloc[0]),
            "price_log":   float(result["Sale_Price_Pred_Log"].iloc[0]),
            "model_name":  self.model_name,
        }


def load_pipeline(model_name: str = DEFAULT_MODEL) -> AmesPipeline:
    """Carga y retorna el pipeline listo para inferencia."""
    return AmesPipeline(model_name=model_name)


def predict_price(features: dict, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Atajo para predecir el precio de una sola propiedad.

    Ejemplo
    -------
    >>> result = predict_price({'Gr_Liv_Area': 1500, 'Overall_Qual': 'Good',
    ...                         'Year_Built': 2005, 'Neighborhood': 'NridgHt'})
    >>> print(f"Precio estimado: ${result['price_usd']:,.0f}")
    """
    pipeline = load_pipeline(model_name)
    return pipeline.predict_single(features)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=== Ames Housing — Pipeline de Predicción ===\n")

    # Verificar métricas usando X_test.npy (ya preprocesado) + modelo directamente
    pipeline = load_pipeline()
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    y_log  = pipeline.model.predict(X_test)
    y_pred = np.expm1(y_log)
    y_real = np.expm1(y_test)

    rmse   = np.sqrt(np.mean((y_real - y_pred) ** 2))
    mae    = np.mean(np.abs(y_real - y_pred))
    r2     = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)

    print(f"Modelo : {pipeline.model_name}")
    print(f"R²     : {r2:.4f}")
    print(f"RMSE   : ${rmse:,.0f}")
    print(f"MAE    : ${mae:,.0f}")

    # Demostración de predict_single con una fila real del dataset crudo
    print("\n--- Ejemplo: predict_single() con datos crudos ---")
    raw_df = pd.read_csv("data/raw/ames_housing_raw.csv")
    ejemplo = raw_df.iloc[0].to_dict()
    precio_real = ejemplo.pop("Sale_Price", None)
    result = predict_price(ejemplo)
    print(f"Precio real     : ${precio_real:,.0f}")
    print(f"Precio estimado : ${result['price_usd']:,.0f}")
    print(f"Error           : ${abs(precio_real - result['price_usd']):,.0f}")
