"""
src/models.py
=============
Entrenamiento, evaluación y comparación de modelos de regresión clásicos + AutoML.

CRISP-DM: Modeling + Evaluation
- Baseline: Ridge Regression
- Modelo 2: Random Forest
- Modelo 3: XGBoost
- Benchmark: AutoML con FLAML
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

import xgboost as xgb

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    print("[WARN] FLAML no instalado. Instalar con: pip install flaml")


# ─── Métricas ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = True) -> dict:
    """
    Calcula RMSE, MAE y R² en la escala original (USD).
    
    Si log_scale=True, aplica expm1() para invertir la transformación log1p.
    """
    if log_scale:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig, y_pred_orig = y_true, y_pred

    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2   = r2_score(y_true_orig, y_pred_orig)

    # RMSE en escala log (métrica estándar en competencias de housing)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "RMSE_USD":  round(rmse, 2),
        "MAE_USD":   round(mae, 2),
        "R2":        round(r2, 4),
        "RMSE_log":  round(rmse_log, 6),
    }


def print_metrics(model_name: str, metrics: dict):
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  RMSE (USD)  : ${metrics['RMSE_USD']:,.0f}")
    print(f"  MAE  (USD)  : ${metrics['MAE_USD']:,.0f}")
    print(f"  R²          : {metrics['R2']:.4f}")
    print(f"  RMSE (log)  : {metrics['RMSE_log']:.6f}")


# ─── Baseline: Ridge Regression ──────────────────────────────────────────────

def train_ridge(X_train, y_train, cfg: dict):
    """
    Baseline interpretable. RidgeCV busca el mejor alpha por CV interno.
    Ridge penaliza los coeficientes → robusto a multicolinealidad.
    """
    alphas = cfg["models"]["ridge"]["alphas"]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)
    print(f"[Ridge] Mejor alpha: {model.alpha_}")
    return model


# ─── Modelo 2: Random Forest ─────────────────────────────────────────────────

def train_random_forest(X_train, y_train, cfg: dict):
    """
    Random Forest: ensemble de árboles con bootstrap.
    Robusto a outliers y no requiere escalado.
    n_estimators=300 es suficiente para este tamaño de dataset.
    """
    params = cfg["models"]["random_forest"]
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=cfg["random_state"],
        n_jobs=params["n_jobs"]
    )
    model.fit(X_train, y_train)
    return model


# ─── Modelo 3: XGBoost ───────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val, cfg: dict):
    """
    XGBoost: Gradient Boosting con regularización L1/L2.
    Estado del arte en datos tabulares estructurados.
    Early stopping sobre validación para evitar sobreajuste.
    """
    params = cfg["models"]["xgboost"]
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        early_stopping_rounds=params["early_stopping_rounds"],
        random_state=cfg["random_state"],
        eval_metric="rmse",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"[XGBoost] Mejor iteración: {model.best_iteration}")
    return model


# ─── Benchmark: AutoML con FLAML ─────────────────────────────────────────────

def train_automl(X_train, y_train, X_val, y_val, cfg: dict):
    """
    FLAML AutoML: busca automáticamente el mejor modelo y sus hiperparámetros
    dentro del presupuesto de tiempo dado.
    
    IMPORTANTE: se evalúa con la MISMA partición y métricas que los modelos
    manuales para garantizar comparación justa.
    """
    if not FLAML_AVAILABLE:
        print("[AutoML] FLAML no disponible. Saltando benchmark.")
        return None, {}

    automl_cfg = cfg["automl"]
    automl = AutoML()

    print(f"\n[AutoML] Iniciando FLAML con time_budget={automl_cfg['time_budget']}s ...")
    start_time = time.time()

    automl.fit(
        X_train, y_train,
        task=automl_cfg["task"],
        metric=automl_cfg["metric"],
        time_budget=automl_cfg["time_budget"],
        eval_method="cv",
        n_splits=5,
        seed=cfg["random_state"],
        log_file_name=automl_cfg["log_file"],
        verbose=0,
    )

    elapsed = time.time() - start_time
    print(f"[AutoML] Finalizado en {elapsed:.1f}s")
    print(f"[AutoML] Mejor modelo: {automl.best_estimator}")
    print(f"[AutoML] Mejor RMSE (CV): {automl.best_loss:.6f}")

    return automl, {
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss_cv": round(automl.best_loss, 6),
        "time_seconds": round(elapsed, 1),
    }


# ─── Cross Validation ────────────────────────────────────────────────────────

def cross_validate_model(model, X_train, y_train, cfg: dict, model_name: str = "Model"):
    """
    Validación cruzada de 5 folds para estimar varianza del error.
    Usa neg_root_mean_squared_error como score.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg["random_state"])
    scores = cross_val_score(
        model, X_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=kf, n_jobs=-1
    )
    rmse_scores = -scores
    print(f"[CV] {model_name} — RMSE (log): {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    return rmse_scores


# ─── Análisis de error ───────────────────────────────────────────────────────

def error_analysis(model, X_test, y_test, test_df: pd.DataFrame,
                   log_scale: bool = True, top_n: int = 20) -> pd.DataFrame:
    """
    Análisis de errores del modelo sobre el test set.
    
    Retorna un DataFrame con las top_n predicciones con mayor error absoluto,
    segmentado por rango de precio.
    """
    y_pred = model.predict(X_test)

    if log_scale:
        y_true_usd = np.expm1(y_test)
        y_pred_usd = np.expm1(y_pred)
    else:
        y_true_usd, y_pred_usd = y_test, y_pred

    error_df = pd.DataFrame({
        "Sale_Price_Real": y_true_usd,
        "Sale_Price_Pred": y_pred_usd,
        "Error_Abs_USD":   np.abs(y_true_usd - y_pred_usd),
        "Error_Pct":       np.abs(y_true_usd - y_pred_usd) / y_true_usd * 100,
    })

    # Segmentación por rango de precio
    bins   = [0, 120_000, 200_000, 300_000, np.inf]
    labels = ["<$120k", "$120k-$200k", "$200k-$300k", ">$300k"]
    error_df["Price_Segment"] = pd.cut(error_df["Sale_Price_Real"], bins=bins, labels=labels)

    # Estadísticas por segmento
    print("\n[Error Analysis] Métricas por segmento de precio:")
    seg_stats = error_df.groupby("Price_Segment", observed=True).agg(
        N=("Error_Abs_USD", "count"),
        RMSE=("Error_Abs_USD", lambda x: np.sqrt((x**2).mean())),
        MAE=("Error_Abs_USD", "mean"),
        Pct_Error=("Error_Pct", "mean")
    ).round(2)
    print(seg_stats.to_string())

    # Top N casos con mayor error
    worst = error_df.nlargest(top_n, "Error_Abs_USD")
    print(f"\n[Error Analysis] Top {top_n} casos con mayor error absoluto:")
    print(worst[["Sale_Price_Real", "Sale_Price_Pred", "Error_Abs_USD", "Error_Pct", "Price_Segment"]].to_string())

    return error_df


# ─── Tabla comparativa de modelos ────────────────────────────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Construye una tabla comparativa de métricas para todos los modelos.
    
    results: { "Nombre Modelo": metrics_dict, ... }
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Modelo": model_name,
            "R²":          metrics.get("R2", "-"),
            "RMSE (USD)":  f"${metrics.get('RMSE_USD', 0):,.0f}",
            "MAE (USD)":   f"${metrics.get('MAE_USD', 0):,.0f}",
            "RMSE (log)":  metrics.get("RMSE_log", "-"),
        })
    df = pd.DataFrame(rows).set_index("Modelo")
    return df


# ─── Guardar modelos ─────────────────────────────────────────────────────────

def save_model(model, name: str, save_dir: str = "models"):
    """Serializa el modelo con joblib."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[Save] Modelo guardado: {path}")


def load_model(name: str, save_dir: str = "models"):
    path = os.path.join(save_dir, f"{name}.pkl")
    return joblib.load(path)
