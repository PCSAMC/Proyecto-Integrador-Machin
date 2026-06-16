"""
src/mlops.py
============
Tracking de experimentos con MLflow para el proyecto Ames Housing.

CRISP-DM: Deployment
- Logging de hiperparámetros, métricas y artefactos por run
- MLflow Model Registry: Staging → Production
- Generación de metadata.json para versionado de datos
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

from src.models import compute_metrics


# ─── Configuración ────────────────────────────────────────────────────────────

def setup_mlflow(cfg: dict):
    """Configura el tracking URI y crea el experimento si no existe."""
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment   = cfg["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"[MLflow] Tracking URI : {tracking_uri}")
    print(f"[MLflow] Experimento  : {experiment}")


# ─── Logging de modelos clásicos (Fase 2) ────────────────────────────────────

def log_sklearn_model(
    model,
    model_name:  str,
    params:      dict,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    artifact_path: str = "model",
) -> str:
    """
    Registra un modelo sklearn en MLflow.
    Retorna el run_id.
    """
    metrics = compute_metrics(y_test, model.predict(X_test), log_scale=True)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("phase",      "2-ML_Clasico")
        mlflow.set_tag("model_type", model_name)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "R2":       metrics["R2"],
            "RMSE_USD": metrics["RMSE_USD"],
            "MAE_USD":  metrics["MAE_USD"],
            "RMSE_log": metrics["RMSE_log"],
        })

        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        run_id = run.info.run_id

    print(f"[MLflow] {model_name:20s} → run_id={run_id[:8]}...  R²={metrics['R2']:.4f}  RMSE=${metrics['RMSE_USD']:,.0f}")
    return run_id


def log_mlp_model(
    metrics:    dict,
    cfg:        dict,
    best_epoch: int,
    model_path: str = "models/mlp_tabular.pt",
) -> str:
    """
    Registra el MLP en MLflow usando métricas pre-calculadas.
    Adjunta el archivo .pt como artefacto (sin importar torch).
    """
    dl_cfg = cfg["models"]["mlp"]

    with mlflow.start_run(run_name="MLP_Tabular") as run:
        mlflow.set_tag("phase",      "3-DeepLearning")
        mlflow.set_tag("model_type", "MLP_Tabular")

        mlflow.log_params({
            "architecture":  "263→512→256→128→1",
            "dropout":       dl_cfg["dropout"],
            "batch_size":    dl_cfg["batch_size"],
            "lr":            dl_cfg["lr"],
            "epochs_max":    dl_cfg["epochs"],
            "patience":      dl_cfg["patience"],
            "best_epoch":    best_epoch,
            "optimizer":     "Adam",
            "scheduler":     "ReduceLROnPlateau",
        })
        mlflow.log_metrics({
            "R2":       metrics["R2"],
            "RMSE_USD": metrics["RMSE_USD"],
            "MAE_USD":  metrics["MAE_USD"],
            "RMSE_log": metrics["RMSE_log"],
        })

        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")
        run_id = run.info.run_id

    print(f"[MLflow] MLP_Tabular          → run_id={run_id[:8]}...  R²={metrics['R2']:.4f}  RMSE=${metrics['RMSE_USD']:,.0f}")
    return run_id


def log_all_models(
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    cfg:        dict,
    mlp_metrics: dict = None,
    best_epoch:  int  = 0,
) -> dict:
    """
    Registra todos los modelos (Fase 2 + Fase 3) en MLflow.
    mlp_metrics: dict con R2, RMSE_USD, MAE_USD, RMSE_log del MLP
                 (se obtiene de notebook 05, evita importar torch aquí).
    Retorna dict {nombre: run_id}.
    """
    run_ids = {}

    # ── Fase 2: modelos clásicos ──────────────────────────────────────────────
    ridge  = joblib.load("models/ridge_baseline.pkl")
    rf     = joblib.load("models/random_forest.pkl")
    xgb    = joblib.load("models/xgboost.pkl")
    automl = joblib.load("models/automl_flaml.pkl")

    ridge_params = {"model": "RidgeCV", "alpha": float(ridge.alpha_), "cv": 5}
    rf_params    = {
        "n_estimators":     cfg["models"]["random_forest"]["n_estimators"],
        "max_depth":        str(cfg["models"]["random_forest"]["max_depth"]),
        "min_samples_leaf": cfg["models"]["random_forest"]["min_samples_leaf"],
    }
    xgb_cfg    = cfg["models"]["xgboost"]
    xgb_params = {
        "n_estimators":          xgb_cfg["n_estimators"],
        "learning_rate":         xgb_cfg["learning_rate"],
        "max_depth":             xgb_cfg["max_depth"],
        "subsample":             xgb_cfg["subsample"],
        "colsample_bytree":      xgb_cfg["colsample_bytree"],
        "early_stopping_rounds": xgb_cfg["early_stopping_rounds"],
        "best_iteration":        int(xgb.best_iteration),
    }
    automl_params = {
        "time_budget":    cfg["automl"]["time_budget"],
        "best_estimator": str(automl.best_estimator),
    }

    run_ids["Ridge"]   = log_sklearn_model(ridge,  "Ridge_Baseline", ridge_params,  X_test, y_test)
    run_ids["RF"]      = log_sklearn_model(rf,     "Random_Forest",  rf_params,     X_test, y_test)
    run_ids["XGBoost"] = log_sklearn_model(xgb,    "XGBoost",        xgb_params,    X_test, y_test)
    run_ids["AutoML"]  = log_sklearn_model(automl, "AutoML_FLAML",   automl_params, X_test, y_test)

    # ── Fase 3: MLP (métricas pre-calculadas, sin importar torch) ────────────
    if mlp_metrics is not None:
        run_ids["MLP"] = log_mlp_model(mlp_metrics, cfg, best_epoch)

    return run_ids


# ─── MLflow Model Registry ────────────────────────────────────────────────────

def register_best_model(run_id: str, cfg: dict, stage: str = "Staging") -> None:
    """
    Registra el modelo de un run en el Model Registry y lo mueve al stage indicado.
    stage: 'Staging' | 'Production'
    """
    model_name = cfg["mlflow"]["registered_model_name"]
    model_uri  = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage
    )
    print(f"[MLflow Registry] Modelo '{model_name}' v{version} → {stage}")


# ─── Versionado de datos (metadata.json) ─────────────────────────────────────

def save_data_metadata(cfg: dict, save_path: str = "data/processed/metadata.json") -> None:
    """
    Genera metadata.json con información de versionado del dataset y los splits.
    """
    import hashlib

    def file_hash(path):
        if not os.path.exists(path):
            return "not_found"
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]

    X_train = np.load("data/processed/X_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    X_test  = np.load("data/processed/X_test.npy")

    metadata = {
        "project":     "Ames Housing — Grupo 4 SIS-351",
        "created_at":  datetime.datetime.now().isoformat(),
        "dataset": {
            "source":   "OpenML ID 41211",
            "records":  2930,
            "features": 81,
            "target":   "Sale_Price",
            "outliers_removed": 3,
        },
        "splits": {
            "random_state": cfg["random_state"],
            "strategy":     "stratified by price quintiles",
            "train": {"rows": int(X_train.shape[0]), "features": int(X_train.shape[1])},
            "val":   {"rows": int(X_val.shape[0]),   "features": int(X_val.shape[1])},
            "test":  {"rows": int(X_test.shape[0]),  "features": int(X_test.shape[1])},
        },
        "file_hashes": {
            "X_train.npy": file_hash("data/processed/X_train.npy"),
            "X_val.npy":   file_hash("data/processed/X_val.npy"),
            "X_test.npy":  file_hash("data/processed/X_test.npy"),
            "preprocessor.pkl": file_hash("models/preprocessor.pkl"),
        },
        "preprocessing": {
            "log_transform_target": cfg["preprocessing"]["log_transform_target"],
            "numeric_features": int(X_train.shape[1]),
            "encoding": "OneHotEncoder (handle_unknown=ignore)",
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[Metadata] Guardado: {save_path}")


# ─── PSI — Population Stability Index ────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual:   np.ndarray,
    n_bins:   int = 10,
    feature_name: str = "feature",
) -> dict:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.

    El PSI mide cuánto cambió la distribución de una variable entre el
    dataset de entrenamiento (expected) y nuevos datos (actual).

    Interpretación estándar:
        PSI < 0.10  → Sin cambio significativo ✓ No reentrenar
        0.10 ≤ PSI < 0.25 → Cambio moderado ⚠ Monitorear
        PSI ≥ 0.25  → Cambio mayor ✗ Reentrenar modelo

    Retorna dict con psi_total, interpretación, tabla por bin.
    """
    # Crear bins basados en la distribución esperada (train)
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # eliminar duplicados

    def _safe_pct(arr, bp):
        counts = np.histogram(arr, bins=bp)[0]
        pct    = counts / len(arr)
        pct    = np.where(pct == 0, 1e-6, pct)   # evitar log(0)
        return pct

    exp_pct = _safe_pct(expected, breakpoints)
    act_pct = _safe_pct(actual,   breakpoints)

    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi_total = float(np.sum(psi_bins))

    if psi_total < 0.10:
        status = "OK — Sin deriva significativa"
        action = "Continuar sin reentrenar"
        emoji  = "✓"
    elif psi_total < 0.25:
        status = "ADVERTENCIA — Deriva moderada"
        action = "Monitorear de cerca; evaluar reentrenamiento"
        emoji  = "⚠"
    else:
        status = "ALERTA — Deriva mayor (PSI ≥ 0.25)"
        action = "REENTRENAR el modelo"
        emoji  = "✗"

    result = {
        "feature":      feature_name,
        "psi_total":    round(psi_total, 6),
        "n_bins":       n_bins,
        "status":       status,
        "action":       action,
        "emoji":        emoji,
        "psi_per_bin":  psi_bins.tolist(),
        "exp_pct":      exp_pct.tolist(),
        "act_pct":      act_pct.tolist(),
    }

    print(f"[PSI] {feature_name:30s}: PSI={psi_total:.4f}  {emoji} {status}")
    return result


def compute_psi_report(
    X_train:       np.ndarray,
    X_new:         np.ndarray,
    feature_names: list,
    top_n:         int = 10,
    n_bins:        int = 10,
    threshold:     float = 0.25,
) -> dict:
    """
    Calcula PSI para todas las features y genera un reporte de deriva.

    Retorna dict con:
        - results: lista de dicts con PSI por feature
        - trigger_retrain: bool (True si alguna feature supera el threshold)
        - features_drifted: lista de features con PSI >= threshold
        - summary_df: DataFrame ordenado por PSI descendente
    """
    results = []
    for i, name in enumerate(feature_names):
        if i >= X_train.shape[1]:
            break
        r = compute_psi(X_train[:, i], X_new[:, i],
                        n_bins=n_bins, feature_name=name)
        results.append(r)

    summary_df = (
        pd.DataFrame([{"Feature": r["feature"], "PSI": r["psi_total"],
                       "Status": r["emoji"] + " " + r["status"]}
                      for r in results])
        .sort_values("PSI", ascending=False)
        .reset_index(drop=True)
    )

    features_drifted = [r["feature"] for r in results if r["psi_total"] >= threshold]
    trigger_retrain  = len(features_drifted) > 0

    print(f"\n[PSI Report] Top-{top_n} features con mayor deriva:")
    print(summary_df.head(top_n).to_string(index=False))
    print(f"\n{'='*55}")
    if trigger_retrain:
        print(f"  TRIGGER DE REENTRENAMIENTO ACTIVADO (PSI >= {threshold})")
        print(f"  Features con deriva: {features_drifted[:5]}")
    else:
        print(f"  Sin deriva mayor. No se requiere reentrenamiento.")
    print(f"{'='*55}")

    return {
        "results":          results,
        "trigger_retrain":  trigger_retrain,
        "features_drifted": features_drifted,
        "summary_df":       summary_df,
    }
