# 🏠 Predicción de Precios Inmobiliarios — Ames Housing

**Grupo 4 · SIS-351 Machine Learning · UCB «San Pablo» · 2026**
Metodología **CRISP-DM** · Dataset **Ames Housing** (OpenML ID 41211)

Sistema de tasación de viviendas que integra tres mini proyectos articulados:
**Machine Learning clásico**, **Deep Learning** y **MLOps + RAG (recuperación de similares)**.

**Integrantes:** Jean Marco Fernández Silva · Sergio Alejandro Arias Mayta · Marvin Larry Mollo Ramírez · Jaime Ignacio Huaycho Clavel · Sergio Alexander Mendoza Choque

---

## 🚀 Cómo ejecutarlo

Hay tres formas, de la más fácil a la más manual.

### Opción 1 · Docker (recomendada — no instalas nada de Python)

Solo necesitas tener **Docker** instalado.

```bash
docker compose up --build
```

En el primer arranque entrena todo el pipeline (unos minutos) y luego abre la demo en:
**http://localhost:8501**

> El segundo arranque es instantáneo (los modelos quedan guardados).

### Opción 2 · Local (con Python 3.11)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Reproducir TODO el pipeline de una sola vez
python run_all.py

# 3. Lanzar la demo web
streamlit run app/streamlit_app.py
```

### Opción 3 · Solo ver los resultados

No hace falta ejecutar nada: **los notebooks en `notebooks/` ya incluyen sus resultados y figuras**. Ábrelos directamente en GitHub o Jupyter para revisar todo el análisis.

---

## 📂 Estructura del proyecto

```
Proyecto-Integrador/
├── data/
│   └── raw/                  Dataset original (Ames Housing, OpenML 41211)
├── notebooks/                Análisis por fase CRISP-DM (con resultados)
│   ├── 01_EDA.ipynb              Fase 2 — Exploración de datos
│   ├── 02_Preprocessing.ipynb    Fase 2 — Limpieza + Feature Engineering
│   ├── 03_ML_Models.ipynb        Fase 2 — ML clásico + AutoML + SHAP
│   ├── 04_Clustering.ipynb       Fase 2 — K-Means (no supervisado)
│   ├── 05_DeepLearning.ipynb     Fase 3 — MLP + Autoencoder
│   └── 06_MLOps.ipynb            Fase 4 — MLflow + FAISS + PSI
├── src/                      Código reutilizable (pipeline, modelos, etc.)
├── app/                      Demo: streamlit_app.py + demo.py (CLI)
├── config/params.yaml        Hiperparámetros y configuración central
├── reports/                  Informe final + figuras
├── run_all.py                Ejecuta los 6 notebooks en orden
├── Dockerfile / docker-compose.yml
├── requirements.txt
└── README.md
```

> Las carpetas `models/`, `data/processed/`, `reports/figures/` y `mlruns/` **no se versionan**: se generan automáticamente al correr el pipeline (`run_all.py` o Docker).

---

## 🔧 Pipeline (CRISP-DM)

| Fase CRISP-DM | Notebook | Qué hace |
|---|---|---|
| Data Understanding | `01_EDA` | EDA, correlaciones, riesgos de datos |
| Data Preparation | `02_Preprocessing` | Imputación, encoding, log, split 70/15/15 |
| Modeling (ML) | `03_ML_Models` | Ridge, RF, XGBoost + AutoML (FLAML) + SHAP |
| Modeling (no superv.) | `04_Clustering` | K-Means K=4, segmentos de mercado |
| Modeling (DL) | `05_DeepLearning` | MLP + Autoencoder, embeddings 128D |
| Deployment | `06_MLOps` | MLflow, FAISS (RAG), PSI (monitoreo) |

---

## 📊 Resultados principales

| Modelo | R² (test) | RMSE (USD) |
|---|---|---|
| **Ridge (mejor)** | **0,9359** | **$20.035** |
| AutoML FLAML | 0,9311 | $20.773 |
| XGBoost | 0,9219 | $22.106 |
| Random Forest | 0,9017 | $24.805 |
| MLP (Deep Learning) | 0,8619 | $29.401 |

- **Recuperación de similares (FAISS):** Precisión@5 = **70,4%** · 196× más rápido que NearestNeighbors
- **Monitoreo (PSI):** trigger de reentrenamiento automático cuando PSI ≥ 0,25

---

## 🔁 Reproducibilidad

- Semilla global `random_state = 42` en todo el proyecto
- Partición estratificada por quintiles de precio (70 / 15 / 15)
- Pipeline ajustado **solo en train** (sin data leakage)
- Trazabilidad de datos con hashes MD5 (`data/processed/metadata.json`)
- Dataset descargado de forma verificable desde **OpenML ID 41211**
