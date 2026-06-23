#!/bin/bash
# ============================================================
#  entrypoint.sh — arranque del contenedor
#  1) Si no existen los modelos, corre TODO el pipeline.
#  2) Si faltan los modelos no, pero falta el registro MLflow, lo regenera.
#  3) Lanza MLflow UI (5000) + la demo web Streamlit (8501).
# ============================================================
set -e

if [ ! -f "models/ridge_baseline.pkl" ]; then
  echo "============================================================"
  echo "  Primer arranque: generando datos y modelos."
  echo "  Esto entrena todo el pipeline y tarda unos minutos..."
  echo "============================================================"
  python run_all.py
elif [ ! -f "mlruns/mlflow.db" ]; then
  echo "============================================================"
  echo "  Modelos presentes pero falta el registro MLflow."
  echo "  Regenerando solo el tracking (notebook 06)..."
  echo "============================================================"
  python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/06_MLOps.ipynb
else
  echo "Modelos y MLflow ya presentes — omitiendo el entrenamiento."
fi

echo "============================================================"
echo "  MLflow UI      →  http://localhost:5000"
echo "  App de tasacion →  http://localhost:8501"
echo "============================================================"

# MLflow UI en segundo plano (lee la BD persistida en mlruns/)
mlflow ui \
     --backend-store-uri sqlite:///mlruns/mlflow.db \
     --host 0.0.0.0 \
     --port 5000 &

# Streamlit en primer plano (mantiene vivo el contenedor)
exec streamlit run app/streamlit_app.py \
     --server.port=8501 \
     --server.address=0.0.0.0
