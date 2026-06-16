#!/bin/bash
# ============================================================
#  entrypoint.sh — arranque del contenedor
#  1) Si no existen los modelos, corre TODO el pipeline.
#  2) Lanza la demo web (Streamlit).
# ============================================================
set -e

if [ ! -f "models/ridge_baseline.pkl" ]; then
  echo "============================================================"
  echo "  Primer arranque: generando datos y modelos."
  echo "  Esto entrena todo el pipeline y tarda unos minutos..."
  echo "============================================================"
  python run_all.py
else
  echo "Modelos ya presentes — omitiendo el entrenamiento."
fi

echo "============================================================"
echo "  Lanzando la app de tasación en:  http://localhost:8501"
echo "============================================================"
exec streamlit run app/streamlit_app.py \
     --server.port=8501 \
     --server.address=0.0.0.0
