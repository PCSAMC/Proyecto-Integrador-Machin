#!/usr/bin/env python
"""
run_all.py — Ejecuta TODO el pipeline de extremo a extremo.
============================================================
Corre los 6 notebooks en orden y regenera todos los artefactos
(datos procesados, modelos, embeddings, figuras y tracking MLflow).

Uso:
    python run_all.py

Equivale a abrir y ejecutar manualmente, en orden:
    01_EDA → 02_Preprocessing → 03_ML_Models →
    04_Clustering → 05_DeepLearning → 06_MLOps
"""
import os
import sys
import subprocess

# Raíz del proyecto (portable): carpeta que contiene este archivo
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

NOTEBOOKS = [
    "01_EDA",
    "02_Preprocessing",
    "03_ML_Models",
    "04_Clustering",
    "05_DeepLearning",
    "06_MLOps",
]

def main():
    print("=" * 60)
    print("  PIPELINE COMPLETO — Grupo 4 Ames Housing")
    print("=" * 60)
    for i, name in enumerate(NOTEBOOKS, 1):
        path = os.path.join("notebooks", f"{name}.ipynb")
        print(f"\n[{i}/{len(NOTEBOOKS)}] Ejecutando {path} ...")
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                path,
            ],
            check=True,
        )
        print(f"      OK — {name} completado.")
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETO ✓")
    print("  Artefactos generados en: data/processed/ · models/ · reports/figures/")
    print("  Ahora puedes lanzar la demo:  streamlit run app/streamlit_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
