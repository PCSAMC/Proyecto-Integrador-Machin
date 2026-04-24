"""
src/utils.py
============
Funciones de utilidad compartidas entre notebooks y scripts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yaml
import os


def load_config(config_path: str = "config/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_plot_style():
    """Estilo de gráficos uniforme para todo el proyecto."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi":      120,
        "axes.titlesize":  13,
        "axes.labelsize":  11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def usd_formatter(x, pos):
    """Formato de eje Y en miles/millones de USD."""
    if x >= 1_000_000:
        return f"${x/1e6:.1f}M"
    elif x >= 1_000:
        return f"${x/1e3:.0f}k"
    return f"${x:.0f}"


def save_fig(fig_name: str, figures_dir: str = "reports/figures"):
    """Guarda la figura actual en PNG."""
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{fig_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Fig] Guardada: {path}")


def print_section(title: str):
    """Imprime un separador de sección para claridad en outputs."""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title.upper()}")
    print(f"{line}\n")


def get_top_correlations(df: pd.DataFrame, target: str = "Sale_Price",
                         n: int = 15) -> pd.Series:
    """
    Retorna las n variables con mayor correlación absoluta con el target.
    Solo considera columnas numéricas.
    """
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()[target].drop(target)
    return corr.abs().sort_values(ascending=False).head(n)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen estadístico de variables numéricas incluyendo skewness y kurtosis.
    """
    desc = df.select_dtypes(include="number").describe().T
    desc["skew"]     = df.select_dtypes(include="number").skew()
    desc["kurtosis"] = df.select_dtypes(include="number").kurtosis()
    desc["missing"]  = df.select_dtypes(include="number").isnull().sum()
    desc["missing%"] = (desc["missing"] / len(df) * 100).round(2)
    return desc.round(3)


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla de valores faltantes por columna (solo las que tienen missing > 0).
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    pct     = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct})
