"""
app/streamlit_app.py — Ames Housing | Grupo 4 SIS-351
python -m streamlit run app/streamlit_app.py
"""
import sys, os, html as _h
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.predict import load_pipeline
from src.similarity import PropertySimilarity
from src.preprocessing import load_config

st.set_page_config(
    page_title="Tasador de Propiedades · Grupo 4",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,300,0,0&display=swap');

:root {
    --bg:#07090E;
    --s1:#0D1220;
    --s2:#131A2C;
    --s3:#192238;
    --s4:#1E2940;
    --a:#00DCAF;
    --a2:#00B891;
    --ag:rgba(0,220,175,.14);
    --ad:rgba(0,220,175,.07);
    --t0:#E8EFFE;
    --t1:#8A9BBB;
    --t2:#3A4D6A;
    --bd:rgba(255,255,255,.07);
    --bd2:rgba(255,255,255,.12);
    --r:14px;
    --ui:'Inter',system-ui,sans-serif;
    --mo:'Space Mono',monospace;
    --ico:'Material Symbols Outlined';
}

/* ── Eliminar chrome de Streamlit ── */
header, footer, #MainMenu,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"] {
    display:none !important; height:0 !important;
    min-height:0 !important; overflow:hidden !important;
    visibility:hidden !important;
}

/* ── Fondo ── */
html,body { margin:0; padding:0; background:var(--bg) !important; }
[data-testid="stAppViewContainer"] { background:var(--bg) !important; }
[data-testid="stMain"] { background:var(--bg) !important; padding-top:0 !important; }
section[data-testid="stMain"] > div { padding-top:0 !important; }
.main { background:var(--bg) !important; padding-top:0 !important; }
.main .block-container {
    padding: 0 1.5rem 5rem !important;
    max-width: 100% !important;
    background: var(--bg) !important;
}

/* ── Navbar ── */
.navbar {
    position:sticky; top:0; z-index:999;
    background:linear-gradient(180deg, #0D1220 85%, rgba(13,18,32,0) 100%);
    border-bottom:1px solid var(--bd);
    padding:14px 1.5rem;
    display:flex; align-items:center; gap:12px;
    margin:0 -1.5rem 2rem;
    backdrop-filter:blur(12px);
}
.nb-icon {
    width:36px; height:36px; border-radius:10px;
    background:var(--ag); border:1px solid rgba(0,220,175,.2);
    display:flex; align-items:center; justify-content:center;
    font-size:18px; flex-shrink:0;
}
.nb-text {}
.nb-title {
    font-family:var(--ui); font-size:15px; font-weight:800;
    color:var(--t0); line-height:1.2; letter-spacing:-.01em;
}
.nb-sub {
    font-family:var(--ui); font-size:11px; color:var(--t2);
    margin-top:1px; letter-spacing:.01em;
}
.nb-accent { color:var(--a); }
.nb-badge {
    margin-left:auto;
    background:var(--s2); border:1px solid var(--bd);
    border-radius:20px; padding:5px 12px;
    font-size:10px; color:var(--t2); font-family:var(--ui);
    letter-spacing:.04em; font-weight:600;
    white-space:nowrap;
}

/* ── Section label ── */
.sec-label {
    font-size:10px; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:var(--t2);
    display:flex; align-items:center; gap:7px;
    font-family:var(--ui); margin-bottom:10px;
    padding:0 2px;
}
.sec-label .ico { color:var(--a); font-size:14px; }

/* ── Cards de columnas del formulario ── */
[data-testid="stColumn"] > div:first-child {
    background:var(--s1) !important;
    border:1px solid var(--bd) !important;
    border-radius:var(--r) !important;
    overflow:hidden !important;
}

/* ── Panel header de cada card ── */
.ph {
    background:linear-gradient(135deg, var(--s2) 0%, var(--s3) 100%);
    border-bottom:1px solid var(--bd);
    padding:11px 16px 10px;
    font-size:9px; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:var(--t1);
    display:flex; align-items:center; gap:7px;
    font-family:var(--ui); margin:-1px -1px 0;
}
.ph .ico { color:var(--a); font-size:15px; }

/* ── Icono Material ── */
.ico {
    font-family:var(--ico);
    font-variation-settings:'FILL' 0,'wght' 300,'GRAD' 0,'opsz' 20;
    font-size:16px; vertical-align:middle; line-height:1;
    display:inline-block;
}

/* ── Labels inputs ── */
[data-testid="stColumn"] label {
    font-family:var(--ui) !important;
    font-size:11.5px !important;
    color:var(--t1) !important;
    font-weight:500 !important;
    letter-spacing:.01em !important;
}
[data-testid="stColumn"] input[type="number"] {
    background:var(--s2) !important;
    border:1px solid var(--bd) !important;
    border-radius:8px !important;
    color:var(--t0) !important;
    font-family:var(--mo) !important;
    font-size:13px !important;
}
[data-testid="stColumn"] input[type="number"]:focus {
    border-color:var(--a) !important;
    box-shadow:0 0 0 3px var(--ag) !important;
}
[data-testid="stColumn"] [data-baseweb="select"] > div:first-child {
    background:var(--s2) !important;
    border:1px solid var(--bd) !important;
    border-radius:8px !important;
    color:var(--t0) !important;
    font-family:var(--ui) !important;
    font-size:13px !important;
}
[data-testid="stColumn"] [data-baseweb="select"] > div:first-child:focus-within {
    border-color:var(--a) !important;
    box-shadow:0 0 0 3px var(--ag) !important;
}

/* ── Hint m² ── */
.m2hint {
    font-size:10px; color:var(--t2); font-family:var(--ui);
    margin-top:-10px; margin-bottom:6px; padding-left:2px;
}

/* ── Boton predecir ── */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg, #00DCAF 0%, #00B48F 100%) !important;
    color:#030810 !important;
    font-family:var(--ui) !important;
    font-weight:800 !important;
    font-size:15px !important;
    padding:15px !important;
    border:none !important;
    border-radius:12px !important;
    width:100% !important;
    letter-spacing:.02em !important;
    transition:opacity .15s, box-shadow .2s, transform .1s !important;
    box-shadow:0 4px 24px rgba(0,220,175,.18) !important;
}
div[data-testid="stButton"] > button:hover {
    opacity:.88 !important;
    box-shadow:0 6px 36px rgba(0,220,175,.3) !important;
    transform:translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
    transform:translateY(0px) !important;
}

/* ── Hero resultado ── */
.hero {
    background:var(--s1);
    border:1px solid rgba(0,220,175,.2);
    border-radius:var(--r); padding:32px 36px 28px;
    position:relative; overflow:hidden;
    animation:fu .4s cubic-bezier(.22,.68,0,1.2) both;
}
.hero::before {
    content:''; position:absolute; top:-100px; right:-60px;
    width:280px; height:280px;
    background:radial-gradient(circle, rgba(0,220,175,.08) 0%, transparent 65%);
    pointer-events:none;
}
.hero::after {
    content:''; position:absolute; bottom:-60px; left:-40px;
    width:180px; height:180px;
    background:radial-gradient(circle, rgba(0,180,145,.05) 0%, transparent 65%);
    pointer-events:none;
}
.h-lbl {
    font-size:9px; font-weight:700; letter-spacing:.18em;
    text-transform:uppercase; color:var(--a); margin-bottom:10px;
    display:flex; align-items:center; gap:6px; font-family:var(--ui);
}
.h-val {
    font-family:var(--mo); font-size:56px; font-weight:700;
    letter-spacing:-.03em; color:var(--t0); line-height:1;
    margin-bottom:8px;
}
.h-rng {
    font-size:12px; color:var(--t1); margin-bottom:22px;
    font-family:var(--ui); display:flex; align-items:center; gap:6px;
}
.h-rng-dot {
    width:6px; height:6px; border-radius:50%;
    background:var(--a); opacity:.5; flex-shrink:0;
}
.h-chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
.h-chip {
    flex:1; min-width:110px;
    background:var(--s2); border:1px solid var(--bd);
    border-radius:10px; padding:12px 14px;
    transition:border-color .2s;
}
.h-chip:hover { border-color:rgba(0,220,175,.15); }
.h-cl {
    font-size:8px; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:var(--t2); margin-bottom:5px;
    display:flex; align-items:center; gap:4px; font-family:var(--ui);
}
.h-cl .ico { font-size:11px; }
.h-cv { font-family:var(--mo); font-size:14px; font-weight:700; color:var(--t0); }
.h-cv-sub { font-size:9px; color:var(--t2); font-family:var(--ui); margin-top:2px; }
.h-bdg {
    display:inline-flex; align-items:center; gap:6px;
    background:var(--ad); border:1px solid rgba(0,220,175,.12);
    border-radius:6px; padding:5px 11px;
    font-size:10px; color:var(--a); font-family:var(--ui);
    font-weight:500;
}

/* ── Divider sección ── */
.sdiv {
    display:flex; align-items:center; gap:12px;
    margin:2rem 0 1.2rem;
}
.sdiv-t {
    font-size:9px; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:var(--t2); white-space:nowrap;
    display:flex; align-items:center; gap:6px; font-family:var(--ui);
}
.sdiv-l { flex:1; height:1px; background:var(--bd); }

/* ── Grid similares ── */
.sgrid {
    display:grid; grid-template-columns:repeat(5,1fr); gap:10px;
}
.scard {
    background:var(--s1); border:1px solid var(--bd);
    border-radius:12px; padding:16px;
    transition:border-color .2s, background .2s, transform .2s, box-shadow .2s;
    animation:fu .45s cubic-bezier(.22,.68,0,1.2) both;
    position:relative; overflow:hidden;
}
.scard::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg, transparent, rgba(0,220,175,.3), transparent);
    opacity:0; transition:opacity .2s;
}
.scard:hover {
    border-color:rgba(0,220,175,.25);
    background:var(--s2);
    transform:translateY(-4px);
    box-shadow:0 12px 32px rgba(0,0,0,.3);
}
.scard:hover::before { opacity:1; }
.sc-n {
    font-family:var(--mo); font-size:9px; font-weight:700;
    letter-spacing:.1em; color:var(--a); margin-bottom:10px;
    opacity:.7;
}
.sc-p {
    font-family:var(--mo); font-size:20px; font-weight:700;
    color:var(--t0); line-height:1.1; margin-bottom:8px;
}
.sc-row {
    display:flex; align-items:center; gap:5px;
    font-size:10px; color:var(--t1); font-family:var(--ui);
    margin-bottom:3px;
}
.sc-row .ico { font-size:12px; color:var(--t2); }
.sc-seg {
    display:inline-block; margin-top:10px;
    font-size:9px; font-weight:700; letter-spacing:.04em;
    padding:3px 8px; border-radius:5px;
    background:var(--s3); color:var(--t2);
    font-family:var(--ui); text-transform:uppercase;
}
.sc-seg.eco { background:rgba(0,180,120,.1); color:#00B478; }
.sc-seg.med { background:rgba(100,140,255,.1); color:#7A9FFF; }
.sc-seg.alt { background:rgba(255,180,0,.1); color:#FFB800; }
.sc-seg.luj { background:rgba(200,100,255,.1); color:#C864FF; }

/* ── Frase info sistema ── */
.sys-info {
    background:var(--s1); border:1px solid var(--bd);
    border-radius:var(--r); padding:20px 28px;
    margin-top:2.5rem;
    display:flex; align-items:center; gap:18px;
    animation:fu .5s ease-out both; animation-delay:.1s;
}
.sys-info-icon {
    width:44px; height:44px; border-radius:12px;
    background:var(--ag); border:1px solid rgba(0,220,175,.15);
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; font-size:20px;
}
.sys-info-text {
    font-family:var(--ui); font-size:13px; color:var(--t1);
    line-height:1.65; flex:1;
}
.sys-info-text strong { color:var(--t0); font-weight:600; }
.sys-info-text .a { color:var(--a); font-weight:700; }

/* ── Empty state ── */
.empty {
    text-align:center; padding:56px 24px 32px;
    animation:fu .4s ease-out both;
}
.empty-glow {
    width:80px; height:80px; border-radius:24px;
    background:linear-gradient(135deg, var(--ag), rgba(0,220,175,.04));
    border:1px solid rgba(0,220,175,.15);
    display:flex; align-items:center; justify-content:center;
    font-size:36px; margin:0 auto 20px;
}
.empty-h {
    font-size:18px; font-weight:700; color:var(--t0);
    margin-bottom:10px; font-family:var(--ui); letter-spacing:-.01em;
}
.empty-p {
    font-size:13px; color:var(--t1); line-height:1.7;
    max-width:320px; margin:0 auto; font-family:var(--ui);
}

/* ── Steps guide ── */
.steps {
    display:flex; gap:10px; justify-content:center;
    margin-top:28px; flex-wrap:wrap;
}
.step {
    display:flex; align-items:center; gap:8px;
    background:var(--s1); border:1px solid var(--bd);
    border-radius:10px; padding:10px 16px;
    font-family:var(--ui); font-size:12px; color:var(--t1);
}
.step-n {
    width:22px; height:22px; border-radius:50%;
    background:var(--ag); border:1px solid rgba(0,220,175,.2);
    display:flex; align-items:center; justify-content:center;
    font-size:10px; font-weight:700; color:var(--a);
    flex-shrink:0; font-family:var(--mo);
}

@keyframes fu {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
@media (prefers-reduced-motion:reduce) {
    * { animation:none!important; transition:none!important; }
}
</style>
""", unsafe_allow_html=True)


# ── Resources ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    cfg      = load_config("config/params.yaml")
    pipeline = load_pipeline("ridge_baseline")
    raw_df   = pd.read_csv("data/raw/ames_housing_raw.csv")
    sim      = PropertySimilarity(metric="cosine")
    sim.fit("models/mlp_embeddings.npy", "models/mlp_embeddings_labels.npy", raw_df)
    return cfg, pipeline, raw_df, sim

cfg, pipeline, raw_df, sim = load_resources()
hoods = sorted(raw_df["Neighborhood"].dropna().unique().tolist())

QUAL_OPTS      = ["Very_Poor","Poor","Fair","Below_Average","Average",
                  "Above_Average","Good","Very_Good","Excellent","Very_Excellent"]
QUAL_OPTS_ES   = ["Muy malo","Malo","Regular","Por debajo del promedio","Promedio",
                  "Sobre el promedio","Bueno","Muy bueno","Excelente","Excepcional"]
GRADE_OPTS     = ["Po","Fa","TA","Gd","Ex"]
GRADE_OPTS_ES  = ["Malo","Regular","Promedio","Bueno","Excelente"]

FT2_TO_M2 = 0.0929


# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar" translate="no">
  <div class="nb-icon">🏠</div>
  <div class="nb-text">
    <div class="nb-title">Tasador de <span class="nb-accent">Propiedades</span></div>
    <div class="nb-sub">Estimación inteligente de valor residencial</div>
  </div>
  <div class="nb-badge">Grupo 4 &nbsp;·&nbsp; SIS-351 UCB 2026</div>
</div>
""", unsafe_allow_html=True)


# ── Formulario ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown('<div class="ph"><span class="ico">straighten</span>Tamaño de la propiedad</div>', unsafe_allow_html=True)
    gr_liv_area   = st.number_input("Área habitable (ft²)",   min_value=300,  max_value=6000, value=1500, step=50)
    st.markdown(f'<div class="m2hint">≈ {gr_liv_area * FT2_TO_M2:.0f} m²</div>', unsafe_allow_html=True)
    total_bsmt_sf = st.number_input("Sótano (ft²)",           min_value=0,    max_value=3500, value=800,  step=50)
    st.markdown(f'<div class="m2hint">≈ {total_bsmt_sf * FT2_TO_M2:.0f} m²</div>', unsafe_allow_html=True)
    first_flr_sf  = st.number_input("Primer piso (ft²)",      min_value=300,  max_value=4500, value=900,  step=50)
    st.markdown(f'<div class="m2hint">≈ {first_flr_sf * FT2_TO_M2:.0f} m²</div>', unsafe_allow_html=True)
    second_flr_sf = st.number_input("Segundo piso (ft²)",     min_value=0,    max_value=2500, value=0,    step=50)
    st.markdown(f'<div class="m2hint">≈ {second_flr_sf * FT2_TO_M2:.0f} m²</div>', unsafe_allow_html=True)
    garage_area   = st.number_input("Garaje (ft²)",           min_value=0,    max_value=1800, value=400,  step=25)
    st.markdown(f'<div class="m2hint">≈ {garage_area * FT2_TO_M2:.0f} m²</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="ph"><span class="ico">grade</span>Características y calidad</div>', unsafe_allow_html=True)
    overall_qual_es = st.selectbox("Calidad general de la vivienda", QUAL_OPTS_ES, index=QUAL_OPTS_ES.index("Bueno"))
    overall_qual    = QUAL_OPTS[QUAL_OPTS_ES.index(overall_qual_es)]
    kitchen_qual_es = st.selectbox("Calidad de la cocina",           GRADE_OPTS_ES, index=GRADE_OPTS_ES.index("Bueno"))
    kitchen_qual    = GRADE_OPTS[GRADE_OPTS_ES.index(kitchen_qual_es)]
    exter_qual_es   = st.selectbox("Calidad del exterior",           GRADE_OPTS_ES, index=GRADE_OPTS_ES.index("Promedio"))
    exter_qual      = GRADE_OPTS[GRADE_OPTS_ES.index(exter_qual_es)]
    garage_cars  = st.number_input("Autos en garaje",   min_value=0, max_value=5, value=2, step=1)
    full_bath    = st.number_input("Baños completos",   min_value=0, max_value=5, value=2, step=1)

with c3:
    st.markdown('<div class="ph"><span class="ico">location_on</span>Ubicación y antigüedad</div>', unsafe_allow_html=True)
    year_built   = st.number_input("Año de construcción",      min_value=1872, max_value=2010, value=1990, step=1)
    year_sold    = st.number_input("Año de venta (2006–2010)", min_value=2006, max_value=2010, value=2008, step=1)
    hoods_display = [h.replace("_", " ") for h in hoods]
    hood_display  = st.selectbox("Vecindario", hoods_display, index=hoods_display.index("North Ames"))
    neighborhood  = hoods[hoods_display.index(hood_display)]
    half_bath    = st.number_input("Medios baños",             min_value=0,    max_value=3,    value=0,    step=1)

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.2, 1, 1.2])
with btn_col:
    predict_btn = st.button("🔍  Calcular precio estimado", use_container_width=True)
st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_input_row():
    row = raw_df.iloc[0].copy()
    for k, v in dict(
        Gr_Liv_Area=gr_liv_area, Total_Bsmt_SF=total_bsmt_sf,
        Garage_Area=garage_area, First_Flr_SF=first_flr_sf,
        Second_Flr_SF=second_flr_sf, Overall_Qual=overall_qual,
        Kitchen_Qual=kitchen_qual, Exter_Qual=exter_qual,
        Year_Built=year_built, Year_Sold=year_sold,
        Neighborhood=neighborhood, Garage_Cars=garage_cars,
        Full_Bath=full_bath, Half_Bath=half_bath,
        Overall_Cond="Average", Bsmt_Full_Bath=0, Bsmt_Half_Bath=0,
    ).items():
        if k in row.index: row[k] = v
    return pd.DataFrame([row.drop("Sale_Price", errors="ignore")])


def find_neighbors(k=5):
    qual_map = {q: i+1 for i, q in enumerate(QUAL_OPTS)}
    hood_med = raw_df.groupby("Neighborhood")["Sale_Price"].median()
    n_emb = len(sim.embeddings)
    df_f = raw_df.iloc[:n_emb].copy()
    df_f["_qual_num"] = df_f["Overall_Qual"].map(qual_map).fillna(5)
    df_f["_hood_med"] = df_f["Neighborhood"].map(hood_med).fillna(hood_med.mean())
    cols  = [c for c in ["Gr_Liv_Area","Total_Bsmt_SF","Year_Built","Garage_Cars",
                         "Full_Bath","_qual_num","_hood_med"] if c in df_f.columns]
    feats = df_f[cols].fillna(0).values.astype(float)
    q_num  = qual_map.get(overall_qual, 5)
    q_hood = hood_med.get(neighborhood, hood_med.mean())
    query  = np.array([gr_liv_area, total_bsmt_sf, year_built, garage_cars,
                       full_bath, q_num, q_hood], dtype=float)[:len(cols)]
    std    = feats.std(axis=0); std[std == 0] = 1
    query_idx = int(np.argmin(np.linalg.norm((feats - query) / std, axis=1)))
    result_df = sim.search(query_idx, k=k)
    return result_df["idx"].values, result_df["price_usd"].values


def make_chart(pred_usd, n_prices):
    BG, A, BAR, T = "#0D1220", "#00DCAF", "#192238", "#8A9BBB"
    labels = ["Tu propiedad"] + [f"Similar #{i+1}" for i in range(len(n_prices))]
    prices = [pred_usd] + list(n_prices)
    hi = max(prices)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    colors = [A] + [BAR] * len(n_prices)
    ax.bar(labels, prices, color=colors, width=0.52, edgecolor="none", zorder=2, linewidth=0)
    ax.bar(labels[:1], [pred_usd], color=A, alpha=.06, width=0.7, edgecolor="none", zorder=1)
    for i, (val, c) in enumerate(zip(prices, colors)):
        ax.text(i, val + hi * .014, f"${val/1e3:.0f}k",
                ha="center", va="bottom", fontsize=8.5, fontweight="700",
                color="#E8EFFE", fontfamily="monospace")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.set_ylim(0, hi * 1.24)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=T, labelsize=8, length=0)
    ax.grid(axis="y", color="#FFF", alpha=.025, linewidth=.8, zorder=0)
    for lbl in ax.get_xticklabels(): lbl.set_color(T)
    for lbl in ax.get_yticklabels(): lbl.set_color(T)
    plt.tight_layout(pad=1.0)
    return fig


SEG_LABELS = {
    "<$120k":      ("Precio económico", "eco"),
    "$120k-$200k": ("Precio medio",     "med"),
    "$200k-$300k": ("Precio alto",      "alt"),
    ">$300k":      ("Precio de lujo",   "luj"),
}


# ── Resultados ─────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Calculando..."):
        try:
            pred_usd = float(pipeline.predict_df(build_input_row())["Sale_Price_Pred"].iloc[0])
            err      = pred_usd * 0.07
            total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
            age      = year_sold - year_built
            price_m2 = pred_usd / (gr_liv_area * FT2_TO_M2)
            hab_m2   = gr_liv_area * FT2_TO_M2
            n_idx, n_prices = find_neighbors()

            st.markdown(f"""
            <div class="hero">
              <div class="h-lbl"><span class="ico">sell</span>Precio estimado de la propiedad</div>
              <div class="h-val" translate="no">${pred_usd:,.0f}</div>
              <div class="h-rng">
                <div class="h-rng-dot"></div>
                <span translate="no">Rango estimado: ${pred_usd-err:,.0f} &mdash; ${pred_usd+err:,.0f} &nbsp;(±7%)</span>
              </div>
              <div class="h-chips">
                <div class="h-chip">
                  <div class="h-cl"><span class="ico">open_with</span>Área habitable</div>
                  <div class="h-cv" translate="no">{gr_liv_area:,} ft²</div>
                  <div class="h-cv-sub" translate="no">≈ {hab_m2:.0f} m²</div>
                </div>
                <div class="h-chip">
                  <div class="h-cl"><span class="ico">layers</span>Superficie total</div>
                  <div class="h-cv" translate="no">{total_sf:,} ft²</div>
                  <div class="h-cv-sub" translate="no">≈ {total_sf * FT2_TO_M2:.0f} m²</div>
                </div>
                <div class="h-chip">
                  <div class="h-cl"><span class="ico">history</span>Antigüedad</div>
                  <div class="h-cv" translate="no">{age} años</div>
                  <div class="h-cv-sub" translate="no">Construida en {year_built}</div>
                </div>
                <div class="h-chip">
                  <div class="h-cl"><span class="ico">payments</span>Precio por m²</div>
                  <div class="h-cv" translate="no">${price_m2:,.0f}</div>
                  <div class="h-cv-sub">por metro cuadrado</div>
                </div>
                <div class="h-chip">
                  <div class="h-cl"><span class="ico">location_on</span>Vecindario</div>
                  <div class="h-cv" translate="no" style="font-size:12px">{_h.escape(neighborhood.replace("_"," "))}</div>
                  <div class="h-cv-sub">Ames, Iowa</div>
                </div>
              </div>
              <div class="h-bdg"><span class="ico" style="font-size:12px">auto_awesome</span>
                Modelo Ridge &nbsp;·&nbsp; Precisión <strong>93.6%</strong> &nbsp;·&nbsp; Error promedio ±$20,035
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sdiv"><div class="sdiv-t"><span class="ico">search</span>5 propiedades más similares</div><div class="sdiv-l"></div></div>', unsafe_allow_html=True)

            qual_es_map = dict(zip(QUAL_OPTS, QUAL_OPTS_ES))
            cards = ""
            for rank, (idx, price) in enumerate(zip(n_idx, n_prices), 1):
                seg_raw  = str(sim.segments[idx])
                seg_lbl, seg_cls = SEG_LABELS.get(seg_raw, (seg_raw, "med"))
                rd       = raw_df.iloc[idx]
                area     = int(rd["Gr_Liv_Area"])   if "Gr_Liv_Area"   in rd.index else 0
                area_m2  = int(area * FT2_TO_M2)
                qual_raw = str(rd["Overall_Qual"])  if "Overall_Qual"  in rd.index else "—"
                qual     = _h.escape(qual_es_map.get(qual_raw, qual_raw))
                yr       = int(rd["Year_Built"])    if "Year_Built"    in rd.index else "—"
                hood     = _h.escape(str(rd["Neighborhood"]).replace("_"," ")) if "Neighborhood" in rd.index else "—"
                cards += (
                    f'<div class="scard" style="animation-delay:{rank*0.06:.2f}s">'
                    f'<div class="sc-n">0{rank}</div>'
                    f'<div class="sc-p" translate="no">${int(price):,}</div>'
                    f'<div class="sc-row"><span class="ico">open_with</span>{area:,} ft² &nbsp;·&nbsp; {area_m2} m²</div>'
                    f'<div class="sc-row"><span class="ico">calendar_today</span>Año {yr}</div>'
                    f'<div class="sc-row"><span class="ico">grade</span>{qual}</div>'
                    f'<div class="sc-row"><span class="ico">location_on</span>{hood}</div>'
                    f'<div class="sc-seg {seg_cls}">{_h.escape(seg_lbl)}</div>'
                    f'</div>'
                )
            st.markdown(f'<div class="sgrid">{cards}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sdiv" style="margin-top:2rem"><div class="sdiv-t"><span class="ico">bar_chart</span>Comparativa de precios</div><div class="sdiv-l"></div></div>', unsafe_allow_html=True)
            fig = make_chart(pred_usd, n_prices)
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.exception(e)

else:
    st.markdown("""
    <div class="empty">
      <div class="empty-glow">🏠</div>
      <div class="empty-h">Descubre el valor de una propiedad</div>
      <div class="empty-p">
        Completa los datos de la propiedad arriba y presiona
        <strong style="color:#E8EFFE">Calcular precio estimado</strong>
        para obtener una estimación basada en inteligencia artificial.
      </div>
      <div class="steps">
        <div class="step"><div class="step-n">1</div>Ingresa el tamaño</div>
        <div class="step"><div class="step-n">2</div>Selecciona la calidad</div>
        <div class="step"><div class="step-n">3</div>Elige el vecindario</div>
        <div class="step"><div class="step-n">4</div>Obtén el precio</div>
      </div>
    </div>

    <div class="sys-info">
      <div class="sys-info-icon">📊</div>
      <div class="sys-info-text">
        Analizamos <strong>2,927 casas reales</strong> para entrenar nuestro sistema.
        Acierta el precio en un <span class="a">93.6%</span> de los casos,
        con una diferencia promedio de <strong>±$20,035</strong> respecto al precio real.
        Las propiedades similares se buscan entre <strong>263 características</strong> usando inteligencia artificial.
      </div>
    </div>
    """, unsafe_allow_html=True)
