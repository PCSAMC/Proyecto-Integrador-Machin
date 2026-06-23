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
    page_title="Ames Housing · Tasacion ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,300,0,0&display=swap');

:root {
    --bg:#07090E; --s1:#0D1220; --s2:#131A2C; --s3:#192238;
    --a:#00DCAF; --ag:rgba(0,220,175,.12); --ad:rgba(0,220,175,.07);
    --t0:#E6EEFF; --t1:#8A9BBB; --t2:#3A4D6A;
    --bd:rgba(255,255,255,.08); --r:12px;
    --ui:'Inter',system-ui,sans-serif; --mo:'Space Mono',monospace;
    --ico:'Material Symbols Outlined';
}

/* ── Eliminar chrome de Streamlit completamente ── */
header, footer, #MainMenu,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
}

/* ── Fondo y padding ── */
html, body { margin:0; padding:0; background:var(--bg) !important; }
[data-testid="stAppViewContainer"] { background:var(--bg) !important; }
[data-testid="stMain"] { background:var(--bg) !important; padding-top:0 !important; }
section[data-testid="stMain"] > div { padding-top:0 !important; }
.main { background:var(--bg) !important; padding-top:0 !important; }
.main .block-container {
    padding: 0 1.5rem 4rem !important;
    max-width: 100% !important;
    background: var(--bg) !important;
}

/* ── Navbar fijo ── */
.navbar {
    position: sticky; top: 0; z-index: 999;
    background: var(--s1);
    border-bottom: 1px solid var(--bd);
    padding: 12px 1.5rem;
    display: flex; align-items: center; gap: 10px;
    margin: 0 -1.5rem 1.5rem;
}
.nb-logo  { font-size:20px; }
.nb-title { font-family:var(--ui); font-size:15px; font-weight:700; color:var(--t0); }
.nb-accent { color:var(--a); }
.nb-right { margin-left:auto; font-size:11px; color:var(--t2); font-family:var(--ui); letter-spacing:.04em; }

/* ── Columnas como cards visuales ── */
[data-testid="stColumn"] > div:first-child {
    background: var(--s1) !important;
    border: 1px solid var(--bd) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}

/* ── Panel header dentro del card ── */
.ph {
    background: var(--s2);
    border-bottom: 1px solid var(--bd);
    padding: 10px 16px 9px;
    font-size: 9px; font-weight:700; letter-spacing:.14em;
    text-transform: uppercase; color: var(--t2);
    display: flex; align-items: center; gap: 7px;
    font-family: var(--ui);
    margin: -1px -1px 0;
}
.ph .ico { color:var(--a); font-size:15px; }

/* ── Icono Material ── */
.ico {
    font-family: var(--ico);
    font-variation-settings: 'FILL' 0,'wght' 300,'GRAD' 0,'opsz' 20;
    font-size: 16px; vertical-align: middle; line-height: 1;
    display: inline-block;
}

/* ── Labels de inputs dentro de cards ── */
[data-testid="stColumn"] label {
    font-family: var(--ui) !important;
    font-size: 12px !important;
    color: var(--t1) !important;
    font-weight: 500 !important;
}
[data-testid="stColumn"] input[type="number"] {
    background: var(--s2) !important;
    border: 1px solid var(--bd) !important;
    border-radius: 7px !important;
    color: var(--t0) !important;
    font-family: var(--mo) !important;
    font-size: 14px !important;
}
[data-testid="stColumn"] input[type="number"]:focus {
    border-color: var(--a) !important;
    box-shadow: 0 0 0 2px var(--ag) !important;
}
[data-testid="stColumn"] [data-baseweb="select"] > div:first-child {
    background: var(--s2) !important;
    border: 1px solid var(--bd) !important;
    border-radius: 7px !important;
    color: var(--t0) !important;
    font-family: var(--ui) !important;
    font-size: 13px !important;
}

/* ── Boton predecir ── */
div[data-testid="stButton"] > button {
    background: var(--a) !important;
    color: #030810 !important;
    font-family: var(--ui) !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    transition: opacity .15s, box-shadow .2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: .83 !important;
    box-shadow: 0 0 36px var(--ag) !important;
}

/* ── Hero precio ── */
.hero {
    background:var(--s1); border:1px solid rgba(0,220,175,.22);
    border-radius:var(--r); padding:28px 32px 24px;
    position:relative; overflow:hidden;
    animation:fu .38s ease-out both;
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:240px; height:240px;
    background:radial-gradient(circle,rgba(0,220,175,.09) 0%,transparent 65%);
    pointer-events:none;
}
.h-lbl { font-size:9px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--a); margin-bottom:8px; display:flex; align-items:center; gap:6px; font-family:var(--ui); }
.h-val { font-family:var(--mo); font-size:52px; font-weight:700; letter-spacing:-.03em; color:var(--t0); line-height:1; margin-bottom:6px; }
.h-rng { font-size:12px; color:var(--t1); margin-bottom:20px; font-family:var(--ui); }
.h-row { display:flex; gap:8px; flex-wrap:wrap; }
.h-chip { flex:1; min-width:100px; background:var(--s2); border:1px solid var(--bd); border-radius:8px; padding:10px 12px; }
.h-cl { font-size:8px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--t2); margin-bottom:3px; display:flex; align-items:center; gap:4px; font-family:var(--ui); }
.h-cl .ico { font-size:11px; }
.h-cv { font-family:var(--mo); font-size:13px; font-weight:700; color:var(--t0); }
.h-bdg { display:inline-flex; align-items:center; gap:5px; background:var(--ad); border:1px solid rgba(0,220,175,.1); border-radius:5px; padding:4px 9px; margin-top:14px; font-size:10px; color:var(--a); font-family:var(--mo); }

/* ── Section divider ── */
.sdiv { display:flex; align-items:center; gap:12px; margin:1.75rem 0 1rem; }
.sdiv-t { font-size:9px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--t2); white-space:nowrap; display:flex; align-items:center; gap:6px; font-family:var(--ui); }
.sdiv-l { flex:1; height:1px; background:var(--bd); }

/* ── Cards similares ── */
.sgrid { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }
.scard { background:var(--s1); border:1px solid var(--bd); border-radius:10px; padding:14px; transition:border-color .18s,background .18s,transform .2s; animation:fu .45s ease-out both; }
.scard:hover { border-color:rgba(0,220,175,.28); background:var(--s2); transform:translateY(-3px); }
.sc-n { font-family:var(--mo); font-size:9px; font-weight:700; letter-spacing:.1em; color:var(--a); margin-bottom:8px; }
.sc-p { font-family:var(--mo); font-size:18px; font-weight:700; color:var(--t0); line-height:1.15; margin-bottom:5px; }
.sc-d { font-size:10px; color:var(--t1); line-height:1.55; font-family:var(--ui); }
.sc-seg { display:inline-block; margin-top:7px; font-size:8px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; padding:2px 6px; border-radius:4px; background:var(--s3); color:var(--t2); font-family:var(--ui); }

/* ── KPI strip ── */
.kpi { display:flex; gap:1px; background:var(--bd); border-radius:var(--r); overflow:hidden; margin-top:2rem; animation:fu .5s ease-out both; animation-delay:.1s; }
.kpi-t { flex:1; background:var(--s1); padding:20px 10px; text-align:center; }
.kpi-l { font-size:8px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--t2); margin-bottom:6px; font-family:var(--ui); }
.kpi-v { font-family:var(--mo); font-size:20px; font-weight:700; color:var(--t0); }
.kpi-v.a { color:var(--a); }

/* ── Empty state ── */
.empty { text-align:center; padding:48px 24px 28px; animation:fu .4s ease-out both; }
.empty-i { font-size:40px; margin-bottom:12px; }
.empty-h { font-size:17px; font-weight:600; color:var(--t0); margin-bottom:8px; font-family:var(--ui); }
.empty-p { font-size:13px; color:var(--t1); line-height:1.65; max-width:300px; margin:0 auto; font-family:var(--ui); }

@keyframes fu { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@media (prefers-reduced-motion:reduce) { *{animation:none!important;transition:none!important} }
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

QUAL_OPTS  = ["Very_Poor","Poor","Fair","Below_Average","Average",
              "Above_Average","Good","Very_Good","Excellent","Very_Excellent"]
GRADE_OPTS = ["Po","Fa","TA","Gd","Ex"]


# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar" translate="no">
  <span class="nb-logo">🏠</span>
  <span class="nb-title">Ames Housing &nbsp;<span class="nb-accent">Tasacion ML</span></span>
  <span class="nb-right">Grupo 4 &nbsp;&middot;&nbsp; SIS-351 UCB 2026 &nbsp;&middot;&nbsp; CRISP-DM</span>
</div>
""", unsafe_allow_html=True)


# ── Formulario ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown('<div class="ph"><span class="ico">straighten</span>Superficies</div>', unsafe_allow_html=True)
    gr_liv_area   = st.number_input("Habitable (ft²)",  min_value=300,  max_value=6000, value=1500, step=50)
    total_bsmt_sf = st.number_input("Sotano (ft²)",     min_value=0,    max_value=3500, value=800,  step=50)
    first_flr_sf  = st.number_input("Piso 1 (ft²)",     min_value=300,  max_value=4500, value=900,  step=50)
    second_flr_sf = st.number_input("Piso 2 (ft²)",     min_value=0,    max_value=2500, value=0,    step=50)
    garage_area   = st.number_input("Garaje (ft²)",     min_value=0,    max_value=1800, value=400,  step=25)

with c2:
    st.markdown('<div class="ph"><span class="ico">grade</span>Calidad</div>', unsafe_allow_html=True)
    overall_qual = st.selectbox("Calidad general", QUAL_OPTS,  index=QUAL_OPTS.index("Good"))
    kitchen_qual = st.selectbox("Cocina",          GRADE_OPTS, index=GRADE_OPTS.index("Gd"))
    exter_qual   = st.selectbox("Exterior",        GRADE_OPTS, index=GRADE_OPTS.index("TA"))
    garage_cars  = st.number_input("Autos en garaje",  min_value=0, max_value=5, value=2, step=1)
    full_bath    = st.number_input("Banos completos",  min_value=0, max_value=5, value=2, step=1)

with c3:
    st.markdown('<div class="ph"><span class="ico">location_on</span>Ubicacion y Tiempo</div>', unsafe_allow_html=True)
    year_built   = st.number_input("Ano de construccion", min_value=1872, max_value=2010, value=1990, step=1)
    year_sold    = st.number_input("Ano de venta",        min_value=2006, max_value=2010, value=2008, step=1)
    neighborhood = st.selectbox("Vecindario", hoods, index=hoods.index("North_Ames"))
    half_bath    = st.number_input("Medios banos", min_value=0, max_value=3, value=0, step=1)

st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.2, 1, 1.2])
with btn_col:
    predict_btn = st.button("Predecir precio", use_container_width=True)
st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)


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
    """
    Búsqueda en 2 etapas:
    1. Euclidiana (7 features) sobre propiedades con embedding → obtiene query_idx
    2. FAISS (128D MLP) → encuentra los k más similares en espacio latente completo
    """
    qual_map = {q: i+1 for i, q in enumerate(QUAL_OPTS)}
    hood_med = raw_df.groupby("Neighborhood")["Sale_Price"].median()

    # Limitar a las filas que tienen embedding
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

    # Propiedad más cercana en espacio raw → ancla para FAISS
    query_idx = int(np.argmin(np.linalg.norm((feats - query) / std, axis=1)))

    # FAISS busca en 128D (captura las 263 features del MLP)
    result_df = sim.search(query_idx, k=k)
    return result_df["idx"].values, result_df["price_usd"].values


def make_chart(pred_usd, n_prices):
    BG, A, BAR, T = "#0D1220", "#00DCAF", "#192238", "#8A9BBB"
    labels = ["Tu propiedad"] + [f"Similar #{i+1}" for i in range(len(n_prices))]
    prices = [pred_usd] + list(n_prices)
    hi = max(prices)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    colors = [A] + [BAR] * len(n_prices)
    ax.bar(labels, prices, color=colors, width=0.5, edgecolor="none", zorder=2)
    ax.bar(labels[:1], [pred_usd], color=A, alpha=.07, width=0.68, edgecolor="none", zorder=1)
    for i, (val, c) in enumerate(zip(prices, colors)):
        ax.text(i, val + hi * .013, f"${val/1e3:.0f}k",
                ha="center", va="bottom", fontsize=8.5, fontweight="700",
                color="#E6EEFF", fontfamily="monospace")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.set_ylim(0, hi * 1.22)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=T, labelsize=8, length=0)
    ax.grid(axis="y", color="#FFF", alpha=.03, linewidth=.8, zorder=0)
    for lbl in ax.get_xticklabels(): lbl.set_color(T)
    for lbl in ax.get_yticklabels(): lbl.set_color(T)
    plt.tight_layout(pad=.8)
    return fig


# ── Resultados ─────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Calculando..."):
        try:
            pred_usd = float(pipeline.predict_df(build_input_row())["Sale_Price_Pred"].iloc[0])
            err      = pred_usd * 0.07
            total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
            age      = year_sold - year_built
            price_m2 = pred_usd / (gr_liv_area * 0.0929)
            n_idx, n_prices = find_neighbors()

            st.markdown(f"""
            <div class="hero">
              <div class="h-lbl"><span class="ico">sell</span>Precio estimado</div>
              <div class="h-val" translate="no">${pred_usd:,.0f}</div>
              <div class="h-rng" translate="no">Intervalo &nbsp;${pred_usd-err:,.0f} &mdash; ${pred_usd+err:,.0f} &nbsp;(&plusmn;7%)</div>
              <div class="h-row">
                <div class="h-chip"><div class="h-cl"><span class="ico">open_with</span>Habitable</div><div class="h-cv" translate="no">{gr_liv_area:,} ft²</div></div>
                <div class="h-chip"><div class="h-cl"><span class="ico">layers</span>Total sf</div><div class="h-cv" translate="no">{total_sf:,} ft²</div></div>
                <div class="h-chip"><div class="h-cl"><span class="ico">calendar_today</span>Antiguedad</div><div class="h-cv" translate="no">{age} anos</div></div>
                <div class="h-chip"><div class="h-cl"><span class="ico">payments</span>Precio/m²</div><div class="h-cv" translate="no">${price_m2:,.0f}</div></div>
                <div class="h-chip"><div class="h-cl"><span class="ico">location_on</span>Vecindario</div><div class="h-cv" translate="no" style="font-size:11px">{_h.escape(neighborhood)}</div></div>
              </div>
              <div class="h-bdg" translate="no"><span class="ico" style="font-size:12px">auto_awesome</span>Ridge Regression &nbsp;&middot;&nbsp; R²=0.9359 &nbsp;&middot;&nbsp; RMSE=$20,035</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sdiv"><div class="sdiv-t"><span class="ico">search</span>5 propiedades mas similares</div><div class="sdiv-l"></div></div>', unsafe_allow_html=True)

            cards = ""
            for rank, (idx, price) in enumerate(zip(n_idx, n_prices), 1):
                seg   = _h.escape(str(sim.segments[idx]))
                rd    = raw_df.iloc[idx]
                area  = int(rd["Gr_Liv_Area"])    if "Gr_Liv_Area"    in rd.index else "—"
                qual  = _h.escape(str(rd["Overall_Qual"])) if "Overall_Qual" in rd.index else "—"
                yr    = int(rd["Year_Built"])      if "Year_Built"     in rd.index else "—"
                hood  = _h.escape(str(rd["Neighborhood"])) if "Neighborhood" in rd.index else "—"
                bsmt  = int(rd["Total_Bsmt_SF"])   if "Total_Bsmt_SF"  in rd.index else "—"
                cars  = int(rd["Garage_Cars"])      if "Garage_Cars"    in rd.index else "—"
                cards += (
                    f'<div class="scard" style="animation-delay:{rank*0.05:.2f}s">'
                    f'<div class="sc-n" translate="no">0{rank}</div>'
                    f'<div class="sc-p" translate="no">${int(price):,}</div>'
                    f'<div class="sc-d" translate="no">{area} ft² &middot; {yr}</div>'
                    f'<div class="sc-d" translate="no">Sótano: {bsmt} ft² &middot; Garaje: {cars}</div>'
                    f'<div class="sc-d">{qual} &middot; {hood}</div>'
                    f'<div class="sc-seg" translate="no">{seg}</div>'
                    f'</div>'
                )
            st.markdown(f'<div class="sgrid">{cards}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sdiv" style="margin-top:1.75rem"><div class="sdiv-t"><span class="ico">bar_chart</span>Comparativa de precios</div><div class="sdiv-l"></div></div>', unsafe_allow_html=True)
            fig = make_chart(pred_usd, n_prices)
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.exception(e)

else:
    st.markdown("""
    <div class="empty">
      <div class="empty-i">🏠</div>
      <div class="empty-h">Configura la propiedad y predice su precio</div>
      <div class="empty-p">Ingresa los datos arriba y presiona <strong style="color:#E6EEFF">Predecir precio</strong> para obtener la estimacion.</div>
    </div>
    <div class="kpi">
      <div class="kpi-t"><div class="kpi-l">Propiedades</div><div class="kpi-v" translate="no">2,927</div></div>
      <div class="kpi-t"><div class="kpi-l">Modelos ML</div><div class="kpi-v" translate="no">5</div></div>
      <div class="kpi-t"><div class="kpi-l">Mejor R²</div><div class="kpi-v a" translate="no">0.9359</div></div>
      <div class="kpi-t"><div class="kpi-l">RMSE</div><div class="kpi-v" translate="no">$20,035</div></div>
      <div class="kpi-t"><div class="kpi-l">Features</div><div class="kpi-v" translate="no">263</div></div>
      <div class="kpi-t"><div class="kpi-l">Precision@5</div><div class="kpi-v a" translate="no">70.4%</div></div>
    </div>
    """, unsafe_allow_html=True)
