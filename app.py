# app.py
# ─────────────────────────────────────────────────────────────────────────────
# David Indicators Confluence — Streamlit Dashboard
#   Tab 1 : Dashboard Señales  (v3 labels + multi-TF sync bar)
#   Tab 2 : Gráficos Estrategia  (9 panels + Estocástico) — ON DEMAND
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import io
import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

from ta.trend    import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from indicators import (
    download_df, clean_yf_df,
    mcginley_dynamic, calculate_pvi, calculate_nvi,
    calc_mfi_blai5, calc_stoch, awesome_osc,
    calculate_bbwp, bbwp_signal,
    compute_blai5_koncorde, blai5_signals,
    clasificar_bitman, detectar_divergencia_simple,
    azul_z_score,
    semaforo_salida,
    calcular_señales_principales, clasificar_fase_v3,
    calcular_alineacion, get_spy_context, asignar_etiqueta,
    get_confluence_dashboard,
    calc_stochastic_full,
)
from config import ALL_TICKERS, TICKER_NAMES, GRUPOS
from signal_engine import scan_ticker, fetch_daily, latest_signal, psm_bars, detectar as detectar_cunas_v3

import yfinance as yf

INTERVAL_CONFIG = {
    "1D": {"yf_interval":"1d",  "yf_period":"2y",  "resample":None, "label":"Diario"},
    "1W": {"yf_interval":"1wk", "yf_period":"5y",  "resample":None, "label":"Semanal"},
    "4h": {"yf_interval":"1h",  "yf_period":"60d", "resample":"4h", "label":"4 horas"},
    "1h": {"yf_interval":"1h",  "yf_period":"30d", "resample":None, "label":"1 hora"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="David L'Crack · Radar de señales",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    body, .stApp { background-color: #0d0f14; color: #ffffff; }
    .stDataFrame thead th { background: #13161e !important; color: #efb030 !important; }
    .stDataFrame tbody td { color: #ffffff !important; font-size: 0.78rem; }
    .stDataFrame tbody tr:hover { background: #1f2430 !important; }
    .stDataFrame { overflow-x: auto !important; }
    [data-testid="stMetricValue"]  { color: #ffffff !important; }
    [data-testid="stMetricDelta"]  { color: #c8cad0 !important; }
    [data-testid="stMetricLabel"]  { color: #efb030 !important; }
    label { color: #ffffff !important; }
    .stSlider label { color: #ffffff !important; }
    .stCaption { color: #aaaaaa !important; }
    /* Navegacion: selectores accesibles que usa Streamlit/BaseWeb */
    div[data-testid="stTabs"] [role="tablist"],
    .stTabs [role="tablist"] {
        background: #111827 !important;
        border-bottom: 2px solid #43546c !important;
        gap: 12px !important;
        padding: 10px 4px 12px !important;
        align-items: center !important;
    }
    div[data-testid="stTabs"] [role="tab"],
    .stTabs [role="tab"] {
        background-color: #27374e !important;
        background-image: none !important;
        color: #ffffff !important;
        opacity: 1 !important;
        font-size: 19px !important;
        font-weight: 800 !important;
        min-height: 60px !important;
        padding: 14px 27px !important;
        border: 2px solid #62748b !important;
        border-radius: 12px !important;
        box-shadow: 0 3px 10px #0008 !important;
    }
    div[data-testid="stTabs"] [role="tab"] :is(p,span,div,svg),
    .stTabs [role="tab"] :is(p,span,div,svg) {
        color: #ffffff !important;
        fill: currentColor !important;
        opacity: 1 !important;
        font-size: inherit !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] [role="tab"]:hover,
    .stTabs [role="tab"]:hover {
        background-color: #36516a !important;
        border-color: #89d7c1 !important;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #14614e !important;
        border-color: #39e2a7 !important;
        color: #ffffff !important;
        box-shadow: 0 0 0 1px #39e2a770, 0 0 15px #39e2a730 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-highlight"] {
        background: #39e2a7 !important;
        height: 3px !important;
    }
    @media (max-width: 768px) {
        .stDataFrame tbody td { font-size: 0.65rem !important; }
        h2 { font-size: 1.1rem !important; }
        div[data-testid="stTabs"] [role="tab"], .stTabs [role="tab"] {
            font-size: 14px !important; padding: 10px 12px !important;
            min-height: 46px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════════════════════

STYLE = dict(
    bg="#0d0f14",        panel="#13161e",      border="#1f2430",
    bull="#26a65b",      bear="#e04040",
    bull_fade="#26a65b55",                      bear_fade="#e0404055",
    mcg="#efb030",       ema200="#6060dd",
    text="#ffffff",      muted="#aaaaaa",
    verde="#2ca85e",     marron="#a06432",
    azul="#4488e0",      media_k="#ffffff",
    pvi="#6090e0",       pvi_ema="#efb030",
    macd_line="#6090e0", macd_sig="#efb030",
    rsi="#a78bfa",       adx="#ffffff",
    pdi="#26a65b",       ndi="#e04040",
    ao_up="#26a65b",     ao_dn="#e04040",
    vol="#3a6ea8",       vol_ma="#efb030",
    grid="#1a1e28",      zero="#2a2e3a",
    stoch_k="#a78bfa",   stoch_d="#efb030",
)

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["border"],
    "axes.labelcolor":  STYLE["muted"],
    "xtick.color":      "#ffffff",
    "ytick.color":      "#ffffff",
    "text.color":       STYLE["text"],
    "grid.color":       STYLE["grid"],
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "font.size":        9,
})


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def download_ohlcv(ticker: str, interval_key: str = "1D") -> pd.DataFrame:
    cfg = INTERVAL_CONFIG[interval_key]
    df  = yf.download(
        ticker, period=cfg["yf_period"], interval=cfg["yf_interval"],
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    df = clean_yf_df(df)
    if df.empty: return df
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum",
        }).dropna()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_dashboard(tickers_tuple: tuple):
    rows=[]
    errors=[]
    for ticker in tickers_tuple:
        try:
            r=scan_ticker(ticker)
            row={'Ticker':ticker, 'Nombre':TICKER_NAMES.get(ticker,ticker),
                 'PSM':r['state'], 'Barras':psm_bars(r['state']),
                 'Ultimo cierre':r['last_date']}
            kinds=[('DIV_ALCISTA','🟠 Divergencia'),('DIV_FANTASMA','🟠👻 Fantasma'),
                   ('CUNA_ROTA','▲ Cuña'),('PSM_LAA','🟣 LAA')]
            ages=[]; confirmed=[]
            for kind,label in kinds:
                e=latest_signal(r['events'],kind,25)
                row[label]=f"{e['age']}v · {e['date'][5:]}" if e else '—'
                if e:
                    ages.append(e['age'])
                    if kind!='DIV_FANTASMA': confirmed.append(e['age'])
            row['Ultima confirmada']=min(confirmed) if confirmed else 100000
            row['Ultima visible']=min(ages) if ages else 100000
            row['Coincidencias']=len(confirmed)
            rows.append(row)
        except Exception as exc:
            errors.append((ticker,str(exc)))
    return pd.DataFrame(rows),errors


@st.cache_data(ttl=3600, show_spinner=False)
def cached_chart_data(ticker: str, interval_key: str = "1D") -> dict:
    df = fetch_daily(ticker) if interval_key == "1D" else download_ohlcv(ticker, interval_key)
    if df.empty or len(df) < 60: return {}
    scan = scan_ticker(ticker, df=df) if interval_key == "1D" else None
    # Geometría V3 original para dibujar directrices. No cambia las señales.
    # Se calcula únicamente al abrir un gráfico diario (bajo caché).
    figuras_cunas = []
    if scan is not None:
        _, figuras_cunas, _ = detectar_cunas_v3(scan["df"], ticker)

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    mcg25     = mcginley_dynamic(close, 25)
    ema200    = EMAIndicator(close=close, window=200).ema_indicator()
    adx_ind   = ADXIndicator(high=high, low=low, close=close, window=14)
    adx_s     = adx_ind.adx()
    pdi_s     = adx_ind.adx_pos()
    ndi_s     = adx_ind.adx_neg()
    ao_s      = awesome_osc(high, low)
    bitman    = clasificar_bitman(df)
    div_df    = detectar_divergencia_simple(df)
    _, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)
    konc      = compute_blai5_koncorde(df, m=15)
    pvi_s     = calculate_pvi(close, volume)
    pvi_ema   = pvi_s.ewm(span=25, adjust=False).mean()
    vol_ma    = volume.rolling(20).mean()
    macd_obj  = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    rsi_s     = RSIIndicator(close=close, window=14).rsi()
    stoch_k, stoch_d = calc_stochastic_full(high, low, close, k_period=18, k_smooth=5, d_period=9)

    return dict(
        df=df, mcg25=mcg25, ema200=ema200,
        adx_s=adx_s, pdi_s=pdi_s, ndi_s=ndi_s,
        ao_s=ao_s, bitman=bitman, div_df=div_df,
        bbwp_s=bbwp_s, konc=konc,
        pvi_s=pvi_s, pvi_ema=pvi_ema, vol_ma=vol_ma,
        macd_line=macd_obj.macd(),
        macd_sig=macd_obj.macd_signal(),
        macd_hist=macd_obj.macd_diff(),
        rsi_s=rsi_s,
        stoch_k=stoch_k, stoch_d=stoch_d,
        scan=scan, figuras_cunas=figuras_cunas,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SYNC BAR HTML
# ══════════════════════════════════════════════════════════════════════════════

def sync_bar_html(fase_1d, embrion_1s, embrion_4h, spy_ok, small=False) -> str:
    sz = "12px" if not small else "10px"
    fs = "0.85rem" if not small else "0.75rem"

    def dot(aligned, label):
        if aligned:
            color = "#00e676"
            shadow = f"0 0 6px #00e67688"
            symbol = "●"
        else:
            color = "#444a5a"
            shadow = "none"
            symbol = "○"
        return (f"<span style='color:{color};font-size:{sz};"
                f"text-shadow:{shadow};margin-right:2px;'>{symbol}</span>"
                f"<span style='color:{color};font-size:{fs};font-weight:bold;"
                f"margin-right:12px;'>{label}</span>")

    phase_colors = {
        "EMBRION": "#00e676", "PRIMERAS SEÑALES": "#26a65b",
        "IMPULSO": "#efb030", "MOMENTUM MAXIMO": "#ff6b35",
        "MADURACION": "#888888", "DECLIVE": "#e04040",
        "CICLO INACTIVO": "#555555", "SIN DATOS": "#333333",
    }
    pc = phase_colors.get(fase_1d, "#555555")

    return (f"<div style='display:inline-flex;align-items:center;"
            f"background:#0e1018;border:1px solid #1f2430;border-radius:8px;"
            f"padding:4px 14px;gap:0;'>"
            f"<span style='color:{pc};font-size:{fs};font-weight:bold;margin-right:14px;'>"
            f"⏱ {fase_1d}</span>"
            f"{dot(True, '1D') if fase_1d == 'EMBRION' else dot(False, '1D')}"
            f"{dot(embrion_1s, '1S')}"
            f"{dot(embrion_4h, '4H')}"
            f"{dot(spy_ok, 'SPY')}"
            f"</div>")


def señal_badge_html(etiqueta: str) -> str:
    colors = {
        "⚡ BOOM ULTRA":      "#00e5ff",
        "🔥 BOOM":           "#00e676",
        "🟢 NO ESTÁ MUY MAL": "#66bb6a",
        "⚠️ SALIDA":         "#ff9800",
    }
    c = colors.get(etiqueta, "#555a6a")
    return (f"<span style='color:{c};font-weight:bold;"
            f"background:{c}22;padding:3px 12px;border-radius:99px;"
            f"border:1px solid {c}55;font-size:0.9rem;'>{etiqueta}</span>")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS GRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

def sv(series: pd.Series, index) -> np.ndarray:
    return series.reindex(index).values

def format_xaxis(ax, index, n_labels: int = 8):
    step   = max(1, len(index) // n_labels)
    ticks  = list(range(0, len(index), step))
    labels = [index[i].strftime("%d %b") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, color="#ffffff")

def panel_style(ax, ylabel="", yticks=5, zero_line=False):
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4, color="#ffffff")
    ax.yaxis.set_major_locator(plt.MaxNLocator(yticks, prune="both"))
    ax.tick_params(axis="both", labelsize=7, colors="#ffffff")
    ax.grid(True, axis="y", linewidth=0.4)
    ax.grid(True, axis="x", linewidth=0.2, alpha=0.4)
    if zero_line:
        ax.axhline(0, color=STYLE["zero"], linewidth=0.8, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SIGNALS (for Tab 2)
# ══════════════════════════════════════════════════════════════════════════════

def build_signals(data: dict) -> list:
    df      = data["df"]; konc = data["konc"]; close = df["Close"]
    mcg25   = data["mcg25"]; ema200 = data["ema200"]; rsi_s = data["rsi_s"]
    pvi_s   = data["pvi_s"]; pvi_ema = data["pvi_ema"]; macd_h = data["macd_hist"]
    macd_l  = data["macd_line"]; bitman = data["bitman"]; div_df = data["div_df"]
    p       = close.iloc[-1]

    def sig(label, bull, neutral=False):
        return {"label":label, "state":"neutral" if neutral else ("bull" if bull else "bear")}

    sigs = [
        sig(f"precio {'>' if p >= mcg25.iloc[-1] else '<'} MCG25",   p >= mcg25.iloc[-1]),
        sig(f"precio {'>' if p >= ema200.iloc[-1] else '<'} EMA200", p >= ema200.iloc[-1]),
    ]
    r = rsi_s.iloc[-1]
    sigs.append(sig(f"RSI {r:.1f}", r > 50, neutral=45 < r < 55))
    sigs.append(sig(f"MACD hist {'↑' if macd_h.iloc[-1] >= 0 else '↓'}", macd_h.iloc[-1] >= 0))
    sigs.append(sig(f"MACD línea {'≥0' if macd_l.iloc[-1] >= 0 else '<0'}", macd_l.iloc[-1] >= 0))
    # Conservamos el area azul de Koncorde como serie visual,
    # pero ya NO la utilizamos como señal de compra.
    if not konc.empty:
        sigs.append(sig(f"Verde {'>' if konc['verde'].iloc[-1] >= konc['marron'].iloc[-1] else '<'} M",
                        konc["verde"].iloc[-1] >= konc["marron"].iloc[-1]))
    scan=data.get('scan')
    if scan:
        state=scan['state']
        sigs.append(sig(f"PSM {state}",state in ('AAA','LAA'),
                        neutral=(state.startswith('L') and state!='LAA')))
    sigs.append(sig(f"PVI {'>' if pvi_s.iloc[-1] >= pvi_ema.iloc[-1] else '<'} EMA25",
                    pvi_s.iloc[-1] >= pvi_ema.iloc[-1]))
    a = data["adx_s"].iloc[-1]
    sigs.append(sig(f"ADX {a:.1f}", a > 25, neutral=18 < a < 25))
    if bitman is not None and not bitman.empty:
        b_etiq = bitman["Bitman_Etiqueta"].iloc[-1]
        b_v    = int(bitman["Bitman_Velas"].iloc[-1])
        sigs.append(sig(f"Bitman {b_etiq[:8]} ({b_v}v)",
                        b_etiq in ("IMPULSO ALCISTA","RETROCESO ALCISTA"),
                        neutral=(b_etiq == "INDEFINICIÓN")))
    if div_df is not None:
        dt = div_df["divergencia_tipo"].iloc[-1]
        if dt == "alcista":   sigs.append({"label":"Div RSI alc","state":"bull"})
        elif dt == "bajista": sigs.append({"label":"Div RSI baj","state":"bear"})

    sk = data["stoch_k"].iloc[-1]
    sd = data["stoch_d"].iloc[-1]
    stoch_bull = sk > sd and sk < 40
    sigs.append(sig(f"Stoch {sk:.0f}/{sd:.0f}", stoch_bull, neutral=40 <= sk <= 60))

    return sigs

def score_signals(sigs: list) -> tuple:
    bulls = sum(1 for s in sigs if s["state"] == "bull")
    bears = sum(1 for s in sigs if s["state"] == "bear")
    total = bulls + bears
    pct   = round(bulls / total * 100) if total else 0
    if pct >= 80:   label = "CONFLUENCIA MÁXIMA"
    elif pct >= 60: label = "SETUP SÓLIDO"
    elif pct >= 40: label = "SEÑALES MIXTAS"
    else:           label = "PRESIÓN BAJISTA"
    return pct, label, bulls, len(sigs)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE — memory-optimized
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(data: dict, ticker: str, n_candles: int = 200) -> bytes:
    """Build figure and return PNG bytes. No st.pyplot — avoids memory leak."""
    df        = data["df"]
    close     = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
    mcg25     = data["mcg25"]; ema200 = data["ema200"]
    adx_s     = data["adx_s"]; pdi_s = data["pdi_s"]; ndi_s = data["ndi_s"]
    ao_s      = data["ao_s"]; bitman = data["bitman"]; div_df = data["div_df"]
    bbwp_s    = data["bbwp_s"]; konc = data["konc"]
    pvi_s     = data["pvi_s"]; pvi_ema = data["pvi_ema"]; vol_ma = data["vol_ma"]
    macd_line = data["macd_line"]; macd_sig = data["macd_sig"]; macd_hist = data["macd_hist"]
    rsi_s     = data["rsi_s"]
    stoch_k   = data["stoch_k"]; stoch_d = data["stoch_d"]

    sigs                              = build_signals(data)
    pct, score_label, bull_n, total_n = score_signals(sigs)
    score_color = STYLE["bull"] if pct >= 60 else (STYLE["bear"] if pct < 40 else STYLE["mcg"])

    n_max   = min(n_candles, len(df))
    df_plot = df.iloc[-n_max:]
    idx     = df_plot.index
    xs      = np.arange(len(idx))

    # En 1D, mismas divergencias confirmadas que radar y Telegram.
    # En otros intervalos se conserva el detector auxiliar histórico.
    scan = data.get('scan')
    if scan is not None:
        shown = [(e, e['index']-(len(df)-n_max)) for e in scan['events']
                 if e['index'] >= len(df)-n_max]
        div_alc_xs = [x for e,x in shown if e['type']=='DIV_ALCISTA']
        div_baj_xs = []
        fant_xs = [x for e,x in shown if e['type']=='DIV_FANTASMA']
        wedge_xs = [x for e,x in shown if e['type']=='CUNA_ROTA']
        laa_xs = [x for e,x in shown if e['type']=='PSM_LAA']
    else:
        div_alc_xs, div_baj_xs, fant_xs, wedge_xs, laa_xs = [], [], [], [], []
        if div_df is not None:
            div_tipos_full = div_df['divergencia_tipo'].reindex(df.index).fillna('ninguna')
            for xi, dt in enumerate(div_tipos_full.iloc[-n_max:]):
                if dt=='alcista': div_alc_xs.append(xi)
                elif dt=='bajista': div_baj_xs.append(xi)

    # 10 paneles; nuevo modulo PSM entre RSI y ADX.
    fig = plt.figure(figsize=(14, 24), facecolor=STYLE['bg'])
    heights = [5,1.2,1.6,0.9,2,2.2,1.4,1.6,1.8,1.6]
    gs=gridspec.GridSpec(10,1,figure=fig,height_ratios=heights,hspace=0.37)
    fig.subplots_adjust(left=0.07,right=0.97,top=0.95,bottom=0.04)
    axes=[fig.add_subplot(gs[i]) for i in range(10)]
    for i in range(1,10): axes[i].tick_params(labelbottom=False)

    last_p  = close.iloc[-1]; prev_p = close.iloc[-2]
    chg     = last_p - prev_p; pct_chg = chg / prev_p * 100
    chg_c   = STYLE["bull"] if chg >= 0 else STYLE["bear"]
    sign    = "+" if chg >= 0 else ""

    nombre = TICKER_NAMES.get(ticker, "")
    title_str = f"{ticker}  ({nombre})" if nombre else ticker

    fig.text(0.07, 0.965, title_str,         fontsize=18, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.30, 0.965, f"{last_p:.2f}",fontsize=16, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.42, 0.965, f"{sign}{chg:.2f}  ({sign}{pct_chg:.2f}%)", fontsize=12, color=chg_c, va="bottom")
    fig.text(0.97, 0.965, f"{score_label}  ·  {bull_n}/{total_n}  ({pct}%)",
             fontsize=11, color=score_color, ha="right", va="bottom", style="italic")

    # PANEL 0 — Velas
    ax0 = axes[0]; w = 0.4
    for i, (_, row) in enumerate(df_plot.iterrows()):
        col = STYLE["bull"] if row["Close"] >= row["Open"] else STYLE["bear"]
        ax0.plot([i,i],[row["Low"],row["High"]], color=col, lw=0.8, zorder=2)
        ax0.add_patch(plt.Rectangle((i-w, min(row["Open"],row["Close"])),
            2*w, max(abs(row["Close"]-row["Open"]),0.001), color=col, zorder=3))
    ax0.plot(xs, sv(mcg25,idx),  color=STYLE["mcg"],   lw=1.4, label="MCG 25",  zorder=4)
    ax0.plot(xs, sv(ema200,idx), color=STYLE["ema200"], lw=1.4, label="EMA 200", zorder=4)
    if scan is not None:
        ax0.plot(xs,sv(scan['psm']['MIX35'],idx),color='#4491ff',
                 lw=1.5,label='PSM MIX35',zorder=5)
    # DIRECTRICES DE CUÑAS V3 — RECUPERADAS DEL MODELO ORIGINAL.
    # Lineas grises discontinuas: techo y suelo de las figuras detectadas.
    # Solo muestra cuñas activas o rupturas alcistas relevantes al zoom;
    # nunca modifica el algoritmo ni sus eventos/alertas Telegram.
    inicio_zoom = len(df) - n_max
    figuras_visibles = []
    for f in data.get('figuras_cunas', []):
        if f.get('tipo') != 'CUÑA':
            continue
        if f.get('estado') not in ('ACTIVA', 'ROTA_AL_ALZA'):
            continue
        fin_figura = int(f['fin']) if f.get('fin') is not None else len(df)-1
        if fin_figura < inicio_zoom or int(f['x0']) >= len(df):
            continue
        figuras_visibles.append((fin_figura, f))
    # Limitamos a las mas recientes para no saturar el precio.
    for _, f in sorted(figuras_visibles, key=lambda item: item[0])[-6:]:
        ini = max(int(f['x0']), inicio_zoom)
        fin = min(int(f['fin']) if f.get('fin') is not None else len(df)-1, len(df)-1)
        if fin < ini:
            continue
        xx_global = np.arange(ini, fin+1)
        xx_local = xx_global - inicio_zoom
        for m, b in ((f['mh'], f['bh']), (f['ml'], f['bl'])):
            yy = m * xx_global + b
            ax0.plot(xx_local, yy, color='#aab4c3', ls='--',
                     lw=1.15, alpha=0.70, zorder=3.4)

    low_arr=sv(low,idx); high_arr=sv(high,idx)
    # Divergencia confirmada: punto naranja vivo, sin cartel.
    for xi in div_alc_xs:
        if xi < len(low_arr) and np.isfinite(low_arr[xi]):
            ax0.scatter(xi, low_arr[xi] * 0.985, marker='o', s=90,
                        color='#ff9100', edgecolors='#fff1d8',
                        linewidths=0.75, zorder=12)
    for xi in div_baj_xs:
        if xi<len(high_arr) and not np.isnan(high_arr[xi]):
            ax0.annotate("▼ DIV BAJ", xy=(xi,high_arr[xi]), xytext=(0,14),
                         textcoords="offset points", fontsize=7, color=STYLE["bear"],
                         ha="center", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2",facecolor="#3d0000",edgecolor=STYLE["bear"],linewidth=0.8))
    # Marcadores oficiales en el grafico de precios.
    for xi in fant_xs:
        if np.isfinite(low_arr[xi]):
            ax0.scatter(xi,low_arr[xi]*0.985,s=43,facecolors='none',
                        edgecolors='#ffac56',linewidths=1.2,zorder=9)
    # Cuña: verde lima fluorescente, distinto al verde de las velas.
    for xi in wedge_xs:
        if np.isfinite(low_arr[xi]):
            ax0.scatter(xi, low_arr[xi] * 0.975, marker='^', s=125,
                        color='#b9ff35', edgecolors='#10251a',
                        linewidths=1.0, zorder=13)
    # LAA: morado habitual, aumento moderado de tamaño.
    for xi in laa_xs:
        if np.isfinite(low_arr[xi]):
            ax0.scatter(xi, low_arr[xi] * 0.985, marker='o', s=76,
                        color='#b46aff', edgecolors='#f6e9ff',
                        linewidths=0.7, zorder=11)
    ax0.set_xlim(-1,len(idx))
    ax0.legend(loc='upper left',fontsize=8,frameon=False,labelcolor='linecolor')
    panel_style(ax0,ylabel="Precio")
    ax0.set_title('Precio · McGinley 25 / MIX35 · EMA 200 · 🟠 DIV · ▲ Cuña · 🟣 LAA',fontsize=9,color=STYLE['muted'],loc='right',pad=4)
    format_xaxis(ax0,idx); ax0.tick_params(labelbottom=True,bottom=True)
    ax0.spines["bottom"].set_linewidth(1.2); ax0.spines["bottom"].set_edgecolor("#ffffff")

    # PANEL 1 — Volumen
    ax1=axes[1]
    vol_v=sv(volume,idx); vol_m=sv(vol_ma,idx)
    vol_colors=[STYLE["bull"] if df_plot["Close"].iloc[i]>=df_plot["Open"].iloc[i] else STYLE["bear"] for i in range(len(df_plot))]
    ax1.bar(xs,vol_v,color=vol_colors,alpha=0.6,width=0.8,zorder=2)
    ax1.fill_between(xs,vol_m,alpha=0.25,color=STYLE["vol_ma"],zorder=1)
    ax1.plot(xs,vol_m,color=STYLE["vol_ma"],lw=1.2,label="Vol MA20",zorder=3)
    ax1.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["vol_ma"]])
    panel_style(ax1,ylabel="Vol")
    ax1.set_title("Volumen  ·  MA 20",fontsize=9,color=STYLE["muted"],loc="right",pad=4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x/1e6:.0f}M" if x>=1e6 else f"{x/1e3:.0f}K"))

    # PANEL 2 — RSI
    ax2=axes[2]; rsi_v=sv(rsi_s,idx)
    ax2.fill_between(xs,rsi_v,70,where=(rsi_v>70),alpha=0.25,color=STYLE["bull"])
    ax2.fill_between(xs,rsi_v,30,where=(rsi_v<30),alpha=0.25,color=STYLE["bear"])
    ax2.plot(xs,rsi_v,color=STYLE["rsi"],lw=1.4)
    for lvl,col,ls in [(70,STYLE["bear"],"--"),(50,STYLE["muted"],":"),(30,STYLE["bull"],"--")]:
        ax2.axhline(lvl,color=col,lw=0.7,ls=ls)
    for xi in div_alc_xs:
        if xi<len(rsi_v) and not np.isnan(rsi_v[xi]):
            ax2.annotate("●",xy=(xi,rsi_v[xi]),fontsize=9,color="#ff9d3f",ha="center",va="top",
                         xytext=(0,-10),textcoords="offset points",fontweight="bold")
            ax2.axvline(xi,color="#ff9d3f",lw=0.6,ls=":",alpha=0.5)
    for xi in div_baj_xs:
        if xi<len(rsi_v) and not np.isnan(rsi_v[xi]):
            ax2.annotate("▼",xy=(xi,rsi_v[xi]),fontsize=9,color=STYLE["bear"],ha="center",va="bottom",
                         xytext=(0,10),textcoords="offset points",fontweight="bold")
            ax2.axvline(xi,color=STYLE["bear"],lw=0.6,ls=":",alpha=0.5)
    ax2.set_ylim(0,100); ax2.yaxis.set_ticks([30,50,70])
    panel_style(ax2,ylabel="RSI",yticks=3)
    ax2.set_title("RSI 14  ·  ▲ div alcista  ▼ div bajista",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 3 — PSM: tres franjas por sesion, debajo del RSI.
    axp=axes[3]
    if scan is not None:
        psm=scan['psm'].reindex(idx)
        for y,col in enumerate(['P','S','M']):
            values=psm[col].astype(str).tolist()
            colors=[{'A':'#26a65b','B':'#e04040','L':'#9da6b2'}.get(v,'#34383f')
                    for v in values]
            axp.bar(xs,[.91]*len(xs),bottom=2-y-.455,width=1.01,
                    color=colors,align='center',linewidth=0)
        axp.text(len(xs)-1,2.9,f"PSM {scan['state']}",fontsize=8,
                 color='#eeeeee',ha='right',va='top',
                 bbox=dict(facecolor='#222733',alpha=.8,edgecolor='none'))
        axp.set_yticks([2,1,0]); axp.set_yticklabels(['Principal','Secundaria','Micro'])
    else:
        axp.text(.5,.5,'PSM validado para temporalidad diaria (1D)',
                 transform=axp.transAxes,ha='center',va='center',color=STYLE['muted'])
        axp.set_yticks([])
    axp.set_ylim(-.5,3); axp.set_xlim(-1,len(xs))
    axp.tick_params(axis='x',bottom=False,labelbottom=False)
    axp.grid(False)
    for spine in axp.spines.values(): spine.set_edgecolor(STYLE['border'])
    axp.set_title('PSM · McGinley MIX35 · secundaria 3 ATR · micro 1.5 ATR',
                  fontsize=9,color=STYLE['muted'],loc='right',pad=3)

    # PANEL 3 — ADX + AO
    ax3=axes[4]; ax3r=ax3.twinx()
    ao_v=sv(ao_s,idx); ao_p=np.roll(ao_v,1); ao_p[0]=ao_v[0]
    ao_c=[STYLE["ao_up"] if ao_v[i]>=ao_p[i] else STYLE["ao_dn"] for i in range(len(ao_v))]
    ax3r.bar(xs,ao_v,color=ao_c,alpha=0.7,width=0.8,zorder=2)
    ax3r.axhline(0,color=STYLE["zero"],lw=0.7)
    ax3r.tick_params(labelsize=7,colors="#ffffff")
    ax3r.set_ylabel("AO",fontsize=7,color="#ffffff")
    ax3r.spines["right"].set_edgecolor(STYLE["border"])
    ax3.plot(xs,sv(adx_s,idx),color=STYLE["adx"],lw=1.6,label="ADX",zorder=3)
    ax3.plot(xs,sv(pdi_s,idx),color=STYLE["pdi"],lw=0.9,ls="--",label="+DI",zorder=3)
    ax3.plot(xs,sv(ndi_s,idx),color=STYLE["ndi"],lw=0.9,ls="--",label="-DI",zorder=3)
    ax3.axhline(25,color=STYLE["muted"],lw=0.6,ls=":")
    ax3.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["adx"],STYLE["pdi"],STYLE["ndi"]])
    panel_style(ax3,ylabel="ADX")
    ax3.set_title("ADX  ·  +DI / -DI  ·  Awesome Oscillator",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 4 — Koncorde
    ax4=axes[5]
    if not konc.empty:
        for key,col in [("verde",STYLE["verde"]),("marron",STYLE["marron"]),("azul",STYLE["azul"])]:
            v=sv(konc[key],idx)
            ax4.fill_between(xs,v,alpha=0.60,color=col,label=key.capitalize(),zorder=2)
            ax4.plot(xs,v,color=col,lw=1.0,zorder=3)
        ax4.plot(xs,sv(konc["media"],idx),color=STYLE["media_k"],lw=1.6,label="Media",zorder=4)
        ax4.legend(loc="upper left",fontsize=7,frameon=False,
                   labelcolor=[STYLE["verde"],STYLE["marron"],STYLE["azul"],STYLE["media_k"]])
    ax4.axhline(0,color=STYLE["zero"],lw=0.7)
    panel_style(ax4,ylabel="Koncorde")
    ax4.set_title("Blai5 Koncorde  ·  Verde / Marrón / Azul / Media",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 5 — BBWP
    ax5=axes[6]; bbwp_v=sv(bbwp_s,idx)
    ax5.fill_between(xs,bbwp_v,20,where=(~np.isnan(bbwp_v))&(bbwp_v<20),alpha=0.25,color=STYLE["azul"],zorder=1)
    ax5.fill_between(xs,bbwp_v,80,where=(~np.isnan(bbwp_v))&(bbwp_v>80),alpha=0.25,color=STYLE["bear"],zorder=1)
    for i in range(1,len(xs)):
        if np.isnan(bbwp_v[i]) or np.isnan(bbwp_v[i-1]): continue
        mid=(bbwp_v[i]+bbwp_v[i-1])/2
        lc=STYLE["azul"] if mid<20 else(STYLE["bear"] if mid>80 else STYLE["muted"])
        ax5.plot([xs[i-1],xs[i]],[bbwp_v[i-1],bbwp_v[i]],color=lc,lw=1.5,zorder=3)
    ax5.axhline(80,color=STYLE["bear"],lw=0.7,ls="--",alpha=0.6)
    ax5.axhline(20,color=STYLE["azul"],lw=0.7,ls="--",alpha=0.6)
    ax5.axhline(50,color=STYLE["muted"],lw=0.5,ls=":",alpha=0.4)
    ax5.set_ylim(-2,102); ax5.yaxis.set_ticks([0,20,50,80,100])
    panel_style(ax5,ylabel="BBWP")
    ax5.set_title("BBWP 13/252  ·  compresión<20  ·  expansión>80",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 6 — PVI
    ax6=axes[7]; pvi_v=sv(pvi_s,idx); pvi_e=sv(pvi_ema,idx)
    ax6.fill_between(xs,pvi_v,pvi_e,where=(pvi_v>=pvi_e),alpha=0.18,color=STYLE["bull"])
    ax6.fill_between(xs,pvi_v,pvi_e,where=(pvi_v<pvi_e), alpha=0.18,color=STYLE["bear"])
    ax6.plot(xs,pvi_v,color=STYLE["pvi"],    lw=1.4,label="PVI")
    ax6.plot(xs,pvi_e,color=STYLE["pvi_ema"],lw=1.4,ls="--",label="EMA 25")
    ax6.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["pvi"],STYLE["pvi_ema"]])
    panel_style(ax6,ylabel="PVI")
    ax6.set_title("PVI  ·  EMA 25",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 7 — MACD
    ax7=axes[8]; hist_v=sv(macd_hist,idx)
    hist_p=np.roll(hist_v,1); hist_p[0]=hist_v[0]
    bar_col=[]
    for i in range(len(hist_v)):
        v,p_v=hist_v[i],hist_p[i]
        if np.isnan(v): bar_col.append(STYLE["muted"]); continue
        bar_col.append(STYLE["bull"] if(v>=0 and v>=p_v) else
                       STYLE["bull_fade"] if v>=0 else
                       STYLE["bear"] if v<=p_v else STYLE["bear_fade"])
    ax7.bar(xs,hist_v,color=bar_col,width=0.8,alpha=0.9,zorder=2)
    ax7.plot(xs,sv(macd_line,idx),color=STYLE["macd_line"],lw=1.3,label="MACD",zorder=3)
    ax7.plot(xs,sv(macd_sig,idx), color=STYLE["macd_sig"], lw=1.3,ls="--",label="Señal",zorder=3)
    ax7.axhline(0,color=STYLE["zero"],lw=0.7)
    ax7.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["macd_line"],STYLE["macd_sig"]])
    panel_style(ax7,ylabel="MACD"); format_xaxis(ax7,idx)
    ax7.tick_params(labelbottom=True)
    ax7.set_title("MACD  12/26/9",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 8 — ESTOCÁSTICO (18, 5, 9)
    ax8=axes[9]; sk_v=sv(stoch_k,idx); sd_v=sv(stoch_d,idx)
    ax8.fill_between(xs,20,alpha=0.08,color=STYLE["bull"],zorder=0)
    ax8.fill_between(xs,80,alpha=0.08,color=STYLE["bear"],zorder=0)
    ax8.plot(xs,sk_v,color=STYLE["stoch_k"],lw=1.4,label="%K (18,5)",zorder=3)
    ax8.plot(xs,sd_v,color=STYLE["stoch_d"],lw=1.4,ls="--",label="%D (9)",zorder=3)
    ax8.axhline(80,color=STYLE["bear"],lw=0.7,ls="--",alpha=0.6)
    ax8.axhline(20,color=STYLE["bull"],lw=0.7,ls="--",alpha=0.6)
    ax8.axhline(50,color=STYLE["muted"],lw=0.5,ls=":",alpha=0.3)
    for i in range(1,len(sk_v)):
        if np.isnan(sk_v[i]) or np.isnan(sk_v[i-1]) or np.isnan(sd_v[i]) or np.isnan(sd_v[i-1]): continue
        if sk_v[i-1] <= sd_v[i-1] and sk_v[i] > sd_v[i] and sk_v[i] < 40:
            ax8.annotate("▲", xy=(i, sk_v[i]), fontsize=8, color="#00e676", ha="center",
                         fontweight="bold", xytext=(0,8), textcoords="offset points")
    ax8.set_ylim(-2,102); ax8.yaxis.set_ticks([0,20,40,60,80,100])
    ax8.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["stoch_k"],STYLE["stoch_d"]])
    panel_style(ax8,ylabel="Stoch"); format_xaxis(ax8,idx)
    ax8.tick_params(labelbottom=True)
    ax8.set_title("Estocástico  (18, 5, 9)  ·  ▲ cruce alcista zona 20-40",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    for ax in axes: ax.set_xlim(-1,len(idx))

    x0=0.07; gap=(0.97-x0)/len(sigs)
    for i,s in enumerate(sigs):
        col=(STYLE["bull"] if s["state"]=="bull" else STYLE["bear"] if s["state"]=="bear" else STYLE["muted"])
        fig.text(x0+i*gap+(gap-0.003)/2, 0.012, s["label"],
                 fontsize=7.5,color=col,ha="center",va="center",
                 bbox=dict(boxstyle="round,pad=0.3",facecolor=STYLE["panel"],edgecolor=col+"66",linewidth=0.8))

    # Save to PNG bytes and close figure immediately
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE STYLES
# ══════════════════════════════════════════════════════════════════════════════

SEÑAL_COLOR = {
    "⚡ BOOM ULTRA":      "#00e5ff",
    "🔥 BOOM":           "#00e676",
    "🟢 NO ESTÁ MUY MAL": "#66bb6a",
    "⚠️ SALIDA":         "#ff9800",
    "—":                 "#555a6a",
}
SALIDA_COLOR = {
    "🟢 MANTENER":           "#26a65b",
    "🟡 VIGILAR POSICIÓN":   "#efb030",
    "🟠 CONSIDERAR REDUCIR": "#e08030",
    "🔴 SALIDA":             "#e04040",
}

def color_señal(val): return f"color:{SEÑAL_COLOR.get(val,'#fff')};font-weight:bold"
def color_salida(val): return f"color:{SALIDA_COLOR.get(val,'#fff')};font-weight:bold"

def sync_cell(val):
    if val == "✅": return "color:#00e676;font-weight:bold;text-align:center"
    elif val == "❌": return "color:#e04040;font-weight:bold;text-align:center"
    return "color:#444a5a;text-align:center"

def style_df(df: pd.DataFrame):
    _s = df.style
    fn = _s.map if hasattr(_s,"map") else _s.applymap
    s  = fn(color_señal, subset=["Señal"])
    for col in ["1D","1S","4H","SPY"]:
        if col in df.columns:
            fn2 = s.map if hasattr(s,"map") else s.applymap
            s = fn2(sync_cell, subset=[col])
    return s.set_properties(**{
        "background-color":"#13161e","color":"#ffffff",
        "border-color":"#1f2430","font-size":"0.80rem",
    }).set_table_styles([{"selector":"th","props":[
        ("background-color","#0d0f14"),("color","#efb030"),
        ("font-size","0.82rem"),("border-bottom","1px solid #1f2430"),
    ]}])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Confluence Cucumber")
    st.markdown("---")
    grupo_sel    = st.selectbox("Grupo de tickers",options=list(GRUPOS.keys()),index=0,key="grupo")
    custom_input = st.text_area("Tickers personalizados\n(uno por línea)",height=120,key="custom_tickers")
    force_refresh_tab1 = st.button("🔄 Recalcular dashboard",key="refresh1")
    st.markdown("---")
    st.markdown("### 🎯 Filtros")
    solo_boom = st.checkbox("Solo señales activas (opción anterior)",value=False,key="solo_boom")
    st.markdown("---")
    st.markdown("<small style='color:#aaa'>Datos: Yahoo Finance · Caché 1h<br>"
                "Divergencias V2.1 + cuñas V3 + PSM LAA<br>"
                "SPY > EMA200 como contexto</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs([
    "📡  RADAR DE SEÑALES",
    "📈  GRÁFICOS ESTRATEGIA",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dashboard Señales
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## 📡 David L'Crack · Radar de señales")
    st.caption('🟠 Divergencia confirmada · 👻 Provisional · ▲ Cuña rota · 🟣 Nueva LAA · PSM actual (P/S/M)')
    tickers_tab1=([t.strip().upper() for t in custom_input.splitlines() if t.strip()]
                  if custom_input.strip() else GRUPOS[grupo_sel])
    tickers_tab1=list(dict.fromkeys(tickers_tab1))
    c1,c2,c3=st.columns([1,1,2])
    with c1:
        max_age=st.slider('Últimas velas',1,25,10,key='radar_age')
    with c2:
        mostrar_todos=st.checkbox('Incluir sin señal',False,key='radar_all')
    with c3:
        tipos=st.multiselect('Tipo de señal',
            ['🟠 Divergencia','🟠👻 Fantasma','▲ Cuña','🟣 LAA'],default=[],key='radar_types')
    if force_refresh_tab1: cached_dashboard.clear()
    with st.spinner(f'Escaneando {len(tickers_tab1)} tickers con motores V2.1 / V3 / PSM...'):
        result,errors=cached_dashboard(tuple(tickers_tab1))
    if result.empty:
        st.warning('Sin cotizaciones procesadas')
    else:
        cols=['🟠 Divergencia','🟠👻 Fantasma','▲ Cuña','🟣 LAA']
        frame=result.copy()
        for col in cols:
            frame.loc[frame[col].str.extract(r'^(\d+)')[0].astype(float)>max_age,col]='—'
        mask=(frame[cols]!='—').any(axis=1)
        if tipos: mask=mask & (frame[tipos]!='—').any(axis=1)
        if not mostrar_todos: frame=frame[mask].copy()
        # Prioridad: confirmado reciente; fantasma sólo después.
        def age(row):
            ages=[]
            for col in cols[:-1]+cols[-1:]:
                if col=='🟠👻 Fantasma': continue
                if row[col]!='—': ages.append(int(row[col].split('v')[0]))
            if ages: return (0,min(ages))
            return (1,int(row['🟠👻 Fantasma'].split('v')[0])
                    if row['🟠👻 Fantasma']!='—' else 9999)
        frame['_order']=frame.apply(age,axis=1)
        frame=frame.sort_values('_order').drop(columns='_order')
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric('Tickers seleccionados',len(tickers_tab1))
        for col,box in zip(cols,[m2,m3,m4,m5]):
            box.metric(col,int((frame[col]!='—').sum()))
        display_cols=['Ticker','Nombre']+cols+['Barras']
        st.dataframe(frame[display_cols],width='stretch',hide_index=True,
                     height=min(740,38+35*len(frame)))
        st.caption('PSM P/S/M: ⬜ lateral · 🟥 bajista · 🟩 alcista. Ordenado por la señal confirmada más reciente.')
        with st.expander('Estado actual y último cierre de todos los tickers'):
            st.dataframe(result[['Ticker','Nombre','PSM','Barras','Ultimo cierre']],
                         width='stretch',hide_index=True)
        if errors:
            with st.expander(f'Errores de descarga ({len(errors)})'):
                st.dataframe(pd.DataFrame(errors,columns=['Ticker','Error']),hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Gráficos (ON DEMAND — 1 ticker, render under button)
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("## 📈 Gráficos Estrategia")
    st.caption("Selecciona un ticker y pulsa 'Ver gráfico' para renderizar. Se genera 1 gráfico por vez para optimizar memoria.")

    col_a, col_b, col_c = st.columns([4, 2, 2])
    with col_a:
        chart_ticker = st.selectbox("📌 Ticker", options=sorted(ALL_TICKERS),
                                    index=sorted(ALL_TICKERS).index("BTC-USD") if "BTC-USD" in ALL_TICKERS else 0,
                                    key="chart_ticker")
    with col_b:
        chart_interval = st.selectbox("Intervalo",options=list(INTERVAL_CONFIG.keys()),index=0,key="chart_interval")
    with col_c:
        zoom_candles = st.selectbox("Zoom",options=[50,100,150,200,252],index=3,key="zoom_candles")

    if chart_interval in ("1h","4h"):
        st.warning(f"⚠️ Intervalo **{chart_interval}**: indicadores de largo plazo orientativos.")

    nombre = TICKER_NAMES.get(chart_ticker, "")
    titulo = f"{chart_ticker}  ({nombre})" if nombre else chart_ticker

    if st.button(f"📊 Ver gráfico de {chart_ticker}", key="render_chart", type="primary"):
        with st.spinner(f"Calculando {chart_ticker}…"):
            try:
                chart_data = cached_chart_data(chart_ticker, interval_key=chart_interval)
            except Exception as e:
                st.error(f"Error calculando **{chart_ticker}**: {e}")
                chart_data = {}

        if not chart_data:
            st.error(f"Sin datos para **{chart_ticker}**.")
        else:
            # Metrics row
            df_c = chart_data["df"]; close = df_c["Close"]
            m1,m2,m3,m4,m5 = st.columns(5)
            last_p = close.iloc[-1]; prev_p = close.iloc[-2]
            chg = last_p - prev_p; pct_c = chg / prev_p * 100
            m1.metric("Último precio",f"{last_p:.2f}",f"{chg:+.2f} ({pct_c:+.2f}%)")
            rsi_val = chart_data["rsi_s"].iloc[-1]
            m2.metric("RSI 14",f"{rsi_val:.1f}",
                      "Sobrecompra" if rsi_val>70 else("Sobreventa" if rsi_val<30 else "Neutral"))
            bbwp_v = chart_data["bbwp_s"].dropna()
            bbwp_l = bbwp_v.iloc[-1] if len(bbwp_v)>0 else np.nan
            m3.metric("BBWP 13/252",
                      f"{bbwp_l:.1f}%" if not np.isnan(bbwp_l) else "n/d",
                      "compresión" if bbwp_l<20 else("expansión" if bbwp_l>80 else "normal"))
            mcg_v = chart_data["mcg25"].iloc[-1]
            m4.metric("McGinley 25",f"{mcg_v:.2f}","↑ sobre MCG" if last_p>mcg_v else "↓ bajo MCG")
            e200_v = chart_data["ema200"].iloc[-1]
            m5.metric("EMA 200",f"{e200_v:.2f}","↑ sobre EMA" if last_p>e200_v else "↓ bajo EMA")

            sk_v = chart_data["stoch_k"].iloc[-1]
            sd_v = chart_data["stoch_d"].iloc[-1]
            st.caption(f"📈 Estocástico: %K={sk_v:.1f}  %D={sd_v:.1f}  "
                       f"{'🟢 Cruce alcista en zona baja' if sk_v > sd_v and sk_v < 40 else ''}")

            # Render chart to PNG bytes (no st.pyplot — saves memory)
            with st.spinner(f"Renderizando {chart_ticker}…"):
                png_bytes = build_figure(chart_data, chart_ticker, n_candles=zoom_candles)

            st.image(png_bytes, width="stretch")

            # Signal pills
            sigs = build_signals(chart_data)
            pct_s, label_s, bull_n, total_n = score_signals(sigs)
            cols_sig = st.columns(len(sigs))
            for col, s in zip(cols_sig, sigs):
                emoji = "🟢" if s["state"]=="bull" else("🔴" if s["state"]=="bear" else "⚪")
                col.markdown(f"<div style='text-align:center;font-size:.7rem;color:#fff'>{emoji}<br>{s['label']}</div>",
                             unsafe_allow_html=True)
            score_col = "green" if pct_s>=60 else("red" if pct_s<40 else "orange")
            st.markdown(f"<h4 style='color:{score_col};text-align:center'>{label_s}  ·  {bull_n}/{total_n}  ({pct_s}%)</h4>",
                        unsafe_allow_html=True)

            # Download button
            st.download_button(
                label=f"⬇️ Descargar PNG — {chart_ticker}",
                data=png_bytes,
                file_name=f"confluence_{chart_ticker.replace('=','').replace('-','_')}.png",
                mime="image/png",
                key=f"dl_{chart_ticker}_{chart_interval}")