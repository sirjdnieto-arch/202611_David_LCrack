# indicators.py
# ─────────────────────────────────────────────────────────────────────────────
# David Indicators Confluence — Core indicators & signal engine
#   · 6 main signals with freshness
#   · Multi-timeframe alignment (1D / 1S / 4H)
#   · Labels: BOOM ULTRA · BOOM · NO ESTÁ MUY MAL · SALIDA
#   · SPY > EMA200 context
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend   import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def download_df(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    return clean_yf_df(df)


def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    return df[needed].dropna().copy()


# ══════════════════════════════════════════════════════════════════════════════
# CORE INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def mcginley_dynamic(close: pd.Series, period: int = 25) -> pd.Series:
    k  = 0.6
    md = close.astype(float).copy()
    for i in range(1, len(close)):
        prev = md.iloc[i - 1]
        cur  = close.iloc[i]
        if prev == 0 or pd.isna(prev):
            md.iloc[i] = cur
        else:
            md.iloc[i] = prev + (cur - prev) / (k * period * (cur / prev) ** 4)
    return md


def calculate_pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    pvi = pd.Series(index=close.index, dtype=float)
    pvi.iloc[0] = 1000.0
    for i in range(1, len(close)):
        if volume.iloc[i] > volume.iloc[i - 1]:
            pct = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct)
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    return pvi


def calculate_nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    nvi = pd.Series(index=close.index, dtype=float)
    nvi.iloc[0] = 1000.0
    for i in range(1, len(close)):
        if volume.iloc[i] < volume.iloc[i - 1]:
            pct = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct)
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
    return nvi


def calc_mfi_blai5(high, low, close, volume, length: int = 14) -> pd.Series:
    src = (high + low + close) / 3.0
    up  = (volume * np.where(src.diff() > 0, src, 0)).rolling(length).sum()
    dn  = (volume * np.where(src.diff() < 0, src, 0)).rolling(length).sum()
    rs  = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_stoch(src, high, low, length: int = 21, smooth_fast_d: int = 3) -> pd.Series:
    ll = low.rolling(length).min()
    hh = high.rolling(length).max()
    k  = 100 * (src - ll) / (hh - ll)
    return k.rolling(smooth_fast_d).mean()


def awesome_osc(high: pd.Series, low: pd.Series) -> pd.Series:
    mid = (high + low) / 2.0
    return mid.rolling(5).mean() - mid.rolling(34).mean()


def calculate_bbwp(
    close: pd.Series,
    bb_len: int = 13,
    lookback: int = 252,
) -> tuple[pd.Series, pd.Series]:
    basis = close.rolling(bb_len).mean()
    dev   = close.rolling(bb_len).std(ddof=0)
    bbw   = 2.0 * dev / basis.replace(0, np.nan)
    arr   = bbw.values
    n     = len(arr)
    bbwp  = np.full(n, np.nan)
    for i in range(bb_len, n):
        cur = arr[i]
        if np.isnan(cur):
            continue
        start  = max(0, i - lookback)
        window = arr[start:i]
        valid  = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        bbwp[i] = np.sum(valid <= cur) / len(valid) * 100.0
    return pd.Series(bbw, index=close.index), pd.Series(bbwp, index=close.index)


def bbwp_signal(bbwp_pct, bbwp_series):
    if pd.isna(bbwp_pct):
        return "⚪", "normal", "→", "nan%"
    if bbwp_pct < 20:
        punto, zona = "🟢", "compresion"
    elif bbwp_pct > 80:
        punto, zona = "🔴", "expansion"
    else:
        punto, zona = "⚪", "normal"
    reciente = bbwp_series.dropna().iloc[-3:]
    if len(reciente) >= 2:
        slope = reciente.iloc[-1] - reciente.iloc[0]
        pendiente = "↑" if slope > 3 else ("↓" if slope < -3 else "→")
    else:
        pendiente = "→"
    return punto, zona, pendiente, f"{bbwp_pct:.1f}%"


# ── BLAI5 KONCORDE ────────────────────────────────────────────────────────────

def compute_blai5_koncorde(df: pd.DataFrame, m: int = 15) -> pd.DataFrame:
    df = clean_yf_df(df)
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    ohlc4 = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    pvi   = calculate_pvi(df["Close"], df["Volume"])
    nvi   = calculate_nvi(df["Close"], df["Volume"])
    pvim  = pvi.ewm(span=m, adjust=False).mean()
    nvim  = nvi.ewm(span=m, adjust=False).mean()
    oscp = (pvi - pvim) * 100 / (
        pvim.rolling(90).max() - pvim.rolling(90).min()
    ).replace(0, np.nan)
    azul = (nvi - nvim) * 100 / (
        nvim.rolling(90).max() - nvim.rolling(90).min()
    ).replace(0, np.nan)
    xmf     = calc_mfi_blai5(df["High"], df["Low"], df["Close"], df["Volume"], 14)
    basis   = ohlc4.rolling(25).mean()
    dev     = 2.0 * ohlc4.rolling(25).std()
    bollosc = ((ohlc4 - basis) / dev).replace([np.inf, -np.inf], np.nan) * 100
    xrsi    = RSIIndicator(close=ohlc4, window=14).rsi()
    stoc    = calc_stoch(ohlc4, df["High"], df["Low"], 21, 3)
    marron = (xrsi + xmf + bollosc + stoc / 3.0) / 2.0
    verde  = marron + oscp
    media  = marron.ewm(span=m, adjust=False).mean()
    out = pd.DataFrame(index=df.index)
    out["azul"]   = azul
    out["marron"] = marron
    out["verde"]  = verde
    out["media"]  = media
    return out


def blai5_signals(kdf: pd.DataFrame) -> pd.DataFrame:
    kdf      = kdf.copy()
    valid    = kdf[["verde", "marron", "azul", "media"]].notna().all(axis=1)
    area_max = kdf[["verde", "marron", "azul"]].max(axis=1)
    area_min = kdf[["verde", "marron", "azul"]].min(axis=1)
    inside   = valid & (kdf["media"] >= area_min) & (kdf["media"] <= area_max)
    punto_media, velas_konk = [], []
    estado, conteo = None, 0
    for i in range(len(kdf)):
        if not valid.iloc[i]:
            punto_media.append(False); velas_konk.append(0); continue
        if inside.iloc[i]:
            if estado != "inside": estado, conteo = "inside", 1
            else: conteo += 1
            punto_media.append(True)
        else:
            if estado != "outside": estado, conteo = "outside", 1
            else: conteo += 1
            punto_media.append(False)
        velas_konk.append(conteo)
    kdf["punto_media_verde"] = punto_media
    kdf["velas_konk"]        = velas_konk
    return kdf


# ── BITMAN ────────────────────────────────────────────────────────────────────

def clasificar_bitman(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 60:
        return pd.DataFrame()
    out     = df.copy()
    adx_ind = ADXIndicator(high=out["High"], low=out["Low"], close=out["Close"], window=14)
    out["ADX"]       = adx_ind.adx()
    out["ADX_Slope"] = out["ADX"].diff()
    out["AO"]        = awesome_osc(out["High"], out["Low"])
    diff             = out["AO"] - out["AO"].shift(1)
    out["AO_Color"]  = np.where(diff <= 0, "rojo", "verde")
    slope_mean_abs = out["ADX_Slope"].abs().rolling(20).mean()
    weak_thr       = (slope_mean_abs * 0.25).fillna(np.nan)
    out["ADX_Giro"]        = False
    out["Bitman_Color"]    = out["AO_Color"]
    out["Bitman_Etiqueta"] = "INDEFINICIÓN"
    out["Bitman_Velas"]    = 0
    last_turn_idx = None
    current_color = "verde"
    counter       = 0
    for i in range(len(out)):
        adx_slope_now = out["ADX_Slope"].iloc[i]
        ao_color_now  = out["AO_Color"].iloc[i]
        if (pd.isna(adx_slope_now) or pd.isna(weak_thr.iloc[i]) or
                abs(adx_slope_now) <= weak_thr.iloc[i]):
            counter += 1
            out.iloc[i, out.columns.get_loc("Bitman_Etiqueta")] = "INDEFINICIÓN"
            out.iloc[i, out.columns.get_loc("Bitman_Color")]    = ao_color_now
            out.iloc[i, out.columns.get_loc("Bitman_Velas")]    = counter
            continue
        adx_dir    = "impulso" if adx_slope_now > 0 else "retroceso"
        prev_slope = out["ADX_Slope"].iloc[i - 1] if i > 0 else np.nan
        prev_weak  = weak_thr.iloc[i - 1]          if i > 0 else np.nan
        giro = (i > 0
                and not pd.isna(prev_slope)
                and np.sign(prev_slope) != np.sign(adx_slope_now)
                and (pd.isna(prev_weak) or abs(prev_slope) > prev_weak))
        if giro:
            out.iloc[i, out.columns.get_loc("ADX_Giro")] = True
            last_turn_idx = i
            counter       = 1
            ao_w    = out["AO_Color"].iloc[max(0, i - 4): i + 1]
            changes = ao_w[ao_w != ao_w.shift(1)].dropna()
            if len(changes) >= 1:
                current_color = changes.iloc[-1]
        else:
            if last_turn_idx is not None and 0 < (i - last_turn_idx) <= 4:
                ao_w    = out["AO_Color"].iloc[last_turn_idx: i + 1]
                changes = ao_w[ao_w != ao_w.shift(1)].dropna()
                if len(changes) > 0:
                    current_color = changes.iloc[-1]
            counter = (i - last_turn_idx + 1) if last_turn_idx is not None else counter + 1
        etiqueta = (
            "IMPULSO ALCISTA"   if adx_dir == "impulso"   and current_color == "verde" else
            "IMPULSO BAJISTA"   if adx_dir == "impulso"   and current_color == "rojo"  else
            "RETROCESO ALCISTA" if adx_dir == "retroceso" and current_color == "verde" else
            "RETROCESO BAJISTA"
        )
        out.iloc[i, out.columns.get_loc("Bitman_Color")]    = current_color
        out.iloc[i, out.columns.get_loc("Bitman_Etiqueta")] = etiqueta
        out.iloc[i, out.columns.get_loc("Bitman_Velas")]    = counter
    return out


# ── DIVERGENCIAS RSI ─────────────────────────────────────────────────────────

def detectar_divergencia_simple(
    df: pd.DataFrame,
    lookback: int = 80,
    order: int = 3,
    max_gap: int = 20,
    tol: float = 0.005,
) -> pd.DataFrame:
    close = df["Close"].copy()
    rsi   = RSIIndicator(close=close, window=14).rsi()
    p, o  = close.values, rsi.values
    n     = len(p)

    def pivots(vals, kind="low"):
        idxs = []
        for i in range(order, n - order):
            if np.isnan(vals[i]): continue
            w = vals[i - order: i + order + 1]
            if kind == "low":
                if vals[i] <= np.nanmin(w) * (1 + tol) and vals[i] <= vals[i-1] and vals[i] <= vals[i+1]:
                    idxs.append(i)
            else:
                if vals[i] >= np.nanmax(w) * (1 - tol) and vals[i] >= vals[i-1] and vals[i] >= vals[i+1]:
                    idxs.append(i)
        return idxs

    def nearest(ref, cands):
        best, best_d = None, 10**9
        for c in cands:
            d = abs(c - ref)
            if d <= max_gap and d < best_d: best, best_d = c, d
        return best

    pl = pivots(p, "low");  ph = pivots(p, "high")
    ol = pivots(o, "low");  oh = pivots(o, "high")
    alc_idx = baj_idx = None
    for j in range(1, len(pl)):
        p1, p2 = pl[j-1], pl[j]
        if p2 - p1 > lookback: continue
        i1, i2 = nearest(p1, ol), nearest(p2, ol)
        if i1 is None or i2 is None or i1 == i2: continue
        if p[p2] < p[p1] and o[i2] > o[i1]: alc_idx = p2
    for j in range(1, len(ph)):
        p1, p2 = ph[j-1], ph[j]
        if p2 - p1 > lookback: continue
        i1, i2 = nearest(p1, oh), nearest(p2, oh)
        if i1 is None or i2 is None or i1 == i2: continue
        if p[p2] > p[p1] and o[i2] < o[i1]: baj_idx = p2

    if alc_idx is not None and baj_idx is not None:
        div_tipo, div_idx = ("alcista", alc_idx) if alc_idx >= baj_idx else ("bajista", baj_idx)
    elif alc_idx is not None:
        div_tipo, div_idx = "alcista", alc_idx
    elif baj_idx is not None:
        div_tipo, div_idx = "bajista", baj_idx
    else:
        div_tipo, div_idx = "ninguna", None

    out = df.copy()
    out["divergencia_tipo"] = "ninguna"
    out["divergencia"]      = "⚪"
    if div_idx is not None:
        out.iloc[div_idx, out.columns.get_loc("divergencia_tipo")] = div_tipo
        out.iloc[div_idx, out.columns.get_loc("divergencia")]      = (
            "🟢" if div_tipo == "alcista" else "🔴"
        )
    return out


# ── ESTOCÁSTICO COMPLETO (18, 5, 9) ─────────────────────────────────────────

def calc_stochastic_full(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 18, k_smooth: int = 5, d_period: int = 9,
) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k_raw = 100 * (close - ll) / (hh - ll)
    k_smoothed = k_raw.rolling(k_smooth).mean()
    d_line     = k_smoothed.rolling(d_period).mean()
    return k_smoothed, d_line


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _velas_desde_cruce(serie_bool: pd.Series) -> int:
    vals = serie_bool.fillna(False).values
    n    = len(vals)
    if n == 0 or not vals[-1]:
        return 999
    for i in range(n - 1, -1, -1):
        if not vals[i]:
            return (n - 1) - i
    return n


def azul_z_score(kdf: pd.DataFrame, window: int = 60) -> float:
    azul = kdf["azul"].dropna()
    if len(azul) < window + 4: return 0.0
    slope = azul.iloc[-1] - azul.iloc[-4]
    std   = azul.rolling(window).std().iloc[-1]
    if pd.isna(std) or std == 0: return 0.0
    return slope / std


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL SYSTEM v3 — 6 MAIN SIGNALS + FRESHNESS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_señales_principales(df: pd.DataFrame, umbral_frescura: int = 3) -> dict:
    close  = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]

    kdf = compute_blai5_koncorde(df, m=15)
    if kdf.empty: return _empty_señales()
    kdf = blai5_signals(kdf)

    bitman_df = clasificar_bitman(df)
    if bitman_df is None or bitman_df.empty: return _empty_señales()

    macd_obj  = MACD(close=close)
    macd_line = macd_obj.macd()
    macd_sig  = macd_obj.macd_signal()
    pvi_s     = calculate_pvi(close, volume)
    pvi_ema   = pvi_s.ewm(span=25, adjust=False).mean()
    ao_s      = awesome_osc(high, low)

    # S1: MACD ↑
    diff        = macd_line - macd_sig
    s1_activa   = diff.iloc[-1] > 0
    s1_cruce    = (diff > 0) & (diff.shift(1) <= 0)
    s1_frescura = _velas_desde_cruce(s1_cruce) if s1_activa else 999
    s1_fresca   = s1_activa and s1_frescura <= umbral_frescura

    # S2: PVI ↑
    s2_activa   = bool(pvi_s.iloc[-1] > pvi_ema.iloc[-1])
    s2_cruce    = (pvi_s > pvi_ema) & (pvi_s.shift(1) <= pvi_ema.shift(1))
    s2_frescura = _velas_desde_cruce(s2_cruce) if s2_activa else 999
    s2_fresca   = s2_activa and s2_frescura <= umbral_frescura

    # S3: Media K en área
    if not kdf.empty and all(c in kdf.columns for c in ["verde","marron","azul","media"]):
        area_max    = kdf[["verde","marron","azul"]].max(axis=1)
        area_min    = kdf[["verde","marron","azul"]].min(axis=1)
        media       = kdf["media"]
        en_area     = (media >= area_min) & (media <= area_max) & media.notna()
        s3_activa   = bool(en_area.iloc[-1])
        s3_cruce    = en_area & ~en_area.shift(1).fillna(False)
        s3_frescura = _velas_desde_cruce(s3_cruce) if s3_activa else 999
        s3_fresca   = s3_activa and s3_frescura <= umbral_frescura
    else:
        s3_activa = s3_fresca = False; s3_frescura = 999

    # S4: AO verde
    ao_verde     = ao_s > ao_s.shift(1)
    s4_activa    = bool(ao_verde.iloc[-1])
    s4_cruce     = ao_verde & ~ao_verde.shift(1).fillna(False)
    s4_frescura  = _velas_desde_cruce(s4_cruce) if s4_activa else 999
    s4_fresca    = s4_activa and s4_frescura <= umbral_frescura

    # S5: Bitman IMPULSO ALCISTA
    if not bitman_df.empty and "Bitman_Etiqueta" in bitman_df.columns:
        etiq          = bitman_df["Bitman_Etiqueta"]
        s5_activa     = bool(etiq.iloc[-1] == "IMPULSO ALCISTA")
        s5_cruce      = (etiq == "IMPULSO ALCISTA") & (etiq.shift(1) != "IMPULSO ALCISTA")
        s5_frescura   = _velas_desde_cruce(s5_cruce) if s5_activa else 999
        s5_fresca     = s5_activa and s5_frescura <= umbral_frescura
        s5_velas_ciclo = int(bitman_df["Bitman_Velas"].iloc[-1])
    else:
        s5_activa = s5_fresca = False; s5_frescura = 999; s5_velas_ciclo = 0

    # S6: Azul K cruza 0 ↑
    if not kdf.empty and "azul" in kdf.columns:
        azul          = kdf["azul"]
        s6_activa     = bool(azul.iloc[-1] > 0)
        s6_cruce      = (azul > 0) & (azul.shift(1) <= 0)
        s6_frescura   = _velas_desde_cruce(s6_cruce) if s6_activa else 999
        s6_fresca     = s6_activa and s6_frescura <= umbral_frescura
    else:
        s6_activa = s6_fresca = False; s6_frescura = 999

    # S7: Estocástico (18,5,9) %K > %D
    sk, sd = calc_stochastic_full(high, low, close, k_period=18, k_smooth=5, d_period=9)
    s7_activa   = bool(sk.iloc[-1] > sd.iloc[-1])
    s7_cruce    = (sk > sd) & (sk.shift(1) <= sd.shift(1))
    s7_frescura = _velas_desde_cruce(s7_cruce) if s7_activa else 999
    s7_fresca   = s7_activa and s7_frescura <= umbral_frescura

    señales = {
        "S1_MACD":  {"nombre": "MACD ↑",       "activa": s1_activa, "fresca": s1_fresca, "velas": s1_frescura if s1_activa else None},
        "S2_PVI":   {"nombre": "PVI ↑",         "activa": s2_activa, "fresca": s2_fresca, "velas": s2_frescura if s2_activa else None},
        "S3_MEDIA": {"nombre": "Media K área",  "activa": s3_activa, "fresca": s3_fresca, "velas": s3_frescura if s3_activa else None},
        "S4_AO":    {"nombre": "AO verde",      "activa": s4_activa, "fresca": s4_fresca, "velas": s4_frescura if s4_activa else None},
        "S5_BITMAN":{"nombre": "Bitman ↑",      "activa": s5_activa, "fresca": s5_fresca, "velas": s5_frescura if s5_activa else None},
        "S6_AZUL":  {"nombre": "Azul K ↑",      "activa": s6_activa, "fresca": s6_fresca, "velas": s6_frescura if s6_activa else None},
        "S7_STOCH": {"nombre": "Stoch %K>%D",   "activa": s7_activa, "fresca": s7_fresca, "velas": s7_frescura if s7_activa else None},
    }

    n_activas = sum(1 for s in señales.values() if s["activa"])
    n_frescas = sum(1 for s in señales.values() if s["fresca"])

    partes = []
    for key, s in señales.items():
        if s["activa"]:
            frescura_txt = f" 🔥{s['velas']}v" if s["fresca"] else ""
            partes.append(f"✅ {s['nombre']}{frescura_txt}")
        else:
            partes.append(f"❌ {s['nombre']}")

    informativas = []
    mcg25_val = mcginley_dynamic(close, 25).iloc[-1]
    precio    = close.iloc[-1]
    if abs(precio / mcg25_val - 1) < 0.012:
        informativas.append("🟡 Precio en soporte MCG25")
    e200_val = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]
    if abs(precio / e200_val - 1) < 0.015:
        informativas.append("🟡 Precio en soporte EMA200")
    if not kdf.empty and "azul" in kdf.columns and "verde" in kdf.columns:
        if kdf["azul"].iloc[-1] > 0 and kdf["verde"].iloc[-1] < 0:
            informativas.append("⚠️ Atención Konkorde (azul+ verde-)")
    rsi_div = detectar_divergencia_simple(df)
    hits    = rsi_div[rsi_div["divergencia_tipo"] == "alcista"]
    if not hits.empty:
        div_idx   = rsi_div.index.get_loc(hits.index[-1])
        div_velas = len(rsi_div) - 1 - div_idx
        if div_velas <= 15:
            informativas.append(f"🟢 Div alcista RSI ({div_velas}v)")
    partes.extend(informativas)

    return {
        "señales": señales, "n_activas": n_activas, "n_frescas": n_frescas,
        "detalle": "  |  ".join(partes),
        "kdf": kdf, "bitman_df": bitman_df,
    }


def _empty_señales() -> dict:
    return {"señales": {}, "n_activas": 0, "n_frescas": 0,
            "detalle": "Sin datos suficientes",
            "kdf": pd.DataFrame(), "bitman_df": pd.DataFrame()}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE & LABEL
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_fase_v3(n_activas: int, n_frescas: int, n_activas_prev: int = 0) -> str:
    if n_activas >= 6 and n_frescas >= 3:   return "MOMENTUM MAXIMO"
    if n_activas >= 6 and n_frescas < 2:    return "MADURACION"
    if n_activas >= 5 and n_frescas >= 1:   return "IMPULSO"
    if 3 <= n_activas <= 4 and n_frescas >= 1: return "PRIMERAS SEÑALES"
    if 1 <= n_activas <= 2 and n_frescas >= 1: return "EMBRION"
    if n_activas_prev >= 4 and n_activas <= 2: return "DECLIVE"
    return "CICLO INACTIVO"


def calcular_fase_timeframe(ticker: str, interval_key: str = "1D") -> str:
    try:
        if interval_key == "1D":
            df = download_df(ticker, period="2y", interval="1d")
        elif interval_key == "1S":
            df = download_df(ticker, period="5y", interval="1wk")
        elif interval_key == "4H":
            df = download_df(ticker, period="60d", interval="1h")
            if not df.empty:
                df = df.resample("4h").agg({
                    "Open":"first","High":"max","Low":"min",
                    "Close":"last","Volume":"sum",
                }).dropna()
        else:
            return "SIN DATOS"
        if df.empty or len(df) < 150:
            return "SIN DATOS"
        sig = calcular_señales_principales(df)
        if not sig["señales"]: return "SIN DATOS"
        return clasificar_fase_v3(sig["n_activas"], sig["n_frescas"])
    except Exception:
        return "SIN DATOS"


def calcular_alineacion(ticker: str, fase_1d_known: str = None) -> dict:
    fase_1d = fase_1d_known if fase_1d_known else calcular_fase_timeframe(ticker, "1D")
    fase_1s = calcular_fase_timeframe(ticker, "1S")
    fase_4h = calcular_fase_timeframe(ticker, "4H")
    embrion_1d = fase_1d == "EMBRION"
    embrion_1s = fase_1s in ("EMBRION", "PRIMERAS SEÑALES")
    embrion_4h = fase_4h in ("EMBRION", "PRIMERAS SEÑALES")
    return {
        "fase_1d": fase_1d, "fase_1s": fase_1s, "fase_4h": fase_4h,
        "embrion_1d": embrion_1d, "embrion_1s": embrion_1s, "embrion_4h": embrion_4h,
    }


def get_spy_context() -> dict:
    try:
        df = download_df("SPY", period="2y", interval="1d")
        if df.empty or len(df) < 200:
            return {"spy_above_ema200": None, "spy_price": 0, "ema200": 0}
        ema200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        price  = df["Close"].iloc[-1]
        return {"spy_above_ema200": price > ema200, "spy_price": float(price), "ema200": float(ema200)}
    except Exception:
        return {"spy_above_ema200": None, "spy_price": 0, "ema200": 0}


def asignar_etiqueta(fase_1d: str, embrion_1s: bool, embrion_4h: bool, spy_ok: bool) -> str:
    if fase_1d in ("MADURACION", "DECLIVE"):
        return "⚠️ SALIDA"
    if fase_1d != "EMBRION":
        return ""
    if embrion_1s and embrion_4h:
        return "⚡ BOOM ULTRA"
    if embrion_1s and spy_ok:
        return "🔥 BOOM"
    return "🟢 NO ESTÁ MUY MAL"


# ══════════════════════════════════════════════════════════════════════════════
# SEMÁFORO DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

def semaforo_salida(df, kdf, bitman_df, macd_line, macd_sig, pvi_s, pvi_ema) -> dict:
    resultado = {"señales": {}, "n_salida": 0, "etiqueta": "🟢 MANTENER", "razones": ""}
    high = df["High"]; low = df["Low"]
    if len(df) < 10: return resultado

    ao_s      = awesome_osc(high, low)
    ao_verde  = ao_s > ao_s.shift(1)
    se1 = bool(not ao_verde.iloc[-1] and ao_verde.iloc[-2])
    resultado["señales"]["SE1_AO"] = {"nombre": "AO verde→rojo", "activa": se1}

    diff = macd_line - macd_sig
    se2 = bool(diff.iloc[-1] < 0 and diff.iloc[-2] >= 0)
    resultado["señales"]["SE2_MACD"] = {"nombre": "MACD cruce ↓", "activa": se2}

    se3 = bool(pvi_s.iloc[-1] < pvi_ema.iloc[-1] and pvi_s.iloc[-2] >= pvi_ema.iloc[-2])
    resultado["señales"]["SE3_PVI"] = {"nombre": "PVI cruza EMA25 ↓", "activa": se3}

    if bitman_df is not None and not bitman_df.empty and len(bitman_df) >= 2:
        etiq_now  = bitman_df["Bitman_Etiqueta"].iloc[-1]
        etiq_prev = bitman_df["Bitman_Etiqueta"].iloc[-2]
        se4 = bool(etiq_now != "IMPULSO ALCISTA" and etiq_prev == "IMPULSO ALCISTA")
    else:
        se4 = False
    resultado["señales"]["SE4_BITMAN"] = {"nombre": "Bitman pierde impulso", "activa": se4}

    n_salida = sum(1 for s in resultado["señales"].values() if s["activa"])
    resultado["n_salida"] = n_salida
    if n_salida == 0:   etiqueta = "🟢 MANTENER"
    elif n_salida == 1: etiqueta = "🟡 VIGILAR POSICIÓN"
    elif n_salida == 2: etiqueta = "🟠 CONSIDERAR REDUCIR"
    else:               etiqueta = "🔴 SALIDA"
    resultado["etiqueta"] = etiqueta

    partes = []
    for s in resultado["señales"].values():
        partes.append(f"🔴 {s['nombre']}" if s["activa"] else f"🟢 {s['nombre']} ok")
    resultado["razones"] = "  |  ".join(partes)
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# CONFLUENCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def get_confluence_dashboard(tickers: list, progress_cb=None) -> pd.DataFrame:
    from config import TICKER_NAMES

    report = []
    total  = len(tickers)
    spy_ctx = get_spy_context()
    spy_ok  = spy_ctx["spy_above_ema200"] or False

    for idx_t, t in enumerate(tickers):
        if progress_cb:
            progress_cb(idx_t, total, t)
        try:
            df = download_df(t, period="2y", interval="1d")
            if df.empty or len(df) < 150: continue

            close  = df["Close"]; volume = df["Volume"]; precio = close.iloc[-1]

            sig = calcular_señales_principales(df)
            if not sig["señales"]: continue

            fase_1d = clasificar_fase_v3(sig["n_activas"], sig["n_frescas"])

            # Add nuance to phase display: show if signals are active even in INACTIVO
            if fase_1d == "CICLO INACTIVO" and sig["n_activas"] >= 3:
                fase_display = f"INACTIVO ({sig['n_activas']}/7 act.)"
            else:
                fase_display = fase_1d

            mcg25_val = mcginley_dynamic(close, 25).iloc[-1]
            e200_val  = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]
            cerca_mcg  = mcg25_val * 0.988 <= precio <= mcg25_val * 1.012
            cerca_e200 = e200_val  * 0.985 <= precio <= e200_val  * 1.015
            s_mcg  = "🟡" if cerca_mcg  else ("🟢" if precio > mcg25_val else "🔴")
            s_e200 = "🟡" if cerca_e200 else ("🟢" if precio > e200_val  else "🔴")

            # Only calculate multi-TF alignment for EMBRION (saves massive memory/time)
            # For other phases, alignment doesn't change the label
            if fase_1d == "EMBRION":
                alineacion = calcular_alineacion(t, fase_1d_known=fase_1d)
            else:
                alineacion = {
                    "fase_1d": fase_1d, "fase_1s": "—", "fase_4h": "—",
                    "embrion_1d": False, "embrion_1s": False, "embrion_4h": False,
                }

            etiqueta = asignar_etiqueta(fase_1d, alineacion["embrion_1s"], alineacion["embrion_4h"], spy_ok)
            nombre = TICKER_NAMES.get(t, t)

            report.append({
                "Ticker":    t,
                "Nombre":    nombre,
                "Precio":    f"{precio:.2f}",
                "Tendencia": f"MCG:{s_mcg} E200:{s_e200}",
                "Fase 1D":   fase_display,
                "1D":        "✅" if alineacion["embrion_1d"] else "—",
                "1S":        "✅" if alineacion["embrion_1s"] else "—",
                "4H":        "✅" if alineacion["embrion_4h"] else "—",
                "SPY":       "✅" if spy_ok else "❌",
                "Señal":     etiqueta if etiqueta else "—",
                "Activas":   f"{sig['n_activas']}/7",
                "Frescas":   f"{sig['n_frescas']}/7",
                "Detalle":   sig["detalle"],
            })

        except Exception as e:
            print(f"❌ {t}: {e}")
            continue

    result = pd.DataFrame(report)
    if not result.empty:
        orden = {"⚡ BOOM ULTRA": 0, "🔥 BOOM": 1, "🟢 NO ESTÁ MUY MAL": 2, "⚠️ SALIDA": 3, "—": 4}
        result["_sort"] = result["Señal"].map(orden).fillna(5)
        result = (result.sort_values(["_sort", "Ticker"])
                        .drop(columns="_sort")
                        .reset_index(drop=True))
    return result