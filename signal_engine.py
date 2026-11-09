# -*- coding: utf-8 -*-
"""ESCANER ESTRUCTURAL 3 SEÑALES — Divergencias V2.1 + LAA PSM + Cuñas V3.

Colab: !pip -q install yfinance scipy pandas numpy tqdm
       %run scanner_david_cunas_psm.py

Detecta divergencias alcistas CONFIRMADAS, fantasmas vivos, nueva LAA del
PSM confirmado y rupturas alcistas de CUÑAS bajistas según motor V3 disponible.
Informa además del estado PSM actual de TODOS los activos descargados.
No filtra entradas por confluencia; solo las describe.
"""
import os
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings('ignore')

# -------------------- CONFIGURACION --------------------
# Ventana para considerar una señal como reciente (sesiones del propio ticker)
VELAS_RECIENTES = 25
# Ultimos años de datos para el escaner autonomo.
# >=3 años cubre >220 velas de vida máxima de figura para las alertas recientes.
PERIODO = '5y'  # Más precalentamiento para la McGinley recursiva
INTERVALO = '1d'

# Si proporcionas CSV original actualizado, se usarán sus rupturas en lugar
# de recalcular cuñas. Debe incluir Ticker, Tipo, Tendencia, Fecha.
# Deja None para detectar las cuñas con el motor V3 disponible.
CSV_CUNAS_ORIGINAL = None

# Fantasmas alcistas: provisionales, NO equivalentes a señales confirmadas.
MOSTRAR_FANTASMAS = True
# Mostrar activos con al menos 1 señal; primera tabla, ordenados por coincidencias.
MOSTRAR_TODOS_LOS_ACTIVOS_CON_SENAL = True

# Parámetros del detector de divergencias V2.1 (no optimizados)
PARAMS = dict(
    rsi14_len=14, atr_len=14,
    pivot_left=3, pivot_right=3,
    min_barras=5, max_barras=50,
    min_rsi_diff=3.0, min_price_atr=0.25,
    nivel_sobreventa=30.0, mostrar_fantasmas=MOSTRAR_FANTASMAS,
)

# PSM LAA CONFIRMADO. Configuración estructural congelada.
PSM_N_MIX = 35
PSM_N_MIN = 15
PSM_CAP_RATIO = 1.5
PSM_K = 0.6
PSM_UMBRAL_P = 0.3655
PSM_ATR_SEC = 3.0
PSM_ATR_MICRO = 1.5

# Parametros congelados del detector V3 facilitado en el proyecto.
L, R, ATR_N = 5, 5, 14
MIN_DURACION, MAX_DURACION, MIN_SEP = 25, 180, 8
TOL_PIVOTE_ATR, TOL_CIERRE_ATR = .70, .50
DESCENSO_MIN_ATR, ANCHURA_MIN_ATR = .70, .65
CANAL_MIN_CONTRACCION, CANAL_MAX_CONTRACCION, CUNA_MAX_CONTRACCION = -.25, .25, .80
MAX_SESIONES_SIN_CONTACTO, MAX_VIDA_DESDE_ORIGEN = 100, 220
PIVOTES_SEMILLA, MAX_SEMILLAS_POR_FECHA = 10, 2

# Lista original de los 151 tickers del scanner de divergencias.
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'META', 'BRK-B', 'TSLA', 'JNJ', 'V', 'PG', 'XOM', 'UNH', 'JPM', 'HD', 'LLY', 'MA', 'CVX', 'ABBV', 'KO', 'PEP', 'COST', 'BAC', 'CRM', 'NFLX', 'ABT', 'MCD', 'LMT', 'EL', 'NEE', 'CAT', 'MRK', 'TPL', 'ASML', 'ADBE', 'AVGO', 'CSCO', 'CMCSA', 'AMD', 'TXN', 'QCOM', 'AMAT', 'LITE', 'LRCX', 'COHR', 'CMI', 'NEM', 'ULTA', 'IT', 'FOXA', 'LUV', 'VLO', 'ADP', 'FN', 'POET', 'KEYS', 'HPE', 'MRVL', 'BRKR', 'AAOI', 'INTU', 'VRTX', 'ZS', 'PLTR', 'MU', 'ON', 'MELI', 'CTSH', 'QBTS', 'RGTI', 'IONQ', 'PDD', 'NIO', 'TCEHY', 'BZUN', 'FUTU', 'MOMO', 'MNSO', 'TAL', 'EDU', 'WB', 'XPEV', 'CRCL', 'PYPL', 'PGR', 'LULU', 'GDDY', 'ZTS', 'ADSK', 'BKNG', 'GEN', 'EXPE', 'RMD', 'CPAY', 'EFX', 'NU', 'RGLD', 'TOST', 'WISE', 'TEL', 'CSU.TO', 'LVMUY', 'SAP', 'OR.PA', 'TTE', 'MC.PA', 'ENGI.PA', 'AIR.PA', 'EL.PA', 'AI.PA', 'BNP.PA', 'SAN.PA', 'KER.PA', 'SU.PA', 'SIE.DE', 'ALV.DE', 'LIN.DE', 'VOW3.DE', 'BMW.DE', 'ADS.DE', 'IFX.DE', 'MUV2.DE', 'FRE.DE', 'DTE.DE', 'RWE.DE', 'SAP.DE', 'NESN.SW', 'ITX.MC', 'BBVA.MC', 'SAN.MC', 'TEF.MC', 'IBE.MC', 'REP.MC', 'FER.MC', 'ACX.MC', 'ACS.MC', 'AENA.MC', 'ANA.MC', 'IAG.MC', 'LOG.MC', 'MAP.MC', 'PUIG.MC', 'NTGY.MC', 'ELE.MC', 'IDR.MC', 'GC=F', 'SI=F', 'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']

# ========== FUNCIONES ORIGINALES DEL V2.1 ==========
def rma_pine(series, length):
    """
    Aproximación explícita a ta.rma() de Pine.

    Inicialización:
        SMA de los primeros 'length' valores válidos.

    Después:
        RMA_t =
            (RMA_(t-1) * (length - 1) + x_t)
            / length
    """

    s = pd.Series(series, copy=True).astype(float)

    values = s.to_numpy(dtype=float)

    out = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    valid_positions = np.flatnonzero(
        np.isfinite(values)
    )

    if len(valid_positions) < length:
        return pd.Series(
            out,
            index=s.index
        )

    seed_positions = valid_positions[:length]

    seed_pos = seed_positions[-1]

    seed = np.mean(
        values[seed_positions]
    )

    out[seed_pos] = seed

    prev = seed


    for i in range(
        seed_pos + 1,
        len(values)
    ):

        x = values[i]

        if not np.isfinite(x):

            out[i] = np.nan

            continue


        prev = (
            prev * (length - 1)
            +
            x
        ) / length

        out[i] = prev


    return pd.Series(
        out,
        index=s.index
    )

def calculate_rsi_pine(series, length):

    s = pd.Series(
        series,
        copy=True
    ).astype(float)


    delta = s.diff()


    gain = delta.clip(
        lower=0
    )


    loss = (
        -delta
    ).clip(
        lower=0
    )


    avg_gain = rma_pine(
        gain,
        length
    )


    avg_loss = rma_pine(
        loss,
        length
    )


    rsi = pd.Series(
        np.nan,
        index=s.index,
        dtype=float
    )


    normal = (
        (avg_loss > 0)
        &
        (avg_gain > 0)
    )


    rsi.loc[normal] = (
        100.0
        -
        100.0
        /
        (
            1.0
            +
            avg_gain.loc[normal]
            /
            avg_loss.loc[normal]
        )
    )


    solo_subidas = (
        (avg_loss == 0)
        &
        (avg_gain > 0)
    )

    rsi.loc[solo_subidas] = 100.0


    solo_bajadas = (
        (avg_gain == 0)
        &
        (avg_loss > 0)
    )

    rsi.loc[solo_bajadas] = 0.0


    plano = (
        (avg_gain == 0)
        &
        (avg_loss == 0)
    )

    rsi.loc[plano] = 50.0


    return rsi

def calculate_atr_pine(df, length=14):

    prev_close = (
        df["Close"]
        .shift(1)
    )


    tr = pd.concat(

        [

            df["High"]
            -
            df["Low"],


            (
                df["High"]
                -
                prev_close
            ).abs(),


            (
                df["Low"]
                -
                prev_close
            ).abs(),

        ],

        axis=1

    ).max(
        axis=1,
        skipna=True
    )


    return rma_pine(
        tr,
        length
    )

def pivot_low_pine_like(
    values,
    center,
    left,
    right
):

    if (
        center - left < 0
        or
        center + right >= len(values)
    ):

        return False


    c = values[center]


    if not np.isfinite(c):

        return False


    izquierda = values[
        center-left:center
    ]


    derecha = values[
        center+1:center+right+1
    ]


    if (
        not np.all(
            np.isfinite(izquierda)
        )
        or
        not np.all(
            np.isfinite(derecha)
        )
    ):

        return False


    return (
        np.all(
            c <= izquierda
        )
        and
        np.all(
            c < derecha
        )
    )

def buscar_referencia_alcista(
    pivots,
    curr_idx,
    curr_price,
    curr_rsi,
    curr_atr,
    params
):
    """
    Busca desde el pivote más reciente hacia atrás.

    Un mínimo antiguo sólo sigue siendo estructural si
    está por debajo de TODOS los mínimos intermedios.

    Se devuelve la PRIMERA divergencia válida.
    """

    if (
        not np.isfinite(curr_price)
        or
        not np.isfinite(curr_rsi)
        or
        not np.isfinite(curr_atr)
        or
        curr_atr <= 0
    ):

        return None


    min_bars = params["min_barras"]
    max_bars = params["max_barras"]

    min_rsi_diff = params["min_rsi_diff"]
    min_price_atr = params["min_price_atr"]

    nivel_os = params["nivel_sobreventa"]


    minimo_intermedio = None


    for ref in reversed(pivots):

        ref_idx = ref["idx"]

        ref_price = ref["price"]

        ref_rsi = ref["rsi"]


        barras_entre = (
            curr_idx
            -
            ref_idx
        )


        if barras_entre > max_bars:

            break


        referencia_estructural = (

            minimo_intermedio is None

            or

            ref_price
            <
            minimo_intermedio

        )


        if (
            referencia_estructural
            and
            barras_entre >= min_bars
            and
            np.isfinite(ref_rsi)
        ):

            diferencia_precio = (
                ref_price
                -
                curr_price
            )


            diferencia_rsi = (
                curr_rsi
                -
                ref_rsi
            )


            estructura_precio = (
                curr_price
                <
                ref_price
            )


            estructura_rsi = (
                diferencia_rsi
                >=
                min_rsi_diff
            )


            precio_ok = (
                diferencia_precio
                >=
                min_price_atr
                *
                curr_atr
            )


            sobreventa_ok = (
                ref_rsi <= nivel_os
                or
                curr_rsi <= nivel_os
            )


            divergencia_valida = (

                estructura_precio

                and

                estructura_rsi

                and

                precio_ok

                and

                sobreventa_ok

            )


            if divergencia_valida:

                return {

                    "ref_idx":
                        int(ref_idx),

                    "ref_price":
                        float(ref_price),

                    "ref_rsi":
                        float(ref_rsi),

                    "bars_between":
                        int(barras_entre),

                    "rsi_diff":
                        float(diferencia_rsi),

                    "price_atr":
                        float(
                            diferencia_precio
                            /
                            curr_atr
                        ),
                }


        if minimo_intermedio is None:

            minimo_intermedio = (
                ref_price
            )

        else:

            minimo_intermedio = min(
                minimo_intermedio,
                ref_price
            )


    return None

def crear_evento(
    direction,
    state,
    signal_idx,
    pivot_idx,
    curr_price,
    curr_rsi,
    referencia
):

    return {

        "direction":
            direction,

        "state":
            state,

        # ----------------------------------------------------
        # Día en que conocemos / vemos la señal
        # ----------------------------------------------------

        "signal_idx":
            int(signal_idx),

        # ----------------------------------------------------
        # Segundo extremo estructural
        # ----------------------------------------------------

        "pivot_idx":
            int(pivot_idx),

        # ----------------------------------------------------
        # Primer extremo estructural elegido por B estructural
        # ----------------------------------------------------

        "p1_idx":
            int(
                referencia["ref_idx"]
            ),

        "p1_price":
            float(
                referencia["ref_price"]
            ),

        "p2_price":
            float(
                curr_price
            ),

        "rsi1":
            float(
                referencia["ref_rsi"]
            ),

        "rsi2":
            float(
                curr_rsi
            ),

        "bars_between":
            int(
                referencia["bars_between"]
            ),

        "rsi_diff":
            float(
                referencia["rsi_diff"]
            ),

        "price_atr":
            float(
                referencia["price_atr"]
            ),
    }

# ========== MOTOR GEOMETRICO V3 SIN ALTERAR ==========
def detectar(d, ticker):
    o,h,l,c,a = [d[k].to_numpy(dtype=float) for k in ('Open','High','Low','Close','ATR')]
    n=len(d)
    conf={}
    for i in range(L,n-R):
        wh=h[i-L:i+R+1]; wl=l[i-L:i+R+1]
        if h[i]==wh.max() and np.sum(wh==h[i])==1:
            conf.setdefault(i+R,[]).append({'tipo':'H','x':i,'y':float(h[i]),'confirmado':i+R})
        if l[i]==wl.min() and np.sum(wl==l[i])==1:
            conf.setdefault(i+R,[]).append({'tipo':'L','x':i,'y':float(l[i]),'confirmado':i+R})

    def geom(contactos,t):
        pts=sorted(contactos,key=lambda p:(p['x'],p['tipo']))
        altos=[p for p in pts if p['tipo']=='H']; bajos=[p for p in pts if p['tipo']=='L']
        if len(altos)<2 or len(bajos)<2: return None
        x0=min(p['x'] for p in pts); x1=max(p['x'] for p in pts); dur=x1-x0
        if not MIN_DURACION<=dur<=MAX_DURACION: return None
        for lado in (altos,bajos):
            if np.any(np.diff(sorted(p['x'] for p in lado))<MIN_SEP): return None
        if sum(p['tipo']!=q['tipo'] for p,q in zip(pts[:-1],pts[1:]))<3: return None
        mh,bh=np.polyfit([p['x'] for p in altos],[p['y'] for p in altos],1)
        ml,bl=np.polyfit([p['x'] for p in bajos],[p['y'] for p in bajos],1)
        if mh>=0 or ml>=0: return None
        ar=a[t]
        if not np.isfinite(ar) or ar<=0: return None
        if min(-mh*dur,-ml*dur)<DESCENSO_MIN_ATR*ar: return None
        if any(abs(p['y']-(mh*p['x']+bh))>TOL_PIVOTE_ATR*a[p['x']] for p in altos): return None
        if any(abs(p['y']-(ml*p['x']+bl))>TOL_PIVOTE_ATR*a[p['x']] for p in bajos): return None
        w0=(mh*x0+bh)-(ml*x0+bl); w1=(mh*x1+bh)-(ml*x1+bl)
        if min(w0,w1)<=ANCHURA_MIN_ATR*ar: return None
        contr=(w0-w1)/w0
        if CANAL_MIN_CONTRACCION<=contr<=CANAL_MAX_CONTRACCION: tipo='CANAL'
        elif CANAL_MAX_CONTRACCION<contr<CUNA_MAX_CONTRACCION and mh<ml: tipo='CUÑA'
        else: return None
        xx=np.arange(x0,t+1)
        band=TOL_CIERRE_ATR*a[xx]
        if not np.isfinite(band).all(): return None
        if np.any(c[xx]>(mh*xx+bh)+band): return None
        if np.any(c[xx]<(ml*xx+bl)-band): return None
        return dict(tipo=tipo,mh=float(mh),bh=float(bh),ml=float(ml),bl=float(bl),
                    x0=int(x0),x1=int(x1),duracion=int(dur),contraccion=float(contr),
                    nH=len(altos),nL=len(bajos),total=len(pts),equilibrio=min(len(altos),len(bajos)))

    def semillas(pivotes,t):
        rec=[p for p in pivotes if t-p['x']<=MAX_DURACION+R][-PIVOTES_SEMILLA:]
        if len(rec)<4: return []
        cand=[]
        for inds in combinations(range(len(rec)),4):
            pts=[rec[i] for i in inds]
            if not any(p['confirmado']==t for p in pts): continue
            g=geom(pts,t)
            if g is not None: cand.append((pts,g))
        if not cand: return []
        maxi=max(cand,key=lambda v:v[1]['duracion'])
        ans=[maxi]
        if MAX_SEMILLAS_POR_FECHA>1:
            mini=min(cand,key=lambda v:v[1]['duracion'])
            if mini!=maxi and mini[1]['duracion']<.75*maxi[1]['duracion']:
                ans.append(mini)
        return ans

    def dup(g,fs,t):
        for f in fs:
            if f['estado']!='ACTIVA': continue
            ratio=min(f['duracion'],g['duracion'])/max(f['duracion'],g['duracion'])
            if ratio<.80: continue
            if abs((f['mh']*t+f['bh'])-(g['mh']*t+g['bh']))<.5*a[t] and \
               abs((f['ml']*t+f['bl'])-(g['ml']*t+g['bl']))<.5*a[t]:
                return True
        return False

    pivotes=[]; figuras=[]; rupturas=[]; nextid=1
    for t in range(max(ATR_N+L+R+5,30),n):
        nuevos=conf.get(t,[])
        # Evaluar señales ANTES de actualizar geometria con pivotes confirmados hoy.
        for f in figuras:
            if f['estado']!='ACTIVA' or t<=f['creada']: continue
            th=f['mh']*t+f['bh']; ty=f['mh']*(t-1)+f['bh']
            arriba=(c[t]>th+TOL_CIERRE_ATR*a[t-1] and c[t-1]<=ty+TOL_CIERRE_ATR*a[t-2])
            if arriba:
                f['estado']='ROTA_AL_ALZA'; f['fin']=t
                rupturas.append(dict(Ticker=ticker, ID=f['id'], Tipo=f['tipo'],t=t,
                     Fecha=d.index[t], Entrada_fecha=d.index[t+1] if t+1<n else pd.NaT,
                     nH=f['nH'], nL=f['nL'],Contactos=f['total'],Equilibrio=f['equilibrio'],
                     Duracion=f['duracion'],Actualizaciones=f['actualizaciones'],
                     Contraccion=f['contraccion'],Ancla_inicio=d.index[f['x0']],
                     Techo_ruptura=th))
                continue
            if c[t]<f['ml']*t+f['bl']-TOL_CIERRE_ATR*a[t-1]:
                f['estado']='ROTA_ABAJO'; f['fin']=t
            elif t-f['ultimo_contacto']>MAX_SESIONES_SIN_CONTACTO or t-f['x0']>MAX_VIDA_DESDE_ORIGEN:
                f['estado']='CADUCADA'; f['fin']=t
        if nuevos:
            pivotes.extend(nuevos)
            for f in figuras:
                if f['estado']!='ACTIVA': continue
                for p in nuevos:
                    if p['x']<=f['x1'] or any(q['tipo']==p['tipo'] and q['x']==p['x'] for q in f['contactos']): continue
                    nivel=(f['mh']*p['x']+f['bh']) if p['tipo']=='H' else (f['ml']*p['x']+f['bl'])
                    if abs(p['y']-nivel)>TOL_PIVOTE_ATR*a[p['x']]: continue
                    propuesta=f['contactos']+[p]; g=geom(propuesta,t)
                    if g is None: continue
                    f['contactos']=propuesta; f.update(g)
                    f['ultimo_contacto']=t; f['actualizaciones']+=1
            for pts,g in semillas(pivotes,t):
                if dup(g,figuras,t): continue
                f=dict(id=nextid,estado='ACTIVA',creada=t,fin=None,ultimo_contacto=t,
                       actualizaciones=0,contactos=list(pts))
                f.update(g); figuras.append(f); nextid+=1
    # Una señal principal por activo y dia. La geometria se conserva en las columnas.
    principales=[]
    for t in sorted(set(s['t'] for s in rupturas)):
        grupo=[s for s in rupturas if s['t']==t]
        principales.append(max(grupo,key=lambda s:(s['Contactos'],s['Equilibrio'],s['Duracion'])))
    return principales,figuras,rupturas


# ============================================================
# DETECTOR ALCISTA EXCLUSIVO V2.1
# Mismas condiciones de pivote y B estructural.
# Se han eliminado ramas bajistas/rojas.
# ============================================================
def detectar_divergencias_alcistas(df, rsi14, atr14, params):
    precio = (df['Open'].to_numpy(dtype=float) + df['High'].to_numpy(dtype=float) +
              df['Low'].to_numpy(dtype=float) + df['Close'].to_numpy(dtype=float)) / 4.0
    rsi = rsi14.to_numpy(dtype=float)
    atr = atr14.to_numpy(dtype=float)
    n = len(df)
    left, right = params['pivot_left'], params['pivot_right']
    max_bars = params['max_barras']
    low_pivots, confirmed = [], []
    bull_ghost = None

    for i in range(n):
        center = i-right
        low_pivot_confirmed = (
            center >= left and
            pivot_low_pine_like(precio, center, left, right)
        )
        if low_pivot_confirmed:
            while low_pivots and center-low_pivots[0]['idx'] > max_bars:
                low_pivots.pop(0)
            ref = buscar_referencia_alcista(
                low_pivots, center, precio[center], rsi[center], atr[center], params
            )
            if ref is not None:
                confirmed.append(crear_evento(
                    direction='ALCISTA', state='CONFIRMADA', signal_idx=i,
                    pivot_idx=center, curr_price=precio[center],
                    curr_rsi=rsi[center], referencia=ref
                ))
            low_pivots.append(dict(
                idx=int(center), price=float(precio[center]),
                rsi=float(rsi[center]) if np.isfinite(rsi[center]) else np.nan
            ))

        if (params['mostrar_fantasmas'] and np.isfinite(precio[i]) and
            np.isfinite(rsi[i]) and np.isfinite(atr[i])):
            ref = buscar_referencia_alcista(
                low_pivots, i, precio[i], rsi[i], atr[i], params
            )
            potencial = ref is not None
            if bull_ghost is not None and i-bull_ghost['p1_idx'] > max_bars:
                bull_ghost = None
            if potencial and bull_ghost is None:
                bull_ghost = crear_evento(
                    direction='ALCISTA', state='FANTASMA', signal_idx=i,
                    pivot_idx=i, curr_price=precio[i], curr_rsi=rsi[i], referencia=ref
                )
            elif bull_ghost is not None and precio[i] < bull_ghost['p2_price']:
                bull_ghost = (
                    crear_evento(direction='ALCISTA', state='FANTASMA',
                                 signal_idx=i, pivot_idx=i,
                                 curr_price=precio[i], curr_rsi=rsi[i], referencia=ref)
                    if potencial else None
                )
        if low_pivot_confirmed:
            bull_ghost = None

    ghosts = [bull_ghost.copy()] if bull_ghost is not None and params['mostrar_fantasmas'] else []
    return confirmed, ghosts


# ========== PSM CONFIRMADO: MIX35 + 3 ATR + 1.5 ATR ==========
def psm_mix35(close, atr):
    """McGinley MIX35 recursiva del PSM; usa solo datos hasta cada cierre."""
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < PSM_N_MIX:
        return out
    out[PSM_N_MIX - 1] = np.mean(c[:PSM_N_MIX])
    for t in range(PSM_N_MIX, n):
        if not np.isfinite(out[t-1]) or out[t-1] <= 0:
            continue
        impulso = (max(0.0, c[t]-c[t-10]) / a[t]
                   if np.isfinite(a[t]) and a[t] > 0 else 0.0)
        recorrido = np.abs(np.diff(c[t-20:t+1])).sum()
        er = abs(c[t]-c[t-20]) / recorrido if recorrido > 0 else 0.0
        fi = np.clip((impulso-2.0)/3.0, 0.0, 1.0)
        fe = np.clip((er-0.25)/0.30, 0.0, 1.0)
        periodo = PSM_N_MIX-(PSM_N_MIX-PSM_N_MIN)*(fi*fe)
        ratio = min(c[t]/out[t-1], PSM_CAP_RATIO)
        if ratio <= 0:
            continue
        alpha = np.clip(1.0/(PSM_K*periodo*ratio**4), 0.0, 1.0)
        out[t] = out[t-1]+alpha*(c[t]-out[t-1])
    return out


def psm_direccion_atr(close, atr, factor):
    """Dirección causal de la subtendencia; ATR fijado en cada extremo."""
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr, dtype=float)
    estados = np.full(len(c), 'L', dtype='<U1')
    validos = np.flatnonzero(np.isfinite(c) & np.isfinite(a) & (a > 0))
    if not len(validos):
        return estados
    inicio = int(validos[0])
    maximo = minimo = c[inicio]
    atr_max = atr_min = a[inicio]
    estado = 'L'
    extremo = c[inicio]
    atr_extremo = a[inicio]
    for t in range(inicio+1, len(c)):
        if not np.isfinite(a[t]) or a[t] <= 0:
            estados[t] = estado
            continue
        precio = c[t]
        if estado == 'L':
            if precio > maximo:
                maximo, atr_max = precio, a[t]
            if precio < minimo:
                minimo, atr_min = precio, a[t]
            sube = precio-minimo >= factor*atr_min
            baja = maximo-precio >= factor*atr_max
            if sube or baja:
                if sube and baja:
                    fa = (precio-minimo)/(factor*atr_min)
                    fb = (maximo-precio)/(factor*atr_max)
                    estado = 'A' if fa >= fb else 'B'
                else:
                    estado = 'A' if sube else 'B'
                extremo, atr_extremo = precio, a[t]
        elif estado == 'A':
            if precio > extremo:
                extremo, atr_extremo = precio, a[t]
            elif extremo-precio >= factor*atr_extremo:
                estado = 'B'
                extremo, atr_extremo = precio, a[t]
        elif estado == 'B':
            if precio < extremo:
                extremo, atr_extremo = precio, a[t]
            elif precio-extremo >= factor*atr_extremo:
                estado = 'A'
                extremo, atr_extremo = precio, a[t]
        estados[t] = estado
    return estados


def calcular_psm(df):
    """Estado en cada cierre y solo primeras apariciones LAA."""
    c = df['Close'].to_numpy(dtype=float)
    a = df['ATR14_PSM'].to_numpy(dtype=float)
    mix = psm_mix35(c, a)
    pnd = (pd.Series(mix, index=df.index) -
           pd.Series(mix, index=df.index).shift(10)) / df['ATR14_PSM']
    p = np.select([pnd > PSM_UMBRAL_P, pnd < -PSM_UMBRAL_P],
                  ['A', 'B'], default='L')
    sec = psm_direccion_atr(c, a, PSM_ATR_SEC)
    mic = psm_direccion_atr(c, a, PSM_ATR_MICRO)
    valido = (np.isfinite(pnd.to_numpy()) &
              np.isin(sec, ['A', 'B']) & np.isin(mic, ['A', 'B']))
    estado = np.array([x+y+z if ok else 'XXX'
                       for x,y,z,ok in zip(p,sec,mic,valido)])
    anterior = np.concatenate((['XXX'],estado[:-1]))
    nueva_laa = ((estado == 'LAA') &
                 (anterior != 'LAA') & (anterior != 'XXX'))
    res = df.copy()
    res['MIX35'] = mix
    res['Pendiente_P_ATR'] = pnd
    res['P'], res['S'], res['M'] = p, sec, mic
    res['ESTADO_PSM'] = estado
    res['NUEVA_LAA'] = nueva_laa
    return res


def escenario_psm(estado):
    if estado == 'LAA':
        return 'LAA | Principal lateral, secundaria y micro alcistas'
    if estado == 'BAA':
        return 'BAA | Recuperacion temprana, principal bajista'
    if estado == 'AAA':
        return 'AAA | Tres tendencias alcistas'
    if estado == 'XXX':
        return 'SIN ESTADO | Calentamiento insuficiente'
    return f'{estado} | P={estado[0]}, S={estado[1]}, M={estado[2]}'



# -----------------------------------------------------------------------------
# INTEGRACION V4: preparacion comun, SIN modificar los detectores V2.1 y V3.
# -----------------------------------------------------------------------------

def prepare_ohlc(df):
    """Acepta las mismas cotizaciones OHLC ajustadas que el escaner Colab."""
    if df is None or df.empty:
        raise ValueError('Sin datos de cotizacion')
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ('Open', 'High', 'Low', 'Close'):
        if c not in df:
            raise ValueError(f'Falta columna {c}')
    df = df.dropna(subset=['Open','High','Low','Close'])
    fechas = pd.to_datetime(df.index)
    if fechas.tz is not None:
        fechas = fechas.tz_localize(None)
    df.index = fechas.normalize()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    if len(df) < 260:
        raise ValueError('Se requieren 260 velas para inicializar los detectores')
    pc = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/ATR_N, adjust=False).mean()
    df['ATR14_PSM'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return df


def fetch_daily(ticker, period='5y'):
    d = yf.download(ticker, period=period, interval='1d', auto_adjust=True,
                    progress=False, threads=False, multi_level_index=False)
    return prepare_ohlc(d)


def scan_ticker(ticker, df=None):
    """Salida unica para radar, graficos y Telegram; fecha = vela de confirmacion."""
    if df is None:
        df = fetch_daily(ticker)
    elif 'ATR14_PSM' not in df.columns or 'ATR' not in df.columns:
        df = prepare_ohlc(df)
    rsi = calculate_rsi_pine(df['Close'], PARAMS['rsi14_len'])
    atr = calculate_atr_pine(df, PARAMS['atr_len'])
    divs, ghosts = detectar_divergencias_alcistas(df, rsi, atr, PARAMS)
    cunas, _, _ = detectar(df, ticker)
    psm = calcular_psm(df)
    events = []

    def add(kind, idx, provisional=False, details=None):
        idx = int(idx)
        events.append({
            'ticker': ticker, 'type': kind, 'index': idx,
            'date': df.index[idx].strftime('%Y-%m-%d'),
            'age': len(df)-1-idx, 'close': float(df.Close.iloc[idx]),
            'provisional': provisional, 'details': details or {},
        })

    for e in divs:
        add('DIV_ALCISTA', e['signal_idx'], details={
            'pivot_date': df.index[e['pivot_idx']].strftime('%Y-%m-%d'),
            'rsi_diff': float(e['rsi_diff']),
            'price_atr': float(e['price_atr']),
        })
    for e in ghosts:
        add('DIV_FANTASMA', e['signal_idx'], provisional=True,
            details={'rsi_diff':float(e['rsi_diff'])})
    for e in cunas:
        if e['Tipo']=='CUÑA':
            add('CUNA_ROTA', e['t'], details={
                'duration':int(e['Duracion']),
                'contacts':int(e['Contactos']),
                'contraction':float(e['Contraccion']),
            })
    for idx in np.flatnonzero(psm['NUEVA_LAA'].to_numpy(bool)):
        add('PSM_LAA', idx, details={
            'prev_state':str(psm['ESTADO_PSM'].iloc[int(idx)-1])})
    events.sort(key=lambda e:(e['index'],e['type']))
    return {'ticker':ticker, 'df':df, 'psm':psm, 'events':events,
            'state':str(psm['ESTADO_PSM'].iloc[-1]),
            'last_date':df.index[-1].strftime('%Y-%m-%d')}


def latest_signal(events, kind, max_age=25):
    matches = [e for e in events if e['type']==kind and e['age']<=max_age]
    return min(matches, key=lambda e:e['age']) if matches else None


def psm_bars(state):
    return ''.join({'A':'🟩','B':'🟥','L':'⬜'}.get(x,'⬛') for x in str(state)[:3])
