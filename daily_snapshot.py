"""Radar de alertas diario L'Crack V4, independiente del repositorio antiguo.

Primera ejecución por ticker: baseline, NO emite señales históricas.
En ejecuciones siguientes: señales únicas posteriores al último cierre escaneado.
Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ALL_TICKERS, TICKER_NAMES
from signal_engine import scan_ticker, psm_bars

STATE_FILE = ROOT / 'snapshots' / 'alertas_enviadas.json'
PREFIX = "🧠 L'CRACK V4 · 202611"
KINDS = {
    'DIV_ALCISTA': '🟠 DIVERGENCIA ALCISTA',
    'DIV_FANTASMA': '🟠👻 DIVERGENCIA PROVISIONAL',
    'CUNA_ROTA': '▲ CUÑA BAJISTA ROTA',
    'PSM_LAA': '🟣 NUEVA LAA',
}


def telegram_send(msg: str) -> None:
    token = os.environ.get('TELEGRAM_BOT_TOKEN','').strip()
    chat_id = os.environ.get('TELEGRAM_CHAT_ID','').strip()
    if not token or not chat_id:
        raise RuntimeError('Faltan secrets TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID')
    data = urlencode({'chat_id': chat_id, 'text': msg, 'disable_web_page_preview': 'true'}).encode()
    request = Request(f'https://api.telegram.org/bot{token}/sendMessage', data=data)
    try:
        with urlopen(request, timeout=25) as response:
            body = json.load(response)
        if not body.get('ok'):
            raise RuntimeError('Telegram rechazó el envío')
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f'Fallo al notificar Telegram: {type(exc).__name__}') from exc


def load_state():
    if STATE_FILE.exists():
        data=json.loads(STATE_FILE.read_text(encoding='utf-8'))
        if data.get('version')!=4:
            raise ValueError('Historial de avisos con versión desconocida')
        return data
    return {'version':4, 'watermarks':{}, 'sent':[]}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True,exist_ok=True)
    tmp=STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(STATE_FILE)


def message(e, state):
    ticker=e['ticker']
    d=e['details']
    lines=[PREFIX, f"{KINDS[e['type']]} — {ticker}",
           f"{TICKER_NAMES.get(ticker,ticker)} · {e['date']}",
           f"Cierre: {e['close']:.2f}",
           f"PSM actual: {psm_bars(state)} ({state})"]
    if e['type']=='PSM_LAA':
        lines.append('Modelo: MIX35 | Secundaria 3 ATR | Micro 1.5 ATR')
    elif e['type']=='DIV_ALCISTA':
        lines.append(f"ΔRSI {d.get('rsi_diff',0):.2f} | ΔPrecio {d.get('price_atr',0):.2f} ATR")
    elif e['type']=='CUNA_ROTA':
        lines.append(f"Duración {d.get('duration')} velas | Contactos {d.get('contacts')}")
    elif e['type']=='DIV_FANTASMA':
        lines.append('Provisional: puede desaparecer antes de confirmarse')
    return '\n'.join(lines)


def run(dry_run=False, tickers=None):
    state=load_state()
    sent=set(state['sent'])
    tickers=list(dict.fromkeys(tickers or ALL_TICKERS))
    print('L\'Crack V4: tickers', len(tickers), '| pruebas',dry_run,flush=True)
    count=0; failures=[]; baselines=0
    today=datetime.now(timezone.utc).date().isoformat()
    for n,ticker in enumerate(tickers,1):
        try:
            r=scan_ticker(ticker)
            # Tras las 23:00 UTC, el cierre diario bursátil (US/EU) ha terminado.
            # Para activos 24/7 (cripto), no usar la barra del día actual.
            is_continuous=ticker.endswith('-USD')
            allowed_today=(datetime.now(timezone.utc).hour>=23 and not is_continuous)
            eligible_dates=[x.strftime('%Y-%m-%d') for x in r['df'].index
                            if x.strftime('%Y-%m-%d')<today or
                            (allowed_today and x.strftime('%Y-%m-%d')==today)]
            if not eligible_dates:
                continue
            last_complete=eligible_dates[-1]
            watermark=state['watermarks'].get(ticker)
            if watermark is None:
                # Importante: nueva instalación, NO enviar señales históricas.
                if not dry_run:
                    state['watermarks'][ticker]=last_complete
                    save_state(state)
                baselines+=1
                print(f'[{n}/{len(tickers)}] {ticker}: línea base {last_complete}',flush=True)
                continue
            latest_events=[e for e in r['events']
                           if watermark<e['date']<=last_complete]
            for e in latest_events:
                key='|'.join([ticker,e['type'],e['date']])
                if key in sent:
                    continue
                msg=message(e,r['state'])
                if dry_run:
                    print('\n[SIMULACION TELEGRAM]\n'+msg+'\n',flush=True)
                else:
                    telegram_send(msg)
                    sent.add(key)
                    state['sent'].append(key)
                    save_state(state)  # Persistir cada éxito, tolera fallos parciales
                count+=1
            if not dry_run:
                state['watermarks'][ticker]=last_complete
                save_state(state)
            print(f'[{n}/{len(tickers)}] {ticker} | cierre {last_complete} | alertas: {len(latest_events)}',flush=True)
        except Exception as exc:
            failures.append((ticker,str(exc)))
            print(f'ERROR {ticker}: {type(exc).__name__}: {exc}',flush=True)
    print(f'Fin: {count} mensajes, {baselines} líneas base, {len(failures)} errores',flush=True)
    if failures:
        raise RuntimeError('Errores detectados; no se avanza el watermark de esos tickers')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true',help='Imprime sin enviar ni persistir')
    ap.add_argument('--tickers',nargs='*',help='Restringir tickers de prueba')
    opts=ap.parse_args()
    run(opts.dry_run,opts.tickers)
