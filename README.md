# David L'Crack — Noviembre 2026

Radar estructural para **divergencias alcistas V2.1 (y fantasmas), rupturas de cuñas V3 y PSM LAA**. Visualización con Streamlit y alertas de Telegram mediante GitHub Actions.

## Qué contiene

- `app.py`: dashboard de señales y gráficos con las tres bandas PSM debajo del RSI, marcadores de divergencias en RSI/precio, triángulos de cuñas y puntos morados LAA.
- `signal_engine.py`: detectores V2.1 y V3 conservados y PSM MIX35 con secundaria **3 ATR** y micro **1,5 ATR**. El radar, los gráficos diarios y Telegram utilizan **el mismo motor**.
- `indicators.py`: indicadores de apoyo del dashboard anterior (MACD, RSI, Koncorde, etc.).
- `config.py`: universo, nombres y grupos de activos del repositorio anterior.
- `scripts/daily_snapshot.py`: escaneo de cierres y alertas nuevas con registro antirrepeticiones independiente.
- `.github/workflows/scheduled_run.yml`: ejecución programada y manual.

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub + Streamlit

1. Subir todos los **archivos y carpetas** del ZIP a la **raíz** del nuevo repositorio. No subir el ZIP como único archivo. Si hay `config.py` o `README.md` existentes, aceptar su sustitución.
2. Conectar el repositorio a Streamlit Community Cloud, seleccionando `app.py` como entrada.
3. Para activar alertas, en GitHub > Settings > Secrets and variables > Actions > New repository secret, crear `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`. **No pegarlos en código ni compartirlos.**
4. Ejecutar manualmente Actions > *L'Crack V4 - Radar diario* > Run workflow. Examinar el log y el archivo `snapshots/alertas_enviadas.json` antes de dejarlo automático.

**Aviso importante:** la primera ejecución, por seguridad, **establece una línea base sin mandar alertas**. Solo ejecuciones posteriores notifican señales nuevas que aparezcan desde el último cierre procesado. Para probar el texto, GitHub Actions acepta `workflow_dispatch` con `dry_run=true`, que imprime mensajes pero no envía.

### Coexistencia con el repositorio antiguo

El repositorio anterior queda intacto. Este usa su propio `snapshots/alertas_enviadas.json` y añade la cabecera `🧠 L'CRACK V4 · 202611` a cada mensaje, incluso cuando comparte bot/chat. Cuando decidas, podrás desactivar las Actions antiguas sin borrar el proyecto.

### Consideraciones

- Radar y alertas solo en velas **diarias**, coherentes con la investigación original. Los gráficos 1W/1h/4h mantienen los indicadores clásicos pero no muestran PSM diario calculado erróneamente sobre barras de otra temporalidad.
- Los cierres del día pueden ser **provisionales** mientras cotiza el mercado; el proceso automatizado no publica alertas sobre la barra del día en curso.
- No se han alterado las condiciones de divergencias V2.1 ni cuñas V3 extraídas del escáner validado; la equivalencia completa en vivo requiere comparar con datos congelados.
- Telegram informa señales nuevas, no equivale a recomendación de compra.
