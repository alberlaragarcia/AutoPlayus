# AutoPlayus

Herramienta personal para resolver automáticamente los minijuegos de **Playus: Minijuegos Arena** (Android).
Eliges un minijuego en una mini-app y el agente especializado correspondiente lo juega.

## Arquitectura

```
Mini-app (Tkinter) ──> Runner ──> Registro de juegos ──> Agente especializado
                                                              │
                                  ┌───────────────────────────┤
                                  ▼                           ▼
                            Visión (OpenCV)            Dispositivo (ADB)
                          captura -> estado            tap / swipe / drag
```

- **device/**: abstracción del móvil. `AdbDevice` usa `adb` (móvil por USB o emulador) para capturar pantalla y enviar toques.
- **vision/**: utilidades de visión (template matching, color, recortes).
- **agents/**: un agente por *familia de mecánica*, no por juego. Cada agente base implementa el bucle `percibir -> decidir -> actuar`.
  - `PatternRecognitionAgent`: match-3, diferencias, memoria visual.
  - `PuzzleLogicAgent`: sudoku, laberintos, conexiones (solvers deterministas).
  - Futuros: `ReflexAgent` (timing/reacción), `RLAgent` (juegos sin solución fija, aprendizaje por refuerzo).
- **games/**: `catalog.yaml` mapea cada minijuego a su agente y parámetros; `registry.py` lo carga.
- **app/**: mini-app para elegir juego, iniciar/parar y ver el log.

## Hoja de ruta

1. **Fase 0 – Infraestructura**: conectar el móvil por ADB, capturar pantalla, enviar toques. (`python main.py --check-device`)
2. **Fase 1 – Un juego por familia**: elegir 1 juego fácil de cada familia y hacerlo funcionar de punta a punta.
3. **Fase 2 – Catálogo**: ir rellenando `games/catalog.yaml` con los 70 minijuegos y asignarles agente.
4. **Fase 3 – RL / modelos aprendidos** solo para los juegos donde un solver clásico no baste. Los capturas con `data/screenshots/` sirven de dataset.

## Puesta en marcha

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Necesitas [Android platform-tools](https://developer.android.com/tools/releases/platform-tools) (`adb` en el PATH) y
depuración USB activada en el móvil.

```powershell
python main.py --check-device   # comprueba conexión y guarda una captura
python main.py                  # abre la mini-app
```

## Cómo añadir un minijuego

1. Añade una entrada en `autoplayus/games/catalog.yaml` (nombre, agente, parámetros).
2. Guarda capturas/plantillas en `data/templates/<id_juego>/`.
3. Si ninguna familia existente encaja, crea un agente nuevo en `autoplayus/agents/` y regístralo en `agents/__init__.py`.
