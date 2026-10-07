# AutoPlayus – Guía de desarrollo

Esta guía explica cómo está organizado el proyecto, qué va en `catalog.yaml` y qué hay que tocar para añadir un juego nuevo con su propio agente.

## 1. Flujo general

```
Mini-app (gui.py) ──elige un juego──> catalog.yaml ──> GameSpec ──build_agent()──> Agente
                                                                                     │
                                          bucle en BaseAgent.run():                  │
                                          captura -> ¿resuelto? -> step()  <─────────┘
                                                │                    │
                                          Device.screenshot()   Device.tap/swipe
                                          (BlueStacks por ADB)  (+ vision/ para analizar)
```

1. En la mini-app eliges un juego del desplegable (se rellena desde `catalog.yaml`).
2. Al pulsar **Resolver**, `GameSpec.build_agent()` busca en `AGENT_TYPES` la clase indicada en `agent:` y la crea con el dispositivo ADB y los `params` del juego.
3. `agent.run()` repite: captura de pantalla → `is_solved(frame)` → `step(frame)` → pausa, hasta resolver, pulsar **Parar** o agotar `max_steps`.

> **Importante:** hoy el agente asume que **el minijuego ya está abierto** en BlueStacks. Navegar por el menú de Playus hasta el juego no está implementado (ver sección 6).

## 2. Qué hace cada paquete

Todo el código está en `autoplayus/`.

| Paquete / archivo | Para qué sirve |
|---|---|
| `config.py` | Rutas del proyecto y ajustes de ADB (`ADB_PATH`, `ADB_SERIAL`). Si cambias el puerto de BlueStacks, se cambia aquí. |
| `device/base.py` | Clase abstracta `Device`: lo único que un agente sabe del móvil (`screenshot`, `screen_size`, `tap`, `swipe`, `drag_path`). Los agentes **nunca** llaman a ADB directamente. |
| `device/adb.py` | `AdbDevice`: implementación de `Device` con ADB. Se conecta a BlueStacks (`adb connect`), captura con `screencap` y envía toques con `input tap/swipe`. Las coordenadas son las de la captura: **720x1280** en vertical. |
| `vision/utils.py` | Utilidades de visión con OpenCV: `crop`, `find_template`, `find_all_templates`. Aquí van las funciones que se reutilicen entre juegos (detección por color, segmentar una cuadrícula, OCR...). |
| `agents/base.py` | `BaseAgent`: el bucle percibir → decidir → actuar. Las subclases implementan `step()` y, opcionalmente, `is_solved()` y `on_start()`. |
| `agents/pattern_recognition.py` | Agente para juegos de reconocer elementos y tocarlos o intercambiarlos (match-3, diferencias, memoria). Hoy es un esqueleto con `NotImplementedError`. |
| `agents/puzzle_logic.py` | Agente para puzzles con solución calculable (sudoku, laberintos, conexiones). También es un esqueleto. |
| `agents/__init__.py` | Diccionario `AGENT_TYPES`: asocia el nombre que escribes en el catálogo con la clase Python. **Todo agente nuevo debe registrarse aquí.** |
| `games/catalog.yaml` | Lista de minijuegos: qué agente usa cada uno y con qué parámetros. |
| `games/registry.py` | Lee el catálogo, lo valida (falla si un `agent:` no existe) y construye el agente. |
| `app/gui.py` | Mini-app Tkinter: desplegable, botones Resolver/Parar y log. Ejecuta el agente en un hilo aparte. |
| `data/templates/<id_juego>/` | Imágenes recortadas (plantillas) que el agente busca en pantalla. |
| `data/screenshots/` | Capturas guardadas: referencia para desarrollar y futuro dataset si se usa aprendizaje. |
| `tests/` | Pruebas. Ahora solo comprueba que el catálogo carga. |

## 3. El catálogo (`games/catalog.yaml`)

Cada entrada tiene cuatro campos:

```yaml
games:
  - id: count                 # identificador único, sin espacios. Es también el nombre de la carpeta data/templates/count/
    name: Count!              # texto que se ve en el desplegable de la mini-app
    agent: count              # clave de AGENT_TYPES (ver sección 4)
    params:                   # libre: lo que ese agente necesite
      board_region: [0, 150, 720, 900]   # x, y, ancho, alto de la zona de juego
      rows: 4
      cols: 4
      match_threshold: 0.88
```

- `id`, `name` y `agent` son **obligatorios**. `params` es opcional y por defecto vacío.
- **`params` no tiene esquema fijo.** El agente lo recibe como `self.params` (un diccionario) y lee lo que necesite. Pon ahí todo lo que sea *dato del juego* y no lógica:
  - Regiones de pantalla: tablero, marcador, botón de "Jugar", pantalla de victoria.
  - Estructura: filas, columnas, número de colores o piezas.
  - Umbrales de visión (`match_threshold`, tolerancia de color).
  - Tiempos: pausas tras una animación, duración del swipe.
  - Nombres de las plantillas de `data/templates/<id>/`.
- Todas las coordenadas van en el sistema de la captura (720x1280).
- Si dos juegos comparten mecánica, pueden usar el mismo `agent:` con distintos `params`. Ese es el objetivo de las familias.

## 4. Añadir un juego con su propio agente

Si quieres hacer un agente específico por juego (recomendable para empezar), los pasos son:

### Paso 1 – Reunir material
- Abre el juego en BlueStacks y guarda capturas con `python main.py --check-device` o un script propio. Hazlas del estado inicial, a mitad de partida y de la pantalla final.
- Recorta los elementos que haya que reconocer y guárdalos en `data/templates/<id_juego>/`.
- Anota las coordenadas de las zonas importantes (puedes abrir la captura en cualquier editor de imágenes).

### Paso 2 – Crear el agente
Crea `autoplayus/agents/<nombre>.py`:

```python
import numpy as np

from autoplayus.agents.base import BaseAgent
from autoplayus.vision.utils import crop, find_template


class CountAgent(BaseAgent):
    def on_start(self) -> None:
        # opcional: cargar plantillas, leer self.params
        ...

    def is_solved(self, frame: np.ndarray) -> bool:
        # opcional: detectar la pantalla de fin / victoria
        return False

    def step(self, frame: np.ndarray) -> None:
        # 1. percibir: analizar `frame` (usa vision/utils.py)
        # 2. decidir: qué hacer
        # 3. actuar: self.device.tap(x, y) / self.device.swipe(...)
        ...
```

Lo que **debes** implementar es `step()`. `is_solved()` y `on_start()` son opcionales. Si no implementas `is_solved()`, el agente solo termina al pulsar Parar o al llegar a `max_steps` (500 por defecto).

### Paso 3 – Registrarlo
En `autoplayus/agents/__init__.py`:

```python
from autoplayus.agents.count import CountAgent

AGENT_TYPES = {
    ...
    "count": CountAgent,
}
```

### Paso 4 – Darlo de alta en el catálogo
Añade la entrada en `catalog.yaml` con `agent: count` y sus `params`.

### Paso 5 – Probarlo
- Abre el juego en BlueStacks, ejecuta `python main.py`, elige el juego y pulsa Resolver.
- Si algo falla, el error aparece en el log de la mini-app.

### Resumen de qué tocar por juego

| Acción | Archivo |
|---|---|
| Crear la lógica del juego | `agents/<nombre>.py` (nuevo) |
| Registrar la clase | `agents/__init__.py` |
| Declarar el juego y sus datos | `games/catalog.yaml` |
| Guardar plantillas | `data/templates/<id>/` |
| Añadir una función de visión reutilizable | `vision/utils.py` (solo si la usarán varios juegos) |

No hace falta tocar `gui.py`, `registry.py`, `device/` ni `base.py`.

### Cuándo reutilizar una familia en vez de crear un agente
Cuando dos juegos se resuelven con la misma lógica y solo cambian regiones y umbrales, usa el mismo agente con distintos `params`. Cuando un agente específico ya funciona y encuentras otro juego parecido, extrae lo común a una clase base o a `vision/utils.py` en lugar de copiar y pegar.

## 5. Consejos para desarrollar sin sufrir

- **Desarrolla la percepción sobre capturas guardadas**, sin ejecutar el juego: carga el PNG con `cv2.imread`, pásalo a tu función y comprueba qué detecta. Es más rápido que probar en vivo.
- **Divide `step()` en funciones pequeñas** (`read_board(frame)`, `choose_move(state)`, `execute(move)`). La decisión, si es pura lógica, se puede probar con tests sin móvil.
- **Registra lo que ve el agente** con `log.info(...)`, para entender el fallo desde el log de la mini-app.
- **Cuida los tiempos:** hay animaciones, y la captura por ADB tarda unos cientos de milisegundos. Ajusta `step_delay` o espera dentro de `step()` tras cada acción.
- **No cambies la resolución ni la orientación de BlueStacks**: todas las coordenadas y plantillas dependen de ella.

## 6. Limitaciones actuales

- **No hay navegación por los menús.** El agente empieza con el juego abierto. Cuando haga falta, se puede añadir un `launch()` por juego (con plantillas de botones) o una capa de navegación común.
- **`drag_path` levanta el dedo** entre tramos porque encadena swipes. No sirve para trazos continuos (laberintos, conectar puntos). Cuando llegue un juego así, habrá que implementar un arrastre continuo en `AdbDevice` (por ejemplo con `input motionevent` o un único `swipe` con puntos intermedios).
- **Captura por ADB lenta** para juegos de reflejos. Habrá que buscar un método más rápido cuando toque esa familia.
- **Los agentes de las dos familias son esqueletos.** `PatternRecognitionAgent` y `PuzzleLogicAgent` lanzan `NotImplementedError` hasta que se implementen. Los cinco juegos de `catalog.yaml` son ejemplos de relleno que no funcionan todavía.
- **No hay datos reales de los juegos.** Antes de decidir la familia de cada uno hay que jugarlos y capturarlos.

## 7. Siguiente paso recomendado

1. Elegir **un juego sencillo** (por ejemplo Count!) y capturar sus pantallas.
2. Seguir los pasos de la sección 4 con un agente propio.
3. Cuando funcione, mirar qué se puede reutilizar y decidir si conviene agrupar en familias.
