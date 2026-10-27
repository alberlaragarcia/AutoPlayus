# AutoPlayus

Mini-app creada con fines educativos en la que agentes juegan solos a los minijuegos de **Playus: Minijuegos Arena** (Android)
igual que lo haría una persona: visión por computador clásica con OpenCV, control en tiempo real y compensación de la
latencia del emulador. Una mini-app de escritorio permite elegir el minijuego y el agente correspondiente lo juega.

> **Proyecto educativo y personal.** Lo he desarrollado para aprender y practicar visión por computador, control en
> tiempo real, modelado de sistemas y validación con datos reales, y como proyecto de portfolio. No está afiliado a
> Playus ni a sus desarrolladores; los nombres y capturas de los juegos pertenecen a sus propietarios.

## Qué demuestra

- **Visión por computador sin modelos entrenados.** Cada agente lee el estado del juego directamente de los píxeles:
  segmentación por color, componentes conexas, morfología, ajuste de círculos, ajuste de rectas y clasificación de
  formas por geometría. Todo corre en pocos milisegundos por captura.
- **Control en tiempo real con latencia medida.** El emulador tarda entre 30 y 130 ms en aplicar un toque y ese retraso
  varía. Los agentes lo miden en partidas reales y lo compensan: predicen dónde estará cada objeto cuando el toque llegue.
- **Modelado y planificación.** En *DangerWall* se midió la física del juego (gravedad, salto, aceleración, colisión
  con un cuadrado que gira) y un planificador por programación dinámica decide en ~13 ms cuándo saltar. No hizo falta
  aprendizaje por refuerzo: un modelo físico medido bastó y sale mucho más barato de probar.
- **Método basado en datos.** Cada juego empieza grabando una partida humana fotograma a fotograma; de ahí salen la
  geometría, los colores, los tiempos y las reglas. Después se valida sin gastar intentos: reproduciendo las
  grabaciones, con simuladores propios y con tests, y solo entonces se prueba en vivo y se calibra con lo que pasó.
- **Ingeniería.** Arquitectura con agentes intercambiables configurados desde un catálogo YAML, abstracción del
  dispositivo, captura de pantalla de ~3 ms vía la API de Windows (ADB tarda ~150-250 ms) y tests automáticos.

## Resultados

La gran mayoría de los juegos implementados logran puntuaciones en el top 1% del ranking global, aunque no es el objetivo figurar en tops altos para no alterar la clasificación, únicamente se busca alcanzar el corte del top 1%.

## Cómo se aborda cada juego

1. **Grabar** una partida humana: `tools/record_game.py` guarda cada captura con su hora (~40-75 por segundo).
2. **Analizar** la grabación: regiones, colores, animaciones, velocidades y cómo puntúa el juego. Por ejemplo, en
   CenterHit el porcentaje que muestra el juego resultó ser la media acumulada de los toques, y deshaciéndola salió la
   fórmula exacta de la nota.
3. **Escribir el agente** como funciones puras de percepción y decisión, más una clase que solo orquesta.
4. **Validar sin gastar intentos** (los minijuegos rotan a diario con intentos limitados): reproducir la partida grabada
   con el agente, simuladores del juego y tests sobre capturas reales.
5. **Probar en vivo** con `tools/run_logged.py`, que guarda lo que vio el agente y cada gesto con su hora, y
   **calibrar** con esos datos.

Algunos hallazgos de este proceso:

- **La latencia no es fija y hay que medirla en cada juego.** En CenterHit un toque cuenta ~33 ms después de lo que se
  ve en pantalla; en OnTime, prácticamente al instante. En OnTime, además, el cronómetro tardaba en arrancar entre 98 y 133 ms según la partida: el
  agente mide en pantalla cuándo arrancó de verdad en vez de suponerlo.
- **El juego no siempre funciona como parece.** En DangerWall la colisión no era un círculo sino el cuadrado girado, y
  su giro dependía solo de la posición horizontal; con ese modelo, huecos que parecían imposibles resultaron pasables.
- **Python y Windows también influyen.** `sleep()` tiene ~15 ms de resolución, así que los toques de precisión usan
  espera activa; y desde la mini-app el hilo de la interfaz podría retrasarlos, por lo que se evita el cambio de hilo
  en el momento crítico.

## Arquitectura

```
Mini-app (Tkinter) ──> catalog.yaml ──> Agente del juego
                                          │  bucle: captura -> ¿fin? -> step()
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                    Visión (OpenCV/NumPy)          Dispositivo (BlueStacks)
                    captura -> estado              captura de ventana (~3 ms)
                                                   toques y arrastres por mensajes de Windows
```

| Carpeta | Contenido |
|---|---|
| `autoplayus/agents/` | Un agente por juego sobre `BaseAgent` (bucle percibir → decidir → actuar, Start/OK comunes) |
| `autoplayus/games/catalog.yaml` | Cada juego, su agente y sus parámetros (regiones, colores, tiempos), comentados |
| `autoplayus/device/` | Abstracción del dispositivo: `AdbDevice` y `BlueStacksDevice` (captura y entrada vía API de Windows) |
| `autoplayus/app/` | Mini-app para elegir el juego, lanzarlo y ver el log |
| `tools/` | `record_game.py` (grabar partidas) y `run_logged.py` (ejecutar un agente registrando lo que ve y hace) |
| `tests/` | Tests sobre capturas reales, imágenes sintéticas y dispositivos simulados |
| `data/templates/` | Capturas de referencia de cada juego que usan los tests |

**Tecnologías:** Python 3.13, OpenCV, NumPy, Tkinter, PyYAML, ctypes (Win32), ADB, BlueStacks, pytest.

## Puesta en marcha

Requisitos: Windows, [BlueStacks](https://www.bluestacks.com/) en vertical a 720x1280 con el juego instalado, y el
`adb` de BlueStacks (ruta configurable en `autoplayus/config.py`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py --check-device   # comprueba la conexión y guarda una captura
python main.py                  # mini-app: elegir juego y pulsar Resolver con el juego abierto en BlueStacks
```

Herramientas de desarrollo:

```powershell
python tools/record_game.py data/recordings/MiJuego_human1          # graba mientras juegas (para con el archivo STOP)
python tools/run_logged.py MiJuego data/recordings/MiJuego_run1     # juega el agente y guarda capturas y gestos
pip install pytest; python -m pytest                                # 87 tests, ~15 s, sin necesitar el emulador
```

## Limitaciones y siguientes pasos

- Está atado a Windows + BlueStacks: la captura y los toques usan la API de Windows sobre su ventana.
- En esta versión hay agentes para 15 de los ~70 minijuegos. Los esqueletos de familias genéricas.
- El aprendizaje por refuerzo queda como opción para juegos sin un modelo que se pueda medir; hasta ahora no ha hecho
  falta, pero se implementará en futuras versiones.
