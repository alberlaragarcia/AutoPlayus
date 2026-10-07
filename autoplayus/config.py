from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
CATALOG_PATH = Path(__file__).resolve().parent / "games" / "catalog.yaml"

COMMON_TEMPLATES_DIR = TEMPLATES_DIR / "common"

ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"  # ruta a adb.exe si no está en el PATH
ADB_SERIAL = "127.0.0.1:5556"  # None = único dispositivo conectado

# Ventana de BlueStacks (toques rápidos como clics de Windows) y tamaño de pantalla del Android
BLUESTACKS_WINDOW = "BlueStacks App Player"
WINDOW_CAPTURE = True  # capturar copiando la ventana (~3 ms) en vez de por ADB (~145-245 ms)
SCREEN_SIZE = (720, 1280)

# Interfaz común a todos los minijuegos (coordenadas de la captura, 720x1280)
UI_START = (360, 1157)  # botón "Start" de la ventana de cada juego
UI_FIRST_TAP = (360, 1030)  # toque que arranca la partida tras pulsar Start
UI_OK = (360, 632)  # botón "OK" de la pantalla final
