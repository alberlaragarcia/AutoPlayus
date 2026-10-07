import logging
import os
import sys
import threading
from pathlib import Path

# En un venv de Windows, Tcl/Tk puede no encontrar sus librerías: las buscamos en el Python base.
_tcl_dir = Path(sys.base_prefix) / "tcl"
for _var, _name in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
    if _var not in os.environ and (_tcl_dir / _name).is_dir():
        os.environ[_var] = str(_tcl_dir / _name)

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

from autoplayus.device import create_device
from autoplayus.games.registry import GameSpec, load_catalog

log = logging.getLogger("autoplayus")


class TextHandler(logging.Handler):
    def __init__(self, widget: tk.Text):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.after(0, self._append, msg)

    def _append(self, msg: str) -> None:
        self.widget.configure(state="normal")
        self.widget.insert("end", msg + "\n")
        self.widget.see("end")
        self.widget.configure(state="disabled")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AutoPlayus")
        self.geometry("420x480")
        self.catalog: dict[str, GameSpec] = load_catalog()
        self.agent = None

        ttk.Label(self, text="Minijuego").pack(pady=(12, 2))
        self.combo = ttk.Combobox(self, state="readonly", values=[g.name for g in self.catalog.values()])
        self.combo.current(0)
        self.combo.pack(fill="x", padx=16)

        row = ttk.Frame(self)
        row.pack(pady=10)
        self.start_btn = ttk.Button(row, text="Resolver", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(row, text="Parar", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.log_box = tk.Text(self, state="disabled", height=20)
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        handler = TextHandler(self.log_box)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _selected(self) -> GameSpec:
        return list(self.catalog.values())[self.combo.current()]

    def start(self) -> None:
        spec = self._selected()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self._run, args=(spec,), daemon=True).start()

    def _run(self, spec: GameSpec) -> None:
        try:
            log.info("Iniciando %s (agente: %s)", spec.name, spec.agent)
            self.agent = spec.build_agent(create_device())
            ok = self.agent.run()
            log.info("%s: %s", spec.name, "resuelto" if ok else "no resuelto")
        except Exception as e:  # noqa: BLE001 - mostrar cualquier fallo en el log de la app
            log.error("Error: %s", e)
        finally:
            self.after(0, self._finished)

    def _finished(self) -> None:
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def stop(self) -> None:
        if self.agent:
            self.agent.stop()


def run_app() -> None:
    App().mainloop()
