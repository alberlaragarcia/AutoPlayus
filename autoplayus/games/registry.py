from dataclasses import dataclass, field
from typing import Any

import yaml

from autoplayus import config
from autoplayus.agents import AGENT_TYPES, BaseAgent
from autoplayus.device.base import Device


@dataclass
class GameSpec:
    id: str
    name: str
    agent: str
    params: dict[str, Any] = field(default_factory=dict)

    def build_agent(self, device: Device) -> BaseAgent:
        return AGENT_TYPES[self.agent](device, self.params)


def load_catalog() -> dict[str, GameSpec]:
    with open(config.CATALOG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    games = {g["id"]: GameSpec(**g) for g in raw["games"]}
    unknown = {g.agent for g in games.values()} - AGENT_TYPES.keys()
    if unknown:
        raise ValueError(f"Agentes desconocidos en el catálogo: {unknown}")
    return games
