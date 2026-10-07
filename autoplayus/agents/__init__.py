from autoplayus.agents.base import BaseAgent
from autoplayus.agents.pattern_recognition import PatternRecognitionAgent
from autoplayus.agents.puzzle_logic import PuzzleLogicAgent
from autoplayus.agents.tower_stack import TowerStack
from autoplayus.agents.fire_fly import FireFly

AGENT_TYPES: dict[str, type[BaseAgent]] = {
    "pattern_recognition": PatternRecognitionAgent,
    "puzzle_logic": PuzzleLogicAgent,
    "tower_stack": TowerStack,
    "fire_fly": FireFly,
}
