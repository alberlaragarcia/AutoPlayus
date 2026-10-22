from autoplayus.agents.base import BaseAgent
from autoplayus.agents.pattern_recognition import PatternRecognitionAgent
from autoplayus.agents.puzzle_logic import PuzzleLogicAgent
from autoplayus.agents.tower_stack import TowerStack
from autoplayus.agents.fire_fly import FireFly
from autoplayus.agents.parcel_panic import ParcelPanic
from autoplayus.agents.piano import Piano
from autoplayus.agents.lane_rush import LaneRush
from autoplayus.agents.react import React
from autoplayus.agents.zig import Zig
from autoplayus.agents.dice_roll import DiceRoll
from autoplayus.agents.grid import Grid
from autoplayus.agents.trace import Trace
from autoplayus.agents.color_memory import ColorMemory
from autoplayus.agents.danger_wall import DangerWall
from autoplayus.agents.center_hit import CenterHit
from autoplayus.agents.swipe_fast import SwipeFast

AGENT_TYPES: dict[str, type[BaseAgent]] = {
    "pattern_recognition": PatternRecognitionAgent,
    "puzzle_logic": PuzzleLogicAgent,
    "tower_stack": TowerStack,
    "fire_fly": FireFly,
    "parcel_panic": ParcelPanic,
    "piano": Piano,
    "lane_rush": LaneRush,
    "react": React,
    "zig": Zig,
    "dice_roll": DiceRoll,
    "grid": Grid,
    "trace": Trace,
    "color_memory": ColorMemory,
    "danger_wall": DangerWall,
    "center_hit": CenterHit,
    "swipe_fast": SwipeFast,
}
