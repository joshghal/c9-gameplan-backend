# Core modules (no database dependencies)
from .weapon_system import WeaponDatabase, WeaponStats, ArmorStats, WeaponCategory
from .economy_engine import EconomyEngine, BuyType, Loadout, TeamEconomy
from .round_state import RoundState, WinProbabilityCalculator
from .behavior_adaptation import BehaviorAdapter, BehaviorModifiers, PlayerTendencies
from .strategy_coordinator import StrategyCoordinator, Strategy, Role
from .ability_system import AbilitySystem, AbilityDatabase

__all__ = [
    "WeaponDatabase",
    "WeaponStats",
    "ArmorStats",
    "WeaponCategory",
    "EconomyEngine",
    "BuyType",
    "Loadout",
    "TeamEconomy",
    "RoundState",
    "WinProbabilityCalculator",
    "BehaviorAdapter",
    "BehaviorModifiers",
    "PlayerTendencies",
    "StrategyCoordinator",
    "Strategy",
    "Role",
    "AbilitySystem",
    "AbilityDatabase",
]

# Optional imports requiring database dependencies
try:
    from .simulation_engine import SimulationEngine
    from .pathfinding import AStarPathfinder
    from .pattern_matcher import PatternMatcher
    __all__.extend(["SimulationEngine", "AStarPathfinder", "PatternMatcher"])
except ImportError:
    pass  # Database dependencies not available
