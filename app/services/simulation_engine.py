"""Simulation Engine for VALORANT tactical scenarios.

Manages the simulation of rounds including:
- Player movement based on learned patterns and strategy coordination
- Combat resolution with weapon mechanics and armor
- Economy-based loadout generation
- Spike plant/defuse scenarios
- What-if scenario branching
- First blood and man advantage tracking
- Ability effects
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..models import SimulationSession, Team, Player, MapConfig
from ..schemas.simulations import (
    SimulationState, PlayerPosition, SimulationEvent, WhatIfScenario
)
from .pattern_matcher import PatternMatcher, PositionPrediction
from .pathfinding import AStarPathfinder

# New system imports
from .weapon_system import (
    WeaponDatabase, WeaponStats, ArmorStats, WeaponCategory,
    meters_from_normalized
)
from .economy_engine import EconomyEngine, BuyType, Loadout, TeamEconomy
from .round_state import RoundState, WinProbabilityCalculator, KillEvent
from .behavior_adaptation import (
    BehaviorAdapter, BehaviorModifiers, PlayerTendencies, PositioningAdvisor,
    SpikeCarrierTendencies
)
from .strategy_coordinator import (
    StrategyCoordinator, Strategy, Role, RoleAssignment
)
from .ability_system import AbilitySystem, AbilityDatabase
from .validated_parameters import (
    VALIDATED_PARAMS, MOVEMENT_ACCURACY_MODIFIERS, WEAPON_SPREAD_DATA,
    PeekersAdvantageParams, EngagementParams, TradeParams, ProMatchStats,
    RoundPacingParams, calculate_movement_accuracy
)
# AI and Information Systems for emergent behavior
from .information_system import (
    InformationManager, PlayerKnowledge, InfoSource, InfoConfidence
)
from .ai_decision_system import (
    AIDecisionSystem, AIBehaviorIntegration, Decision, DecisionContext, DecisionResult
)
# Neural Network AI System - learned behavior instead of rules
from .neural_ai_system import (
    NeuralAISystem, NeuralAIPlayer, GameStateFeatures, Action as NeuralAction
)
# VCT Data Loader for hold angles and realism data
from .data_loader import get_data_loader
# Realistic Combat Model - tick-by-tick simulation
from .combat_model import (
    RealisticCombatModel, PlayerCombatProfile, CombatResult,
    MovementState, EngagementType, combat_model, resolve_combat
)


@dataclass
class SimulatedPlayer:
    """Internal state for a simulated player."""
    player_id: str
    team_id: str
    side: str
    x: float
    y: float
    is_alive: bool = True
    health: int = 100
    agent: str = "unknown"
    has_spike: bool = False
    current_pattern_id: Optional[int] = None

    # NEW: Weapon & Economy fields
    shield: int = 0
    weapon: Optional[WeaponStats] = None
    armor: Optional[ArmorStats] = None
    loadout_value: int = 0

    # NEW: Ability & Status fields
    is_flashed: bool = False
    flash_end_ms: int = 0
    is_slowed: bool = False
    is_revealed: bool = False

    # NEW: Role & Behavior fields
    role: Optional[Role] = None
    tendencies: Optional[PlayerTendencies] = None
    behavior_modifiers: Optional[BehaviorModifiers] = None

    # NEW: Spike carrier tendencies (VCT-derived per-player behavior)
    spike_carrier_tendencies: Optional[SpikeCarrierTendencies] = None

    # NEW: Combat stats
    kills: int = 0
    deaths: int = 0
    headshot_rate: float = 0.25  # Default headshot rate

    # AUTONOMOUS: Player's preferred engagement distance (from VCT data)
    # Players perform better at their natural engagement distance
    # Range: ~1500 (close/aggressive) to ~2200 (long/passive)
    preferred_engagement_distance: float = 1800.0  # Default mid-range

    # Player profile for combat model (optional)
    combat_profile: Optional['PlayerCombatProfile'] = None

    # P3 FIX: Ultimate economy tracking
    ultimate_points: int = 0  # Current ultimate charge
    ultimate_cost: int = 7  # Default cost (varies by agent)
    ultimate_orbs_collected: int = 0  # Orbs picked up this half

    # P0 FIX: Plant/Defuse action states
    is_planting: bool = False
    is_defusing: bool = False
    action_start_ms: int = 0
    defuse_progress_ms: int = 0  # For half-defuse checkpoint (saves progress at 3500ms)

    # P1 FIX: Movement state tracking for peeker's advantage & accuracy
    prev_x: float = 0.0
    prev_y: float = 0.0
    moved_this_tick: bool = False
    last_move_ms: int = 0  # Timestamp of last movement
    is_running: bool = True  # Running vs walking (running = faster but loud/inaccurate)

    # P3 FIX: Sound cue tracking
    made_sound_this_tick: bool = False  # Did player make audible sound?
    heard_enemy_sound: bool = False  # Has player heard enemy footsteps?
    last_sound_location: Optional[Tuple[float, float]] = None  # Where sound was heard from

    # Molly repositioning target (set by ability system when in danger zone)
    reposition_target: Optional[Tuple[float, float]] = None

    @property
    def is_in_action(self) -> bool:
        """Check if player is currently in a vulnerable action state."""
        return self.is_planting or self.is_defusing

    @property
    def weapon_name(self) -> str:
        return self.weapon.name if self.weapon else "Classic"

    @property
    def armor_name(self) -> str:
        return self.armor.name if self.armor else "none"

    @property
    def effective_hp(self) -> int:
        return self.health + self.shield

    @property
    def movement_accuracy_modifier(self) -> float:
        """P1 FIX: Calculate accuracy modifier based on movement state and weapon.

        Uses validated VALORANT spread data:
        - Rifles: 15% running, 30% walking (6°/3° spread penalty)
        - SMGs: 45% running, 65% walking (2.5°/1° spread penalty)
        - Snipers: 5% running, 10% walking (extreme penalty)
        - Pistols: 25% running, 45% walking
        - Shotguns: 50% running, 60% walking (minimal penalty)

        Source: validated_parameters.py (VALORANT Wiki spread data)
        """
        if self.is_in_action:
            return 0.0  # Can't shoot while planting/defusing

        # Determine weapon category
        weapon_name = self.weapon_name.lower() if self.weapon else 'classic'
        weapon_category = 'rifle'  # Default

        # Categorize weapon
        if weapon_name in ['vandal', 'phantom', 'guardian', 'bulldog']:
            weapon_category = 'rifle'
        elif weapon_name in ['spectre', 'stinger']:
            weapon_category = 'smg'
        elif weapon_name in ['operator', 'marshal', 'outlaw']:
            weapon_category = 'sniper'
        elif weapon_name in ['classic', 'ghost', 'sheriff', 'frenzy', 'shorty']:
            weapon_category = 'pistol'
        elif weapon_name in ['judge', 'bucky']:
            weapon_category = 'shotgun'

        # Get accuracy modifiers from validated data
        modifiers = MOVEMENT_ACCURACY_MODIFIERS.get(weapon_category, MOVEMENT_ACCURACY_MODIFIERS['rifle'])

        if self.moved_this_tick and self.is_running:
            return modifiers['running']
        if self.moved_this_tick and not self.is_running:
            return modifiers['walking']
        # Not moving this tick
        return modifiers['standing']

    @property
    def ultimate_ready(self) -> bool:
        """P3 FIX: Check if ultimate is available."""
        return self.ultimate_points >= self.ultimate_cost

    def grant_ultimate_point(self, reason: str = "kill") -> bool:
        """P3 FIX: Grant an ultimate point. Returns True if ultimate became ready."""
        was_ready = self.ultimate_ready
        self.ultimate_points = min(self.ultimate_points + 1, self.ultimate_cost)
        return not was_ready and self.ultimate_ready

    def use_ultimate(self) -> bool:
        """P3 FIX: Use ultimate if ready. Returns True if used."""
        if self.ultimate_ready:
            self.ultimate_points = 0
            return True
        return False


# P3 FIX: Agent ultimate costs (real VALORANT values)
AGENT_ULTIMATE_COSTS = {
    # Duelists
    'Jett': 7, 'Raze': 8, 'Phoenix': 6, 'Reyna': 6, 'Yoru': 7, 'Neon': 7, 'Iso': 7,
    # Initiators
    'Sova': 8, 'Breach': 8, 'Skye': 6, 'KAY/O': 8, 'Fade': 8, 'Gekko': 7,
    # Controllers
    'Omen': 7, 'Brimstone': 7, 'Viper': 8, 'Astra': 7, 'Harbor': 7, 'Clove': 7,
    # Sentinels
    'Killjoy': 8, 'Cypher': 6, 'Sage': 8, 'Chamber': 8, 'Deadlock': 8, 'Vyse': 8,
}


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state for what-if scenarios."""
    id: str
    time_ms: int
    phase: str
    players: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    spike_planted: bool
    spike_site: Optional[str]


class SimulationEngine:
    """Engine for running VALORANT tactical simulations.

    Integrates:
    - Weapon system with damage calculations
    - Economy engine for loadout generation
    - Round state tracking (first blood, man advantage)
    - Behavior adaptation based on round state
    - Strategy coordination for role-based movement
    - Ability system for utility effects

    Phases:
    - opening: First 20 seconds, players move from spawn
    - mid_round: Main phase, engagements happen
    - post_plant: After spike plant (45 second timer)
    - retake: Defenders attempting to retake site
    """

    TICK_DURATION_MS = 128  # ms per tick
    ROUND_TIME_MS = 100000  # 1:40 round time
    SPIKE_TIME_MS = 45000  # 45 second spike timer

    # P0 FIX: Plant and defuse duration constants (real VALORANT values)
    PLANT_DURATION_MS = 4000  # 4 seconds to plant spike
    DEFUSE_DURATION_MS = 7000  # 7 seconds to fully defuse
    HALF_DEFUSE_MS = 3500  # 3.5 seconds checkpoint (progress saved)

    # Phase timings: expanded to 6 phases for more nuanced behavior
    # Pre-plant phases based on round time
    PHASE_TIMINGS = {
        'early': 15000,         # First 15 seconds - setup/info gather
        'mid': 50000,           # 15s - 50s - default play
        'late': 100000,         # 50s+ - time pressure/execute
        # Post-plant sub-phases (time since plant)
        'post_plant_early': 10000,   # 0-10s after plant - setup holds
        'post_plant_mid': 25000,     # 10-25s - hold phase
        'post_plant_late': 45000,    # 25s+ - late retake pressure
    }

    # Combat engagement distance (normalized map units)
    # VCT average engagement: 1846 units (~18m)
    # On ~10,000 unit wide map: 1846/10000 = 0.18 normalized
    # Using 0.40 to account for LOS checks (not all players can see each other)
    ENGAGEMENT_DISTANCE = 0.40  # ~40% of map - accounts for visibility

    # P2 FIX: Trade mechanics constants (VCT-calibrated)
    # Source: VCT extraction - 3,036 trades from 12,029 kills
    # - Trade rate: 25.2%
    # - Avg trade time: 1.72s, 79.3% within 3s
    # - Avg trade distance: 1584 units
    # - Distribution: 28% close, 48% medium, 18% far, 6% very far
    TRADE_WINDOW_MS = 3000  # 3 seconds (VCT: 79.3% of trades within 3s)

    # Trade distance thresholds (VCT-derived from 100 sample trades)
    # Average trade distance: 1584 units (~16m on 100m map = 0.16 normalized)
    # Map is ~10,000 units wide, so: 1000 units = 0.10 normalized
    TRADE_DISTANCE_CLOSE = 0.10   # <1000 units (~10m) - 28% of trades happen here
    TRADE_DISTANCE_MEDIUM = 0.20  # 1000-2000 units (~20m) - 48% of trades (most common)
    TRADE_DISTANCE_FAR = 0.30     # 2000-3000 units (~30m) - 18% of trades

    # P3 FIX: Sound cue constants (real VALORANT values)
    # Running footsteps audible at ~40 meters, walking is silent
    # Map is ~100m wide, so normalized: 40/100 = 0.4
    SOUND_RANGE_RUNNING = 0.40  # Running footsteps audible at 40% of map width
    SOUND_RANGE_GUNFIRE = 0.60  # Gunfire audible at 60% of map width
    SOUND_RANGE_ABILITY = 0.50  # Most abilities audible at 50%
    SOUND_RANGE_SPIKE = 0.80  # Spike plant/defuse sounds travel far

    # P3 FIX: Ultimate orb positions (2 orbs per map, roughly mid-map)
    # Orbs spawn at round start, grant +1 ultimate point when collected
    ORB_POSITIONS = {
        'default': [(0.35, 0.5), (0.65, 0.5)],  # Default mid positions
    }

    # P1 FIX: Map-specific spawn positions and site locations
    # All coordinates are normalized (0.0 - 1.0)
    # Sites include center position and plant radius
    MAP_DATA = {
        # === 2-SITE MAPS ===
        'ascent': {
            'sites': {
                'A': {'center': (0.25, 0.25), 'radius': 0.08},  # A site top-left area
                'B': {'center': (0.75, 0.25), 'radius': 0.08},  # B site top-right area
            },
            'spawns': {
                'attack': [(0.5, 0.92), (0.45, 0.88), (0.55, 0.88), (0.4, 0.85), (0.6, 0.85)],
                'defense': [(0.5, 0.15), (0.3, 0.2), (0.7, 0.2), (0.25, 0.25), (0.75, 0.25)],
            },
        },
        'bind': {
            'sites': {
                'A': {'center': (0.2, 0.3), 'radius': 0.08},   # A site left side
                'B': {'center': (0.8, 0.3), 'radius': 0.08},   # B site right side
            },
            'spawns': {
                'attack': [(0.5, 0.9), (0.45, 0.87), (0.55, 0.87), (0.4, 0.85), (0.6, 0.85)],
                'defense': [(0.5, 0.2), (0.25, 0.25), (0.75, 0.25), (0.2, 0.3), (0.8, 0.3)],
            },
        },
        'split': {
            'sites': {
                'A': {'center': (0.2, 0.2), 'radius': 0.08},   # A site
                'B': {'center': (0.8, 0.2), 'radius': 0.08},   # B site
            },
            'spawns': {
                'attack': [(0.5, 0.95), (0.45, 0.9), (0.55, 0.9), (0.4, 0.88), (0.6, 0.88)],
                'defense': [(0.5, 0.12), (0.25, 0.18), (0.75, 0.18), (0.2, 0.22), (0.8, 0.22)],
            },
        },
        'icebox': {
            'sites': {
                'A': {'center': (0.25, 0.22), 'radius': 0.08},  # A site
                'B': {'center': (0.72, 0.28), 'radius': 0.1},   # B site (larger area)
            },
            'spawns': {
                'attack': [(0.5, 0.88), (0.45, 0.85), (0.55, 0.85), (0.4, 0.82), (0.6, 0.82)],
                'defense': [(0.45, 0.15), (0.25, 0.2), (0.7, 0.25), (0.3, 0.22), (0.65, 0.22)],
            },
        },
        'breeze': {
            'sites': {
                'A': {'center': (0.18, 0.25), 'radius': 0.09},  # A site (pyramid area)
                'B': {'center': (0.82, 0.28), 'radius': 0.08},  # B site
            },
            'spawns': {
                'attack': [(0.5, 0.92), (0.45, 0.88), (0.55, 0.88), (0.4, 0.85), (0.6, 0.85)],
                'defense': [(0.5, 0.12), (0.2, 0.2), (0.8, 0.2), (0.25, 0.25), (0.75, 0.25)],
            },
        },
        'fracture': {
            'sites': {
                'A': {'center': (0.2, 0.5), 'radius': 0.08},   # A site (left side)
                'B': {'center': (0.8, 0.5), 'radius': 0.08},   # B site (right side)
            },
            'spawns': {
                # Fracture has attackers spawning in middle (unique map)
                'attack': [(0.5, 0.1), (0.5, 0.9), (0.45, 0.12), (0.55, 0.12), (0.45, 0.88)],
                'defense': [(0.5, 0.5), (0.3, 0.5), (0.7, 0.5), (0.2, 0.5), (0.8, 0.5)],
            },
        },
        'pearl': {
            'sites': {
                'A': {'center': (0.22, 0.25), 'radius': 0.08},  # A site
                'B': {'center': (0.78, 0.28), 'radius': 0.08},  # B site
            },
            'spawns': {
                'attack': [(0.5, 0.9), (0.45, 0.87), (0.55, 0.87), (0.4, 0.85), (0.6, 0.85)],
                'defense': [(0.5, 0.15), (0.25, 0.2), (0.75, 0.2), (0.22, 0.25), (0.78, 0.25)],
            },
        },
        'sunset': {
            'sites': {
                'A': {'center': (0.22, 0.22), 'radius': 0.08},  # A site
                'B': {'center': (0.78, 0.25), 'radius': 0.08},  # B site
            },
            'spawns': {
                'attack': [(0.5, 0.9), (0.45, 0.87), (0.55, 0.87), (0.4, 0.85), (0.6, 0.85)],
                'defense': [(0.5, 0.15), (0.25, 0.2), (0.75, 0.2), (0.22, 0.22), (0.78, 0.22)],
            },
        },
        'abyss': {
            'sites': {
                'A': {'center': (0.25, 0.25), 'radius': 0.08},  # A site
                'B': {'center': (0.75, 0.28), 'radius': 0.08},  # B site
            },
            'spawns': {
                'attack': [(0.5, 0.88), (0.45, 0.85), (0.55, 0.85), (0.4, 0.82), (0.6, 0.82)],
                'defense': [(0.5, 0.18), (0.28, 0.22), (0.72, 0.22), (0.25, 0.25), (0.75, 0.25)],
            },
        },
        # === 3-SITE MAPS ===
        'haven': {
            'sites': {
                'A': {'center': (0.15, 0.25), 'radius': 0.07},  # A site (left)
                'B': {'center': (0.5, 0.18), 'radius': 0.07},   # B site (middle)
                'C': {'center': (0.85, 0.25), 'radius': 0.07},  # C site (right)
            },
            'spawns': {
                'attack': [(0.5, 0.9), (0.45, 0.87), (0.55, 0.87), (0.4, 0.85), (0.6, 0.85)],
                'defense': [(0.5, 0.35), (0.18, 0.28), (0.82, 0.28), (0.5, 0.22), (0.35, 0.3)],
            },
        },
        'lotus': {
            'sites': {
                'A': {'center': (0.18, 0.28), 'radius': 0.07},  # A site (left)
                'B': {'center': (0.5, 0.2), 'radius': 0.07},    # B site (middle)
                'C': {'center': (0.82, 0.28), 'radius': 0.07},  # C site (right)
            },
            'spawns': {
                'attack': [(0.5, 0.88), (0.45, 0.85), (0.55, 0.85), (0.4, 0.82), (0.6, 0.82)],
                'defense': [(0.5, 0.38), (0.2, 0.3), (0.8, 0.3), (0.5, 0.25), (0.35, 0.32)],
            },
        },
    }

    # Default map data for unknown maps
    DEFAULT_MAP_DATA = {
        'sites': {
            'A': {'center': (0.3, 0.3), 'radius': 0.08},
            'B': {'center': (0.7, 0.3), 'radius': 0.08},
        },
        'spawns': {
            'attack': [(0.15, 0.8), (0.2, 0.8), (0.25, 0.8), (0.3, 0.8), (0.35, 0.8)],
            'defense': [(0.65, 0.2), (0.7, 0.2), (0.75, 0.2), (0.8, 0.2), (0.85, 0.2)],
        },
    }

    @staticmethod
    def _get_role_from_agent(agent: str) -> str:
        """Map agent name to role for spike carrier tendencies.

        Args:
            agent: Agent name (e.g., 'jett', 'killjoy')

        Returns:
            Role string: 'duelist', 'initiator', 'controller', or 'sentinel'
        """
        duelists = ['jett', 'raze', 'reyna', 'phoenix', 'neon', 'yoru', 'iso']
        initiators = ['sova', 'skye', 'breach', 'kayo', 'fade', 'gekko']
        controllers = ['omen', 'brimstone', 'astra', 'viper', 'harbor', 'clove']
        sentinels = ['killjoy', 'cypher', 'sage', 'chamber', 'deadlock', 'vyse']

        agent_lower = agent.lower()
        if agent_lower in duelists:
            return 'duelist'
        elif agent_lower in initiators:
            return 'initiator'
        elif agent_lower in controllers:
            return 'controller'
        elif agent_lower in sentinels:
            return 'sentinel'
        return 'initiator'  # Default to initiator for unknown agents

    def __init__(self, db: AsyncSession):
        self.db = db
        self.pattern_matcher = PatternMatcher(db)
        self.pathfinder = AStarPathfinder()

        # Simulation state
        self.players: Dict[str, SimulatedPlayer] = {}
        self.events: List[SimulationEvent] = []
        self.snapshots: List[SimulationSnapshot] = []
        self.spike_planted = False
        self.spike_site: Optional[str] = None
        self.spike_plant_time: Optional[int] = None
        self.map_name: str = 'ascent'  # P1 FIX: Track current map for site positions

        # NEW: System instances
        self.round_state = RoundState()
        self.strategy_coordinator = StrategyCoordinator()
        self.ability_system = AbilitySystem()

        # NEW: Team economies
        self.attack_economy: Optional[TeamEconomy] = None
        self.defense_economy: Optional[TeamEconomy] = None
        self.attack_strategy: Optional[Strategy] = None
        self.defense_strategy: Optional[Strategy] = None

        # P2 FIX: Trade tracking
        self.pending_trade: Optional[Dict[str, Any]] = None  # Tracks kill needing trade response

        # P3 FIX: Ultimate orb tracking
        # Each map has 2 orbs that spawn at round start
        self.orbs_available: List[Tuple[float, float]] = []

        # P3 FIX: Sound cue events (for info gathering)
        self.sound_events: List[Dict[str, Any]] = []

        # SPIKE FLOW FIX: Site execute tracking
        # Tracks when attackers decide to commit to a site execute
        self.site_execute_active: bool = False  # Is a site execute in progress?
        self.target_site: Optional[str] = None  # Which site are we executing on?
        self.execute_start_time: Optional[int] = None  # When did execute start?
        self.site_control_achieved: bool = False  # Have attackers taken site control?

        # AI SYSTEM: Information and decision making for emergent behavior
        self.info_manager = InformationManager()
        self.ai_system = AIDecisionSystem()
        self.ai_behavior: Optional[AIBehaviorIntegration] = None
        self.use_ai_decisions: bool = True  # Toggle for AI vs hardcoded decisions

        # NEURAL AI SYSTEM: Learned behavior from player data
        self.neural_ai = NeuralAISystem(learning_mode=False)
        self.use_neural_ai: bool = True  # Use neural network AI (True) or utility AI (False)

    async def initialize(
        self,
        session: SimulationSession,
        round_type: str = "full"
    ) -> SimulationState:
        """Initialize simulation from session configuration.

        Args:
            session: Simulation session configuration
            round_type: Type of buy round ('pistol', 'eco', 'force', 'half', 'full')
        """
        # Load teams and players
        attack_team = await self._load_team(session.attack_team_id)
        defense_team = await self._load_team(session.defense_team_id)

        # P1 FIX: Store map name for site lookups
        self.map_name = session.map_name.lower() if session.map_name else 'ascent'

        # Load map configuration
        map_config = await self._load_map_config(session.map_name)

        # Initialize economies
        self._initialize_economies(round_type)

        # Generate loadouts based on economy
        attack_loadouts = EconomyEngine.generate_team_loadout(
            self.attack_economy,
            round_num=session.current_time_ms // 1000,  # Simplified round detection
            agent_roles=[p.get('agent', 'unknown') for p in attack_team.get('players', [])[:5]],
            side='attack',
            forced_buy_type=BuyType[round_type.upper()] if round_type != 'full' else None
        )

        defense_loadouts = EconomyEngine.generate_team_loadout(
            self.defense_economy,
            round_num=session.current_time_ms // 1000,
            agent_roles=[p.get('agent', 'unknown') for p in defense_team.get('players', [])[:5]],
            side='defense',
            forced_buy_type=BuyType[round_type.upper()] if round_type != 'full' else None
        )

        # Initialize players at spawn positions with loadouts
        self.players = {}
        spawn_positions = self._get_spawn_positions(session.map_name)

        # Get data loader for player profiles
        data_loader = get_data_loader()

        # Attack team
        for i, player in enumerate(attack_team.get('players', [])[:5]):
            pos = spawn_positions['attack'][i % len(spawn_positions['attack'])]
            loadout = attack_loadouts[i] if i < len(attack_loadouts) else None

            agent_name = player.get('agent', 'unknown')
            player_name = player.get('name', '')

            # AUTONOMOUS PLAYER: Load real stats from VCT data
            player_profile = data_loader.get_player_profile(player_name) if player_name else None

            if player_profile:
                # Use real VCT-derived stats
                aggression = player_profile.aggression
                clutch = player_profile.clutch_potential
                hs_rate = player_profile.headshot_rate if player_profile.headshot_rate > 0 else 0.25
                trade_awareness = min(0.9, player_profile.kd_ratio * 0.5) if player_profile.kd_ratio else 0.6
                # AUTONOMOUS: Preferred engagement distance (normalized to 0-1 map scale)
                # VCT data is in game units (~1500-2200), normalize to map fraction
                pref_distance = player_profile.avg_kill_distance / 10000.0 if player_profile.avg_kill_distance else 0.18
            else:
                # Fallback to random for unknown players
                aggression = random.uniform(0.3, 0.7)
                clutch = random.uniform(0.3, 0.7)
                hs_rate = 0.25
                trade_awareness = random.uniform(0.4, 0.8)
                pref_distance = 0.18  # Default mid-range

            # Create spike carrier tendencies based on role and VCT profile
            role = self._get_role_from_agent(agent_name)
            profile_dict = None
            if player_profile:
                profile_dict = {
                    'aggression': player_profile.aggression,
                    'first_kill_rate': player_profile.first_kill_rate,
                    'clutch_potential': player_profile.clutch_potential,
                    'first_death_rate': getattr(player_profile, 'first_death_rate', 0.1),
                }
            spike_tendencies = BehaviorAdapter.create_spike_carrier_tendencies(role, profile_dict)

            self.players[player['id']] = SimulatedPlayer(
                player_id=player['id'],
                team_id=session.attack_team_id,
                side='attack',
                x=pos[0],
                y=pos[1],
                agent=agent_name,
                has_spike=(i == 0),  # First player has spike
                weapon=loadout.weapon if loadout else WeaponDatabase.WEAPONS['classic'],
                armor=loadout.armor if loadout else WeaponDatabase.ARMOR['none'],
                shield=loadout.armor.shield_value if loadout else 0,
                loadout_value=loadout.total_value if loadout else 0,
                headshot_rate=hs_rate,  # AUTONOMOUS: Real player headshot rate
                preferred_engagement_distance=pref_distance,  # AUTONOMOUS: Preferred fight distance
                tendencies=PlayerTendencies(
                    base_aggression=aggression,  # AUTONOMOUS: Real player aggression
                    clutch_factor=clutch,  # AUTONOMOUS: Real player clutch factor
                    trade_awareness=trade_awareness,  # AUTONOMOUS: Derived from K/D
                ),
                # VCT-derived spike carrier behavior
                spike_carrier_tendencies=spike_tendencies,
                # P3 FIX: Set agent-specific ultimate cost
                ultimate_cost=AGENT_ULTIMATE_COSTS.get(agent_name, 7),
            )

            # Initialize ability state
            self.ability_system.initialize_player(player['id'], agent_name)

        # Defense team
        for i, player in enumerate(defense_team.get('players', [])[:5]):
            pos = spawn_positions['defense'][i % len(spawn_positions['defense'])]
            loadout = defense_loadouts[i] if i < len(defense_loadouts) else None

            agent_name = player.get('agent', 'unknown')
            player_name = player.get('name', '')

            # AUTONOMOUS PLAYER: Load real stats from VCT data
            player_profile = data_loader.get_player_profile(player_name) if player_name else None

            if player_profile:
                # Use real VCT-derived stats
                aggression = player_profile.aggression
                clutch = player_profile.clutch_potential
                hs_rate = player_profile.headshot_rate if player_profile.headshot_rate > 0 else 0.25
                trade_awareness = min(0.9, player_profile.kd_ratio * 0.5) if player_profile.kd_ratio else 0.6
                # AUTONOMOUS: Preferred engagement distance
                pref_distance = player_profile.avg_kill_distance / 10000.0 if player_profile.avg_kill_distance else 0.18
            else:
                # Fallback to random for unknown players
                aggression = random.uniform(0.3, 0.7)
                clutch = random.uniform(0.3, 0.7)
                hs_rate = 0.25
                trade_awareness = random.uniform(0.4, 0.8)
                pref_distance = 0.18

            # Create spike carrier tendencies for defense too (for spike transfer scenarios)
            role = self._get_role_from_agent(agent_name)
            profile_dict = None
            if player_profile:
                profile_dict = {
                    'aggression': player_profile.aggression,
                    'first_kill_rate': player_profile.first_kill_rate,
                    'clutch_potential': player_profile.clutch_potential,
                    'first_death_rate': getattr(player_profile, 'first_death_rate', 0.1),
                }
            spike_tendencies = BehaviorAdapter.create_spike_carrier_tendencies(role, profile_dict)

            self.players[player['id']] = SimulatedPlayer(
                player_id=player['id'],
                team_id=session.defense_team_id,
                side='defense',
                x=pos[0],
                y=pos[1],
                agent=agent_name,
                weapon=loadout.weapon if loadout else WeaponDatabase.WEAPONS['classic'],
                armor=loadout.armor if loadout else WeaponDatabase.ARMOR['none'],
                shield=loadout.armor.shield_value if loadout else 0,
                loadout_value=loadout.total_value if loadout else 0,
                headshot_rate=hs_rate,  # AUTONOMOUS: Real player headshot rate
                preferred_engagement_distance=pref_distance,  # AUTONOMOUS: Preferred fight distance
                tendencies=PlayerTendencies(
                    base_aggression=aggression,  # AUTONOMOUS: Real player aggression
                    clutch_factor=clutch,  # AUTONOMOUS: Real player clutch factor
                    trade_awareness=trade_awareness,  # AUTONOMOUS: Derived from K/D
                ),
                # VCT-derived spike carrier behavior
                spike_carrier_tendencies=spike_tendencies,
                # P3 FIX: Set agent-specific ultimate cost
                ultimate_cost=AGENT_ULTIMATE_COSTS.get(agent_name, 7),
            )

            self.ability_system.initialize_player(player['id'], agent_name)

        # Select strategies
        self._select_strategies(session.map_name, round_type)

        # Assign roles based on strategies
        self._assign_roles(attack_team, defense_team)

        # Initialize round state
        self.round_state = RoundState(
            attack_buy_type=round_type,
            defense_buy_type=round_type
        )

        # Load navigation grid for pathfinding (use v4 masks)
        self.pathfinder.load_nav_grid_from_v4(session.map_name)

        self.events = []
        self.snapshots = []
        self.spike_planted = False
        self.spike_site = None

        # SPIKE FLOW FIX: Reset site execute state
        self.site_execute_active = False
        self.target_site = None
        self.execute_start_time = None
        self.site_control_achieved = False

        # AI SYSTEM: Initialize information manager and AI behavior
        self.info_manager.reset()
        for player_id, player in self.players.items():
            self.info_manager.initialize_player(player_id, player.side)
        self.ai_behavior = AIBehaviorIntegration(self.info_manager)

        # P3 FIX: Initialize ultimate orbs for this map
        orb_positions = self.ORB_POSITIONS.get(session.map_name, self.ORB_POSITIONS['default'])
        self.orbs_available = list(orb_positions)  # Copy the positions
        self.sound_events = []

        return self._build_state(session, session.current_time_ms)

    def _initialize_economies(self, round_type: str):
        """Initialize team economies based on round type."""
        if round_type == 'pistol':
            self.attack_economy = TeamEconomy(credits=[800] * 5)
            self.defense_economy = TeamEconomy(credits=[800] * 5)
        elif round_type == 'eco':
            self.attack_economy = TeamEconomy(credits=[2000] * 5, loss_streak=1)
            self.defense_economy = TeamEconomy(credits=[2000] * 5, loss_streak=1)
        elif round_type == 'force':
            self.attack_economy = TeamEconomy(credits=[3000] * 5, loss_streak=2)
            self.defense_economy = TeamEconomy(credits=[3000] * 5, loss_streak=2)
        elif round_type == 'half':
            self.attack_economy = TeamEconomy(credits=[4000] * 5)
            self.defense_economy = TeamEconomy(credits=[4000] * 5)
        else:  # full
            self.attack_economy = TeamEconomy(credits=[5000] * 5)
            self.defense_economy = TeamEconomy(credits=[5000] * 5)

    def _select_strategies(self, map_name: str, round_type: str):
        """Select strategies for both teams."""
        attack_credits = self.attack_economy.total_credits if self.attack_economy else 25000
        defense_credits = self.defense_economy.total_credits if self.defense_economy else 25000

        self.attack_strategy = self.strategy_coordinator.select_strategy(
            team_id="attack",
            map_name=map_name,
            side='attack',
            round_type=round_type,
            team_credits=attack_credits,
            round_number=0
        )

        # Create a separate coordinator for defense
        defense_coordinator = StrategyCoordinator()
        self.defense_strategy = defense_coordinator.select_strategy(
            team_id="defense",
            map_name=map_name,
            side='defense',
            round_type=round_type,
            team_credits=defense_credits,
            round_number=0
        )

    def _assign_roles(self, attack_team: Dict, defense_team: Dict):
        """Assign roles to players based on selected strategies."""
        # Attack roles
        if self.attack_strategy:
            attack_players = [
                {'player_id': p['id'], 'agent': p.get('agent', 'unknown')}
                for p in attack_team.get('players', [])[:5]
            ]
            attack_assignments = self.strategy_coordinator.assign_roles(
                attack_players, self.attack_strategy
            )
            for player_id, assignment in attack_assignments.items():
                if player_id in self.players:
                    self.players[player_id].role = assignment.role

        # Defense roles (use separate coordinator)
        if self.defense_strategy:
            defense_players = [
                {'player_id': p['id'], 'agent': p.get('agent', 'unknown')}
                for p in defense_team.get('players', [])[:5]
            ]
            defense_coordinator = StrategyCoordinator()
            defense_coordinator.current_strategy = self.defense_strategy
            defense_assignments = defense_coordinator.assign_roles(
                defense_players, self.defense_strategy
            )
            for player_id, assignment in defense_assignments.items():
                if player_id in self.players:
                    self.players[player_id].role = assignment.role

    async def _load_team(self, team_id: str) -> Dict[str, Any]:
        """Load team with players from database."""
        result = await self.db.execute(
            select(Team)
            .options(selectinload(Team.players))
            .where(Team.id == team_id)
        )
        team = result.scalar_one_or_none()
        if not team:
            # Return default team structure
            return {
                'id': team_id,
                'name': team_id,
                'players': [
                    {'id': f'{team_id}_player_{i}', 'name': f'Player {i}', 'agent': 'unknown'}
                    for i in range(5)
                ]
            }

        return {
            'id': team.id,
            'name': team.name,
            'players': [
                {'id': p.id, 'name': p.name, 'agent': p.role or 'unknown'}
                for p in team.players
            ]
        }

    async def _load_map_config(self, map_name: str) -> Optional[MapConfig]:
        """Load map configuration from database."""
        result = await self.db.execute(
            select(MapConfig).where(MapConfig.map_name == map_name)
        )
        return result.scalar_one_or_none()

    def _get_spawn_positions(self, map_name: str) -> Dict[str, List[Tuple[float, float]]]:
        """P1 FIX: Get map-specific spawn positions (normalized coordinates)."""
        map_key = map_name.lower() if map_name else 'ascent'
        map_data = self.MAP_DATA.get(map_key, self.DEFAULT_MAP_DATA)
        return map_data['spawns']

    def _get_map_sites(self) -> Dict[str, Dict[str, Any]]:
        """P1 FIX: Get site positions for the current map.

        Returns dict with site name -> {'center': (x,y), 'radius': float}
        Supports 2-site maps (A, B) and 3-site maps like Haven/Lotus (A, B, C).
        """
        map_data = self.MAP_DATA.get(self.map_name, self.DEFAULT_MAP_DATA)
        return map_data['sites']

    def _get_current_phase(self, time_ms: int) -> str:
        """Determine current game phase based on time.

        Returns one of 6 phases:
        - Pre-plant: 'early' (0-15s), 'mid' (15-50s), 'late' (50s+)
        - Post-plant: 'post_plant_early' (0-10s), 'post_plant_mid' (10-25s), 'post_plant_late' (25s+)
        """
        if self.spike_planted:
            # Calculate time since plant for sub-phase determination
            plant_elapsed = time_ms - (self.spike_plant_time_ms if hasattr(self, 'spike_plant_time_ms') else time_ms)
            if plant_elapsed < self.PHASE_TIMINGS['post_plant_early']:
                return 'post_plant_early'  # Setup holds phase
            elif plant_elapsed < self.PHASE_TIMINGS['post_plant_mid']:
                return 'post_plant_mid'    # Hold phase
            else:
                return 'post_plant_late'   # Late retake pressure

        # Pre-plant phases
        if time_ms < self.PHASE_TIMINGS['early']:
            return 'early'
        elif time_ms < self.PHASE_TIMINGS['mid']:
            return 'mid'
        return 'late'

    async def advance(self, session: SimulationSession, ticks: int = 1) -> SimulationState:
        """Advance simulation by specified number of ticks."""
        current_time = session.current_time_ms

        for _ in range(ticks):
            current_time += self.TICK_DURATION_MS

            # Check round end conditions
            if self._check_round_end(current_time):
                break

            phase = self._get_current_phase(current_time)

            # Update round state trade window
            self.round_state.update_trade_window(current_time)

            # Update behavior modifiers for all players (now with phase for VCT behaviors)
            self._update_behaviors(current_time, phase)

            # AI SYSTEM: Update information state and make decisions
            if self.use_ai_decisions:
                self._update_information_state(current_time)
                self._update_ai_decisions(current_time, phase)

            # Check and apply ability usage
            self._process_abilities(current_time, phase)

            # Update player positions
            await self._update_positions(session, current_time, phase)

            # Apply ability effects (flash, smoke damage, etc.)
            self._apply_ability_effects(current_time)

            # Check for combat (pass phase for engagement rate calculation)
            self._resolve_combat(current_time, phase)

            # P2 FIX: Check for trade opportunities after combat
            self._check_trade_opportunities(current_time)

            # P3 FIX: Process sound cues
            self._process_sound_cues(current_time)

            # P3 FIX: Check orb collection
            self._check_orb_collection(current_time)

            # SPIKE FLOW FIX: Check site execute decision
            # This determines when attackers should commit to a site
            if not self.spike_planted:
                self._check_site_execute_decision(current_time)
                self._check_spike_plant(current_time)
            else:
                # P0 FIX: Check for spike defuse attempts
                self._check_spike_defuse(current_time)

        return self._build_state(session, current_time)

    def _update_behaviors(self, time_ms: int, phase: str = 'mid'):
        """Update behavior modifiers for all players based on round state.

        Args:
            time_ms: Current time in milliseconds
            phase: Current game phase for VCT behavior modifiers
        """
        for player_id, player in self.players.items():
            if not player.is_alive:
                continue

            # Calculate distance to last kill for trade detection
            distance_to_kill = None
            if self.round_state.last_kill_position:
                dx = player.x - self.round_state.last_kill_position[0]
                dy = player.y - self.round_state.last_kill_position[1]
                distance_to_kill = math.sqrt(dx * dx + dy * dy)

            player.behavior_modifiers = BehaviorAdapter.calculate_behavior_modifiers(
                player_id=player_id,
                player_team=player.side,
                round_state=self.round_state,
                time_ms=time_ms,
                tendencies=player.tendencies,
                distance_to_last_kill=distance_to_kill,
                phase=phase
            )

    def _process_abilities(self, time_ms: int, phase: str):
        """Process ability usage for all players."""
        for player_id, player in self.players.items():
            if not player.is_alive:
                continue

            # P3 FIX: Check ultimate usage first
            if player.ultimate_ready:
                ult_used = self._check_ultimate_usage(player, time_ms, phase)
                if ult_used:
                    continue  # Used ultimate this tick, skip regular abilities

            # Check if player should use an ability
            ability_result = self.ability_system.should_use_ability(
                player_id=player_id,
                time_ms=time_ms,
                phase=phase,
                round_state=self.round_state,
                player_position=(player.x, player.y),
                team=player.side
            )

            if ability_result:
                ability, target_pos = ability_result
                active = self.ability_system.use_ability(
                    player_id=player_id,
                    ability=ability,
                    position=target_pos,
                    time_ms=time_ms,
                    team=player.side
                )

                # Record ability event
                self.events.append(SimulationEvent(
                    timestamp_ms=time_ms,
                    event_type='ability',
                    player_id=player_id,
                    position_x=target_pos[0],
                    position_y=target_pos[1],
                    details={
                        'ability_name': ability.name,
                        'ability_category': ability.category.value,
                        'duration_ms': ability.duration_ms,
                    }
                ))

    def _check_ultimate_usage(self, player: SimulatedPlayer, time_ms: int, phase: str) -> bool:
        """P3 FIX: Check if player should use their ultimate.

        Returns True if ultimate was used.
        """
        # Don't use ult if in action
        if player.is_in_action:
            return False

        # Calculate situation factors
        allies_alive = sum(1 for p in self.players.values()
                          if p.is_alive and p.side == player.side)
        enemies_alive = sum(1 for p in self.players.values()
                           if p.is_alive and p.side != player.side)

        # Ultimate usage probability based on situation
        use_probability = 0.0

        # Clutch situations - save ult for impact
        if allies_alive == 1 and enemies_alive >= 2:
            use_probability = 0.4  # Clutch - might need it

        # Man advantage - use aggressively
        elif allies_alive > enemies_alive:
            use_probability = 0.15  # Save unless good opportunity

        # Even or disadvantage
        else:
            use_probability = 0.25  # Moderate chance

        # Phase-based adjustments
        if phase == 'post_plant':
            if player.side == 'defense':
                use_probability += 0.3  # Defenders need ult to retake
            else:
                use_probability += 0.15  # Attackers hold with ult
        elif phase == 'opening' and time_ms < 15000:
            use_probability *= 0.3  # Don't waste ult early

        # Time pressure for attackers
        if player.side == 'attack' and not self.spike_planted:
            if time_ms > 85000:  # Less than 15 seconds
                use_probability += 0.4  # Must execute

        # Agent-specific ultimate timing
        # Some ults are better at certain times
        agent_lower = player.agent.lower()
        if agent_lower in ['sage', 'phoenix', 'clove']:
            # Revive ults - use when teammate died recently
            if self.round_state.last_kill_time_ms and time_ms - self.round_state.last_kill_time_ms < 10000:
                if self.round_state.last_killed_team == player.side:
                    use_probability += 0.5  # Teammate just died

        elif agent_lower in ['sova', 'fade', 'cypher']:
            # Info ults - use mid-round or post-plant
            if phase in ['mid_round', 'post_plant', 'retake']:
                use_probability += 0.2

        elif agent_lower in ['raze', 'jett', 'chamber']:
            # Kill ults - use when engaging
            nearby_enemies = sum(1 for p in self.players.values()
                                if p.is_alive and p.side != player.side
                                and math.sqrt((p.x - player.x)**2 + (p.y - player.y)**2) < 0.2)
            if nearby_enemies > 0:
                use_probability += 0.3

        elif agent_lower in ['brimstone', 'killjoy', 'breach']:
            # Area denial ults - use post-plant
            if phase == 'post_plant' and self.spike_site:
                use_probability += 0.4

        # Roll the dice
        if random.random() < use_probability:
            # Use the ultimate!
            player.use_ultimate()

            # Record ultimate event
            self.events.append(SimulationEvent(
                timestamp_ms=time_ms,
                event_type='ultimate',
                player_id=player.player_id,
                position_x=player.x,
                position_y=player.y,
                details={
                    'agent': player.agent,
                    'phase': phase,
                    'allies_alive': allies_alive,
                    'enemies_alive': enemies_alive,
                }
            ))
            return True

        return False

    def _apply_ability_effects(self, time_ms: int):
        """Apply ability effects to players."""
        effects = self.ability_system.update_effects(time_ms, self.players)

        for player_id, player_effects in effects.items():
            player = self.players.get(player_id)
            if not player or not player.is_alive:
                continue

            # Apply flash
            if player_effects.get('is_flashed', False):
                player.is_flashed = True
                player.flash_end_ms = player_effects.get('flash_end_ms', time_ms + 2000)

            # Clear flash if expired
            if player.is_flashed and time_ms >= player.flash_end_ms:
                player.is_flashed = False

            # Apply slow
            player.is_slowed = player_effects.get('is_slowed', False)

            # Apply reveal
            player.is_revealed = player_effects.get('revealed', False)

            # Apply damage
            damage = player_effects.get('damage_taken', 0)
            if damage > 0:
                self._apply_damage(player, int(damage), time_ms)

            # Apply healing
            healing = player_effects.get('healing', 0)
            if healing > 0:
                player.health = min(100, player.health + int(healing))

            # Handle molly repositioning (was dead code - now integrated)
            if player_effects.get('should_reposition', False):
                danger_pos = player_effects.get('reposition_away_from')
                if danger_pos:
                    # Find safe position away from molly
                    safe_pos = self.ability_system.find_safe_position(
                        current_pos=(player.x, player.y),
                        danger_pos=danger_pos,
                        unsafe_radius=0.12
                    )
                    # Override player's target to move away from danger
                    player.reposition_target = safe_pos

    def _apply_damage(
        self,
        player: SimulatedPlayer,
        damage: int,
        time_ms: int,
        source_player_id: Optional[str] = None
    ):
        """Apply damage to a player, accounting for shield."""
        if player.shield > 0:
            shield_damage = min(damage, player.shield)
            player.shield -= shield_damage
            remaining_damage = damage - shield_damage
            player.health -= remaining_damage
        else:
            player.health -= damage

        if player.health <= 0:
            player.health = 0
            player.is_alive = False

    async def _update_positions(self, session: SimulationSession, time_ms: int, phase: str):
        """Update all player positions based on patterns, strategy, and pathfinding."""
        alive_attack = [p for p in self.players.values() if p.side == 'attack' and p.is_alive]
        alive_defense = [p for p in self.players.values() if p.side == 'defense' and p.is_alive]

        # Get strategy-based target positions
        for player in alive_attack + alive_defense:
            # P1 FIX: Save previous position for movement tracking
            player.prev_x = player.x
            player.prev_y = player.y

            # P0 FIX: Skip movement if player is planting or defusing (immobilized)
            if player.is_in_action:
                player.moved_this_tick = False
                continue

            # Get target from strategy coordinator
            strategy_target = self.strategy_coordinator.get_player_target_position(
                player.player_id, time_ms, (player.x, player.y)
            )

            # AI SYSTEM: Override or blend movement based on AI decisions
            if self.use_ai_decisions and self.ai_behavior:
                ai_decision = self.ai_behavior.get_decision_for_player(player.player_id)
                if ai_decision and ai_decision.target_position:
                    sites = self._get_map_sites()
                    site_positions = {name: data['center'] for name, data in sites.items()}

                    # Get AI-determined movement target
                    ai_target = self.ai_behavior.get_movement_target(
                        player.player_id, (player.x, player.y), site_positions
                    )

                    if ai_target:
                        # Blend AI target with strategy target
                        # AI decisions should have strong influence for combat actions
                        from .ai_decision_system import Decision
                        decision_type = ai_decision.decision

                        if decision_type == Decision.PEEK:
                            # Peeking overrides strategy - move toward known enemy
                            strategy_target = ai_target
                        elif decision_type == Decision.ADVANCE:
                            # Advance uses mostly AI target
                            strategy_target = (
                                ai_target[0] * 0.7 + (strategy_target[0] if strategy_target else ai_target[0]) * 0.3,
                                ai_target[1] * 0.7 + (strategy_target[1] if strategy_target else ai_target[1]) * 0.3
                            )
                        elif decision_type == Decision.RETREAT:
                            # Retreat uses AI target
                            strategy_target = ai_target
                        elif decision_type == Decision.HOLD:
                            # Hold - stay mostly in place
                            strategy_target = (player.x, player.y)

            # Apply behavior modifiers to movement
            behavior = player.behavior_modifiers or BehaviorModifiers()

            # P3 FIX: Sound-reactive movement
            # If player heard enemy footsteps, adjust behavior
            sound_reaction_target = None
            force_walking = False

            if player.heard_enemy_sound and player.last_sound_location:
                sound_x, sound_y = player.last_sound_location
                dist_to_sound = math.sqrt(
                    (player.x - sound_x) ** 2 + (player.y - sound_y) ** 2
                )

                # Calculate relative numbers
                allies_alive = sum(1 for p in self.players.values()
                                   if p.is_alive and p.side == player.side)
                enemies_alive = sum(1 for p in self.players.values()
                                    if p.is_alive and p.side != player.side)

                # Decision: investigate or avoid?
                if allies_alive >= enemies_alive and behavior.aggression > 0:
                    # Advantage or even - investigate the sound
                    # Move toward sound location but slow down
                    sound_reaction_target = (sound_x, sound_y)
                    force_walking = True  # Go silent when investigating
                elif dist_to_sound < 0.15:
                    # Close contact - react based on aggression
                    if behavior.aggression > 0.3:
                        # Aggressive - push the sound
                        sound_reaction_target = (sound_x, sound_y)
                    else:
                        # Passive - back off from sound
                        away_x = player.x + (player.x - sound_x) * 0.5
                        away_y = player.y + (player.y - sound_y) * 0.5
                        sound_reaction_target = (away_x, away_y)
                    force_walking = True
                else:
                    # Far sound, outnumbered - go silent but continue strategy
                    force_walking = True

            # SPIKE FLOW FIX: Site execute movement override for attackers
            site_execute_target = None
            if (player.side == 'attack' and self.site_execute_active and
                self.target_site and not self.spike_planted):
                # Attackers should move toward target site during execute
                sites = self._get_map_sites()
                if self.target_site in sites:
                    site_data = sites[self.target_site]
                    site_center = site_data['center']
                    site_radius = site_data.get('radius', 0.08)

                    # Distance to site
                    dist_to_site = math.sqrt(
                        (player.x - site_center[0])**2 + (player.y - site_center[1])**2
                    )

                    # If not yet on site, move toward it
                    if dist_to_site > site_radius:
                        # Add some variance to avoid clustering
                        offset_x = random.uniform(-site_radius, site_radius)
                        offset_y = random.uniform(-site_radius, site_radius)
                        site_execute_target = (
                            site_center[0] + offset_x,
                            site_center[1] + offset_y
                        )

            # Adjust movement based on aggression
            if strategy_target:
                target_x, target_y = strategy_target

                # SPIKE FLOW FIX: Override target if site execute is active
                if site_execute_target:
                    # Heavily weight toward site target during execute
                    target_x = site_execute_target[0] * 0.8 + target_x * 0.2
                    target_y = site_execute_target[1] * 0.8 + target_y * 0.2

                # P3 FIX: Override target if reacting to sound
                if sound_reaction_target:
                    # Blend sound reaction with strategy (70% sound, 30% strategy)
                    target_x = sound_reaction_target[0] * 0.7 + target_x * 0.3
                    target_y = sound_reaction_target[1] * 0.7 + target_y * 0.3

                # MOLLY REPOSITIONING: Highest priority - survival
                # If in a molly, MUST move away immediately
                if player.reposition_target:
                    # Full override - getting out of molly is critical
                    target_x, target_y = player.reposition_target
                    player.is_running = True  # Sprint out of danger
                    # Clear the target once we've started moving
                    if abs(player.x - target_x) < 0.02 and abs(player.y - target_y) < 0.02:
                        player.reposition_target = None

                # Add position variance based on behavior
                variance = 0.02 * (1.0 - behavior.aggression)  # Less variance when aggressive
                target_x += random.uniform(-variance, variance)
                target_y += random.uniform(-variance, variance)

                # Calculate movement speed
                base_speed = 0.015  # Base movement per tick
                speed = base_speed * behavior.movement_speed

                # Slow effect reduces speed
                if player.is_slowed:
                    speed *= 0.5

                # P3 FIX: Slow down when reacting to sound
                if force_walking:
                    speed *= 0.6  # Walk speed is slower

                # P1 FIX: Determine if running or walking based on behavior
                # P3 FIX: Force walking if heard enemy sound nearby
                if force_walking:
                    player.is_running = False  # Go silent
                else:
                    # Lower aggression = more likely to walk (quieter, more accurate)
                    player.is_running = behavior.aggression > 0.4 or random.random() > 0.3

                # Move toward target
                dx = target_x - player.x
                dy = target_y - player.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > speed:
                    # Normalize and scale by speed
                    player.x += (dx / dist) * speed
                    player.y += (dy / dist) * speed
                else:
                    player.x = target_x
                    player.y = target_y

            # P1 FIX: Track if player moved this tick
            move_dist = math.sqrt(
                (player.x - player.prev_x) ** 2 + (player.y - player.prev_y) ** 2
            )
            player.moved_this_tick = move_dist > 0.001  # Small threshold for floating point
            if player.moved_this_tick:
                player.last_move_ms = time_ms

        # Also get predictions from pattern matcher for additional context
        attack_data = [
            {'player_id': p.player_id, 'x': p.x, 'y': p.y, 'is_alive': p.is_alive}
            for p in alive_attack
        ]
        attack_predictions = await self.pattern_matcher.predict_team_positions(
            attack_data, session.attack_team_id, time_ms,
            session.map_name, 'attack', phase
        )

        defense_data = [
            {'player_id': p.player_id, 'x': p.x, 'y': p.y, 'is_alive': p.is_alive}
            for p in alive_defense
        ]
        defense_predictions = await self.pattern_matcher.predict_team_positions(
            defense_data, session.defense_team_id, time_ms,
            session.map_name, 'defense', phase
        )

        # Blend strategy movement with pattern predictions
        for pred in attack_predictions + defense_predictions:
            player = self.players.get(pred.player_id)
            if player and player.is_alive:
                # Blend current position toward prediction with low weight
                blend = 0.1  # 10% influence from patterns
                player.x = player.x * (1 - blend) + pred.position[0] * blend
                player.y = player.y * (1 - blend) + pred.position[1] * blend

    def _resolve_combat(self, time_ms: int, phase: str = 'mid_round'):
        """Check for and resolve combat encounters using weapon mechanics."""
        alive_attack = [p for p in self.players.values() if p.side == 'attack' and p.is_alive]
        alive_defense = [p for p in self.players.values() if p.side == 'defense' and p.is_alive]

        # Track engagements this tick to avoid duplicate processing
        engaged_pairs = set()

        # Check each attacker vs each defender
        for attacker in alive_attack:
            for defender in alive_defense:
                if not defender.is_alive or not attacker.is_alive:
                    continue

                pair_key = tuple(sorted([attacker.player_id, defender.player_id]))
                if pair_key in engaged_pairs:
                    continue

                distance = math.sqrt(
                    (attacker.x - defender.x) ** 2 +
                    (attacker.y - defender.y) ** 2
                )

                # Check if within engagement range
                if distance > self.ENGAGEMENT_DISTANCE:
                    continue

                # Check line of sight (including smoke check)
                if not self._has_line_of_sight(
                    (attacker.x, attacker.y),
                    (defender.x, defender.y),
                    time_ms
                ):
                    continue

                engaged_pairs.add(pair_key)

                # Calculate engagement result using weapon system
                self._resolve_engagement(attacker, defender, distance, time_ms, phase)

    def _check_trade_opportunities(self, time_ms: int):
        """P2 FIX: Check and process trade opportunities.

        When a player is killed, nearby teammates have a window to 'trade'
        by killing the enemy who got the kill. This simulates the refrag
        mechanic that's crucial in VALORANT.

        Trade probability based on distance to kill:
        - Close (<10% of map): 70% base chance to attempt trade
        - Medium (10-20%): 40% base chance
        - Far (20-30%): 15% base chance
        """
        if not self.pending_trade:
            return

        trade = self.pending_trade
        time_since_kill = time_ms - trade['kill_time_ms']

        # Expire trade window
        if time_since_kill > self.TRADE_WINDOW_MS:
            self.pending_trade = None
            return

        # Find the killer
        killer = self.players.get(trade['killer_id'])
        if not killer or not killer.is_alive:
            self.pending_trade = None  # Killer died, trade already happened
            return

        kill_pos = trade['kill_position']
        victim_team = trade['victim_team']

        # Find nearby teammates of the victim who could trade
        potential_traders = [
            p for p in self.players.values()
            if p.side == victim_team and p.is_alive and not p.is_planting and not p.is_defusing
        ]

        for trader in potential_traders:
            dist_to_kill = math.sqrt(
                (trader.x - kill_pos[0]) ** 2 + (trader.y - kill_pos[1]) ** 2
            )

            # Calculate base trade attempt probability (VALIDATED)
            # Source: validated_parameters.py - calibrated for 20% trade rate
            trade_params = TradeParams()
            if dist_to_kill < self.TRADE_DISTANCE_CLOSE:
                base_trade_chance = trade_params.CLOSE_TRADE_RATE / 5  # ~7% per tick when close
            elif dist_to_kill < self.TRADE_DISTANCE_MEDIUM:
                base_trade_chance = trade_params.MEDIUM_TRADE_RATE / 5  # ~4% per tick at medium range
            elif dist_to_kill < self.TRADE_DISTANCE_FAR:
                base_trade_chance = trade_params.FAR_TRADE_RATE / 5  # ~1.6% per tick when far
            else:
                continue  # Too far to trade

            # Modify by player's trade awareness stat
            trade_awareness = getattr(trader.tendencies, 'trade_awareness', 0.5) if trader.tendencies else 0.5
            trade_chance = base_trade_chance * (0.5 + trade_awareness)

            # Higher chance early in trade window (immediate reaction)
            if time_since_kill < 1000:  # First second
                trade_chance *= 1.5

            if random.random() < trade_chance:
                # Attempt trade - move toward killer and engage
                dist_to_killer = math.sqrt(
                    (trader.x - killer.x) ** 2 + (trader.y - killer.y) ** 2
                )

                # If close enough to killer, force engagement
                if dist_to_killer < self.ENGAGEMENT_DISTANCE * 1.5:
                    # Check LOS
                    if self._has_line_of_sight((trader.x, trader.y), (killer.x, killer.y), time_ms):
                        # Trade attempt - trader has advantage (peeking to trade)
                        self._resolve_trade_engagement(trader, killer, dist_to_killer, time_ms)
                        return  # One trade attempt per tick
                else:
                    # Move trader aggressively toward killer
                    move_speed = 0.03  # Fast push
                    dx = killer.x - trader.x
                    dy = killer.y - trader.y
                    dist = max(0.001, math.sqrt(dx * dx + dy * dy))
                    trader.x += (dx / dist) * move_speed
                    trader.y += (dy / dist) * move_speed
                    trader.moved_this_tick = True
                    trader.is_running = True

    def _resolve_trade_engagement(
        self,
        trader: 'SimulatedPlayer',
        target: 'SimulatedPlayer',
        distance: float,
        time_ms: int
    ):
        """P2 FIX: Resolve a trade engagement with advantage to the trader.

        The trader (peeking to avenge teammate) has significant advantage:
        - 25% bonus win probability (they're expecting the fight)
        - Already moving/peeking so peeker's advantage applies
        """
        # Trader has significant advantage in trade situation
        trader_weapon = trader.weapon if trader.weapon else WeaponDatabase.WEAPONS['classic']
        target_weapon = target.weapon if target.weapon else WeaponDatabase.WEAPONS['classic']

        # Base accuracy from weapons
        trader_acc = trader_weapon.get_accuracy_at_distance(distance)
        target_acc = target_weapon.get_accuracy_at_distance(distance)

        # Apply movement modifiers
        trader_acc *= trader.movement_accuracy_modifier
        target_acc *= target.movement_accuracy_modifier

        # P2: Trade advantage (VALIDATED)
        # Source: Pro play analysis, KAST% data
        # - Trader is prepared for fight: +20% bonus
        # - Killer is repositioning/reloading: -15% penalty
        trade_bonus = VALIDATED_PARAMS['trader_bonus']  # 1.20
        post_kill_penalty = VALIDATED_PARAMS['post_kill_penalty']  # 0.85

        trader_win_chance = 0.5 * trader_acc * trade_bonus
        target_win_chance = 0.5 * target_acc * post_kill_penalty

        # Normalize
        total = trader_win_chance + target_win_chance
        if total > 0:
            trader_win_chance /= total
        else:
            trader_win_chance = 0.5

        if random.random() < trader_win_chance:
            # Trader wins - kills the target (trade successful)
            target.health = 0
            target.is_alive = False
            target.deaths += 1
            trader.kills += 1

            # Record as trade kill
            kill_event = self.round_state.record_kill(
                time_ms=time_ms,
                killer_id=trader.player_id,
                killer_team=trader.side,
                victim_id=target.player_id,
                victim_team=target.side,
                position=(target.x, target.y),
                weapon=trader_weapon.name,
                is_headshot=random.random() < trader.headshot_rate  # AUTONOMOUS: Player-specific HS rate
            )

            self.events.append(SimulationEvent(
                timestamp_ms=time_ms,
                event_type='kill',
                player_id=target.player_id,
                target_id=target.player_id,
                position_x=target.x,
                position_y=target.y,
                details={
                    'killer_id': trader.player_id,
                    'headshot': kill_event.is_headshot,
                    'weapon': trader_weapon.name,
                    'distance': distance,
                    'is_first_blood': kill_event.is_first_blood,
                    'is_trade': True,  # Mark explicitly as trade
                    'damage_dealt': 150,
                }
            ))

            # P3 FIX: Grant ultimate points on trade kill
            trader.grant_ultimate_point("kill")
            target.grant_ultimate_point("death")

            self.pending_trade = None  # Trade completed
        else:
            # Target wins - trader dies attempting trade
            trader.health = 0
            trader.is_alive = False
            trader.deaths += 1
            target.kills += 1

            self.round_state.record_kill(
                time_ms=time_ms,
                killer_id=target.player_id,
                killer_team=target.side,
                victim_id=trader.player_id,
                victim_team=trader.side,
                position=(trader.x, trader.y),
                weapon=target_weapon.name,
                is_headshot=random.random() < target.headshot_rate * 0.8  # AUTONOMOUS: Player HS rate (reduced for failed trade)
            )

            self.events.append(SimulationEvent(
                timestamp_ms=time_ms,
                event_type='kill',
                player_id=trader.player_id,
                target_id=trader.player_id,
                position_x=trader.x,
                position_y=trader.y,
                details={
                    'killer_id': target.player_id,
                    'headshot': False,
                    'weapon': target_weapon.name,
                    'distance': distance,
                    'is_first_blood': False,
                    'is_trade': False,
                    'damage_dealt': 150,
                }
            ))

            # P3 FIX: Grant ultimate points on failed trade kill
            target.grant_ultimate_point("kill")
            trader.grant_ultimate_point("death")
            # Don't clear pending_trade - another teammate might try

    def _process_sound_cues(self, time_ms: int):
        """P3 FIX: Process sound cues for all players.

        Running footsteps are audible within SOUND_RANGE_RUNNING.
        Walking is silent. Sound cues affect enemy awareness.
        """
        # Reset sound flags each tick
        for player in self.players.values():
            player.made_sound_this_tick = False
            player.heard_enemy_sound = False
            player.last_sound_location = None

        # Determine which players made sounds
        for player in self.players.values():
            if not player.is_alive:
                continue

            # Running players make sound
            if player.moved_this_tick and player.is_running:
                player.made_sound_this_tick = True

                # Check which enemies can hear this sound
                for enemy in self.players.values():
                    if not enemy.is_alive or enemy.side == player.side:
                        continue

                    distance = math.sqrt(
                        (player.x - enemy.x) ** 2 + (player.y - enemy.y) ** 2
                    )

                    if distance <= self.SOUND_RANGE_RUNNING:
                        enemy.heard_enemy_sound = True
                        enemy.last_sound_location = (player.x, player.y)

                        # Record sound event for debugging/analysis
                        self.sound_events.append({
                            'time_ms': time_ms,
                            'source_id': player.player_id,
                            'source_position': (player.x, player.y),
                            'listener_id': enemy.player_id,
                            'listener_position': (enemy.x, enemy.y),
                            'sound_type': 'footstep',
                            'distance': distance
                        })

    def _check_orb_collection(self, time_ms: int):
        """P3 FIX: Check if any player picks up an ultimate orb.

        Orbs are collected when a player gets within pickup range.
        Grants +1 ultimate point.
        """
        ORB_PICKUP_RANGE = 0.03  # 3% of map width (pretty close)

        orbs_to_remove = []

        for orb_pos in self.orbs_available:
            for player in self.players.values():
                if not player.is_alive:
                    continue

                distance = math.sqrt(
                    (player.x - orb_pos[0]) ** 2 + (player.y - orb_pos[1]) ** 2
                )

                if distance <= ORB_PICKUP_RANGE:
                    # Player picks up the orb!
                    orbs_to_remove.append(orb_pos)
                    player.ultimate_orbs_collected += 1
                    ult_ready = player.grant_ultimate_point("orb")

                    # Record orb pickup event
                    self.events.append(SimulationEvent(
                        timestamp_ms=time_ms,
                        event_type='orb_pickup',
                        player_id=player.player_id,
                        position_x=orb_pos[0],
                        position_y=orb_pos[1],
                        details={
                            'agent': player.agent,
                            'ultimate_points': player.ultimate_points,
                            'ultimate_ready': ult_ready
                        }
                    ))

                    # Only one player can pick up each orb
                    break

        # Remove collected orbs
        for orb in orbs_to_remove:
            if orb in self.orbs_available:
                self.orbs_available.remove(orb)

    def _build_combat_profile(self, player: 'SimulatedPlayer') -> PlayerCombatProfile:
        """Build a PlayerCombatProfile from SimulatedPlayer attributes.

        If the player has a pre-configured combat_profile (from GRID data),
        use that. Otherwise, derive one from player stats.
        """
        # Use pre-configured profile if available
        if player.combat_profile is not None:
            return player.combat_profile

        # Derive profile from player stats
        # Base skill from headshot rate (pro HS rate ~25-30%, avg ~18%)
        hs_rate = player.headshot_rate
        skill_from_hs = min(1.0, max(0.0, (hs_rate - 0.15) / 0.20))  # 0.15=0, 0.35=1

        # Get tendencies if available
        tendencies = player.tendencies

        # Extract crosshair placement from aggression
        if tendencies:
            aggression = tendencies.aggression if hasattr(tendencies, 'aggression') else 0.5
            utility_usage = tendencies.utility_usage if hasattr(tendencies, 'utility_usage') else 0.5
        else:
            aggression = 0.5
            utility_usage = 0.5

        # Map skill to reaction time: 0.0 → 280ms, 1.0 → 160ms
        base_reaction = 280 - (skill_from_hs * 120)

        # Crosshair placement from skill and aggression
        crosshair_placement = 0.40 + (skill_from_hs * 0.35) + (aggression * 0.20)

        # Spray control from skill
        spray_control = 0.30 + (skill_from_hs * 0.60)

        # Counter-strafe skill
        counter_strafe = 0.45 + (skill_from_hs * 0.50)

        # First shot discipline
        first_shot_discipline = 0.50 + (utility_usage * 0.25) + (skill_from_hs * 0.20)

        return PlayerCombatProfile(
            base_reaction_ms=base_reaction,
            reaction_variance=40 - (skill_from_hs * 20),
            crosshair_placement=min(0.95, crosshair_placement),
            headshot_rate=hs_rate,
            spray_control=min(0.95, spray_control),
            first_shot_discipline=min(0.95, first_shot_discipline),
            counter_strafe_skill=min(0.95, counter_strafe),
            clutch_factor=0.9 + (skill_from_hs * 0.2),
            peek_aggression=aggression
        )

    def _get_movement_state(self, player: 'SimulatedPlayer') -> MovementState:
        """Convert player movement flags to MovementState enum."""
        if player.is_running:
            return MovementState.RUNNING
        elif player.moved_this_tick:
            # Moved but not running = walking or counter-strafing
            return MovementState.WALKING
        else:
            return MovementState.STATIONARY

    def _has_line_of_sight(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
        time_ms: int
    ) -> bool:
        """Check line of sight, accounting for smokes."""
        # Check pathfinder LOS first
        if not self.pathfinder.has_line_of_sight(pos1, pos2):
            return False

        # Check if line passes through smoke
        smokes = self.ability_system.get_active_smokes(time_ms)
        for smoke_pos, smoke_radius in smokes:
            # Simple line-circle intersection check
            if self._line_intersects_circle(pos1, pos2, smoke_pos, smoke_radius):
                return False

        return True

    def _line_intersects_circle(
        self,
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        circle_center: Tuple[float, float],
        circle_radius: float
    ) -> bool:
        """Check if a line segment intersects a circle."""
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]

        fx = line_start[0] - circle_center[0]
        fy = line_start[1] - circle_center[1]

        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - circle_radius * circle_radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return False

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def _count_teammates_with_los_on_enemy(
        self,
        player: SimulatedPlayer,
        enemy: SimulatedPlayer,
        time_ms: int
    ) -> int:
        """Count how many of player's teammates also have LOS on the enemy.

        This enables EMERGENT man advantage: teams with more players
        naturally create more crossfire opportunities.

        Returns count of teammates (excluding the player) with LOS on enemy.
        """
        count = 0
        for teammate in self.players.values():
            if (teammate.is_alive and
                teammate.side == player.side and
                teammate.player_id != player.player_id):
                # Check if teammate has LOS on enemy
                if self._has_line_of_sight(
                    (teammate.x, teammate.y),
                    (enemy.x, enemy.y),
                    time_ms
                ):
                    count += 1
        return count

    def _get_information_advantage(
        self,
        player: SimulatedPlayer,
        enemy: SimulatedPlayer,
        time_ms: int
    ) -> float:
        """Calculate information advantage based on team's prior knowledge of enemy.

        EMERGENT MAN ADVANTAGE via SOUND SYSTEM:
        - More players = more ears = better sound coverage
        - If ANY teammate heard the enemy recently, the player is "warned"
        - Warned players react faster (less surprised)

        This naturally creates man advantage: 5-player teams have 5 sets of ears
        covering the map, while 3-player teams have coverage gaps.

        Returns: multiplier (>1.0 = informed, 1.0 = neutral, <1.0 = surprised)
        """
        # Check if any teammate heard this enemy recently (within 3 seconds)
        INFO_RECENCY_MS = 3000
        teammates_with_info = 0
        player_has_info = False

        for teammate in self.players.values():
            if teammate.side != player.side or not teammate.is_alive:
                continue

            # Check if this teammate heard the enemy
            if teammate.heard_enemy_sound and teammate.last_sound_location:
                sound_x, sound_y = teammate.last_sound_location
                # Is the sound location close to where the enemy actually is?
                dist_to_enemy = math.sqrt(
                    (sound_x - enemy.x) ** 2 + (sound_y - enemy.y) ** 2
                )
                # Sound location within 15% of map = relevant intel
                if dist_to_enemy < 0.15:
                    teammates_with_info += 1
                    if teammate.player_id == player.player_id:
                        player_has_info = True

        # Also check information system for recent sightings
        knowledge = self.info_manager.get_knowledge(player.player_id)
        if knowledge:
            enemy_info = knowledge.enemies.get(enemy.player_id)
            if enemy_info:
                info_age = time_ms - enemy_info.last_seen_ms
                if info_age < INFO_RECENCY_MS:
                    player_has_info = True

        # Calculate advantage
        if player_has_info:
            # Player personally knew where enemy was - significant advantage
            return 1.20  # 20% faster reaction
        elif teammates_with_info > 0:
            # Teammates warned via callout - some advantage
            return 1.0 + (0.08 * min(teammates_with_info, 2))  # Up to 16% advantage
        else:
            # No intel - neutral
            return 1.0

    def _get_crossfire_advantage(
        self,
        player: SimulatedPlayer,
        enemy: SimulatedPlayer,
        time_ms: int
    ) -> float:
        """Calculate crossfire advantage based on teammate support.

        EMERGENT MAN ADVANTAGE: Instead of hardcoded penalties, advantage
        comes naturally from having teammates who can support engagements.

        If 2+ teammates have LOS on the enemy, the player gains advantage.
        If enemy has crossfire support and player doesn't, player is disadvantaged.

        Returns: multiplier (>1.0 = advantage, <1.0 = disadvantage)
        """
        # Count teammates supporting this player
        player_support = self._count_teammates_with_los_on_enemy(player, enemy, time_ms)
        # Count enemies supporting the enemy
        enemy_support = self._count_teammates_with_los_on_enemy(enemy, player, time_ms)

        # Net crossfire advantage
        net_support = player_support - enemy_support

        # Convert to multiplier
        # +1 support: 15% advantage (faster reaction, better pre-aim)
        # +2 support: 25% advantage
        # -1 support: 15% disadvantage
        # -2 support: 30% disadvantage (facing crossfire)
        if net_support > 0:
            # Player has crossfire support - confidence boost
            return 1.0 + (0.15 * min(net_support, 2))
        elif net_support < 0:
            # Player facing crossfire - must split attention
            return 1.0 / (1.0 + (0.15 * min(abs(net_support), 2)))
        else:
            return 1.0

    def _get_engagement_distance_advantage(
        self,
        player: SimulatedPlayer,
        actual_distance: float
    ) -> float:
        """Calculate advantage based on player's preferred engagement distance.

        AUTONOMOUS PLAYER: Each player has a preferred engagement distance
        from VCT data. They perform better at their natural range.

        - Close-range players (1500 units / 0.15 norm): aggressive duelists
        - Mid-range players (1800 units / 0.18 norm): balanced flex
        - Long-range players (2200 units / 0.22 norm): OPers, passive players

        Returns: multiplier (>1.0 = at comfort range, <1.0 = outside comfort)
        """
        pref_dist = player.preferred_engagement_distance

        # Calculate how far from preferred distance (as ratio)
        if pref_dist > 0:
            distance_ratio = actual_distance / pref_dist
        else:
            return 1.0

        # At preferred distance: 1.0
        # 50% closer than preferred: 0.9 (uncomfortable, too close)
        # 50% farther than preferred: 0.9 (uncomfortable, too far)
        # 2x distance from preferred: 0.8 (very uncomfortable)
        if 0.8 <= distance_ratio <= 1.2:
            # Within comfort zone - slight bonus
            return 1.05
        elif 0.5 <= distance_ratio <= 2.0:
            # Outside comfort but manageable
            deviation = abs(1.0 - distance_ratio)
            return 1.0 - (deviation * 0.15)  # Up to 15% penalty
        else:
            # Way outside comfort zone
            return 0.85

    def _resolve_engagement(
        self,
        attacker: SimulatedPlayer,
        defender: SimulatedPlayer,
        distance: float,
        time_ms: int,
        phase: str = 'mid_round'
    ):
        """Resolve a combat engagement between two players using realistic combat model.

        Uses tick-by-tick simulation with:
        - Reaction time modeling
        - Crosshair placement quality
        - Counter-strafe mechanics
        - Time-to-kill calculations
        - Crossfire penalty for outnumbered players
        """
        # Convert normalized distance to meters
        distance_meters = meters_from_normalized(distance)

        # Engagement probability per tick (VALIDATED - phase-based)
        # Source: validated_parameters.py - calibrated to achieve ~7.5 kills/round
        engagement_params = EngagementParams()
        if phase == 'opening' and time_ms < 20000:
            engagement_prob = engagement_params.EARLY_ROUND_RATE  # 0.0008
        elif phase in ['post_plant', 'retake']:
            engagement_prob = engagement_params.POST_PLANT_RATE  # 0.0035
        elif time_ms > 60000:  # Late round
            engagement_prob = engagement_params.LATE_ROUND_RATE  # 0.0040
        else:  # Mid round
            engagement_prob = engagement_params.MID_ROUND_RATE  # 0.0025

        # SPIKE FLOW FIX: Higher engagement during site execute
        if self.site_execute_active and not self.spike_planted:
            engagement_prob = max(engagement_prob, 0.0080)

        # Boost engagement at close range
        if distance < 0.10:
            engagement_prob *= 2.0

        if random.random() >= engagement_prob:
            return  # No engagement this tick

        # EMERGENT ADVANTAGES via Information + Crossfire + Distance
        # 1. Information advantage: Did your team know enemy was coming?
        # 2. Crossfire advantage: Do teammates have LOS support?
        # 3. Distance advantage: Is this fight at player's preferred range?

        # Check if either player's team has recent intel on the enemy
        attacker_info_advantage = self._get_information_advantage(attacker, defender, time_ms)
        defender_info_advantage = self._get_information_advantage(defender, attacker, time_ms)

        # Crossfire advantage from teammate LOS support
        attacker_crossfire = self._get_crossfire_advantage(attacker, defender, time_ms)
        defender_crossfire = self._get_crossfire_advantage(defender, attacker, time_ms)

        # AUTONOMOUS: Engagement distance advantage - players perform better at their preferred range
        attacker_distance_adv = self._get_engagement_distance_advantage(attacker, distance)
        defender_distance_adv = self._get_engagement_distance_advantage(defender, distance)

        # If in a severe crossfire disadvantage, chance of immediate loss
        # This represents walking into multiple pre-aimed players
        if attacker_crossfire < 0.80:  # Facing 2+ enemies with LOS
            crossfire_loss_chance = (1.0 - attacker_crossfire) * 0.40  # Up to ~24% instant loss
            if random.random() < crossfire_loss_chance:
                killer = defender
                victim = attacker
                killer_weapon = defender.weapon or WeaponDatabase.WEAPONS['classic']
                is_headshot = random.random() < defender.headshot_rate
                victim.health = 0
                victim.is_alive = False
                victim.deaths += 1
                killer.kills += 1
                self.round_state.record_kill(
                    time_ms=time_ms,
                    killer_id=killer.player_id,
                    killer_team=killer.side,
                    victim_id=victim.player_id,
                    victim_team=victim.side,
                    position=(victim.x, victim.y),
                    weapon=killer_weapon.name,
                    is_headshot=is_headshot
                )
                return

        if defender_crossfire < 0.80:  # Defender facing crossfire
            crossfire_loss_chance = (1.0 - defender_crossfire) * 0.40
            if random.random() < crossfire_loss_chance:
                killer = attacker
                victim = defender
                killer_weapon = attacker.weapon or WeaponDatabase.WEAPONS['classic']
                is_headshot = random.random() < attacker.headshot_rate
                victim.health = 0
                victim.is_alive = False
                victim.deaths += 1
                killer.kills += 1
                self.round_state.record_kill(
                    time_ms=time_ms,
                    killer_id=killer.player_id,
                    killer_team=killer.side,
                    victim_id=victim.player_id,
                    victim_team=victim.side,
                    position=(victim.x, victim.y),
                    weapon=killer_weapon.name,
                    is_headshot=is_headshot
                )
                return

        # P0 FIX: Players planting/defusing are sitting ducks - auto-lose
        if attacker.is_in_action:
            killer = defender
            victim = attacker
            is_headshot = random.random() < defender.headshot_rate
            killer_weapon = defender.weapon or WeaponDatabase.WEAPONS['classic']
            # Calculate lethal damage
            health_dmg = victim.health
            shield_dmg = victim.shield
            new_shield = 0
        elif defender.is_in_action:
            killer = attacker
            victim = defender
            is_headshot = random.random() < attacker.headshot_rate
            killer_weapon = attacker.weapon or WeaponDatabase.WEAPONS['classic']
            health_dmg = victim.health
            shield_dmg = victim.shield
            new_shield = 0
        else:
            # Both can fight - use realistic combat model

            # Build combat profiles from player stats
            attacker_profile = self._build_combat_profile(attacker)
            defender_profile = self._build_combat_profile(defender)

            # EMERGENT INFORMATION ADVANTAGE: Apply based on prior knowledge
            # Teams with more players have more "ears" and better sound coverage
            # If your team knew enemy was coming, you react faster (less surprised)
            net_info_advantage = attacker_info_advantage - defender_info_advantage
            if net_info_advantage != 0:
                # Apply to reaction times - informed player reacts faster
                if attacker_info_advantage > 1.0:
                    attacker_profile = PlayerCombatProfile(
                        base_reaction_ms=int(attacker_profile.base_reaction_ms / attacker_info_advantage),
                        reaction_variance=attacker_profile.reaction_variance,
                        crosshair_placement=attacker_profile.crosshair_placement,
                        headshot_rate=attacker_profile.headshot_rate,
                        spray_control=attacker_profile.spray_control,
                        first_shot_discipline=attacker_profile.first_shot_discipline,
                        counter_strafe_skill=attacker_profile.counter_strafe_skill,
                        clutch_factor=attacker_profile.clutch_factor,
                        peek_aggression=attacker_profile.peek_aggression
                    )
                if defender_info_advantage > 1.0:
                    defender_profile = PlayerCombatProfile(
                        base_reaction_ms=int(defender_profile.base_reaction_ms / defender_info_advantage),
                        reaction_variance=defender_profile.reaction_variance,
                        crosshair_placement=defender_profile.crosshair_placement,
                        headshot_rate=defender_profile.headshot_rate,
                        spray_control=defender_profile.spray_control,
                        first_shot_discipline=defender_profile.first_shot_discipline,
                        counter_strafe_skill=defender_profile.counter_strafe_skill,
                        clutch_factor=defender_profile.clutch_factor,
                        peek_aggression=defender_profile.peek_aggression
                    )

            # EMERGENT CROSSFIRE: Apply advantage/disadvantage based on teammate support
            # This replaces hardcoded man-advantage penalties with situation-aware modifiers
            if attacker_crossfire < 1.0:
                # Attacker facing crossfire - slower reaction, worse aim
                attacker_profile = PlayerCombatProfile(
                    base_reaction_ms=int(attacker_profile.base_reaction_ms / attacker_crossfire),
                    reaction_variance=attacker_profile.reaction_variance,
                    crosshair_placement=attacker_profile.crosshair_placement * attacker_crossfire,
                    headshot_rate=max(0.05, attacker_profile.headshot_rate * attacker_crossfire),
                    spray_control=attacker_profile.spray_control,
                    first_shot_discipline=attacker_profile.first_shot_discipline,
                    counter_strafe_skill=attacker_profile.counter_strafe_skill,
                    clutch_factor=attacker_profile.clutch_factor,
                    peek_aggression=attacker_profile.peek_aggression
                )
            elif attacker_crossfire > 1.0:
                # Attacker has crossfire support - confidence boost
                attacker_profile = PlayerCombatProfile(
                    base_reaction_ms=int(attacker_profile.base_reaction_ms / attacker_crossfire),
                    reaction_variance=attacker_profile.reaction_variance,
                    crosshair_placement=min(1.0, attacker_profile.crosshair_placement * attacker_crossfire),
                    headshot_rate=min(0.40, attacker_profile.headshot_rate * attacker_crossfire),
                    spray_control=attacker_profile.spray_control,
                    first_shot_discipline=attacker_profile.first_shot_discipline,
                    counter_strafe_skill=attacker_profile.counter_strafe_skill,
                    clutch_factor=attacker_profile.clutch_factor,
                    peek_aggression=attacker_profile.peek_aggression
                )

            if defender_crossfire < 1.0:
                # Defender facing crossfire
                defender_profile = PlayerCombatProfile(
                    base_reaction_ms=int(defender_profile.base_reaction_ms / defender_crossfire),
                    reaction_variance=defender_profile.reaction_variance,
                    crosshair_placement=defender_profile.crosshair_placement * defender_crossfire,
                    headshot_rate=max(0.05, defender_profile.headshot_rate * defender_crossfire),
                    spray_control=defender_profile.spray_control,
                    first_shot_discipline=defender_profile.first_shot_discipline,
                    counter_strafe_skill=defender_profile.counter_strafe_skill,
                    clutch_factor=defender_profile.clutch_factor,
                    peek_aggression=defender_profile.peek_aggression
                )
            elif defender_crossfire > 1.0:
                # Defender has crossfire support
                defender_profile = PlayerCombatProfile(
                    base_reaction_ms=int(defender_profile.base_reaction_ms / defender_crossfire),
                    reaction_variance=defender_profile.reaction_variance,
                    crosshair_placement=min(1.0, defender_profile.crosshair_placement * defender_crossfire),
                    headshot_rate=min(0.40, defender_profile.headshot_rate * defender_crossfire),
                    spray_control=defender_profile.spray_control,
                    first_shot_discipline=defender_profile.first_shot_discipline,
                    counter_strafe_skill=defender_profile.counter_strafe_skill,
                    clutch_factor=defender_profile.clutch_factor,
                    peek_aggression=defender_profile.peek_aggression
                )

            # AUTONOMOUS: Engagement distance advantage - players at their preferred range perform better
            # Close-range players: better at 0.12-0.18 (in-your-face duels)
            # Long-range players: better at 0.20-0.25 (holding angles with OP)
            if attacker_distance_adv != 1.0:
                attacker_profile = PlayerCombatProfile(
                    base_reaction_ms=attacker_profile.base_reaction_ms,
                    reaction_variance=attacker_profile.reaction_variance,
                    crosshair_placement=attacker_profile.crosshair_placement * attacker_distance_adv,
                    headshot_rate=max(0.05, min(0.40, attacker_profile.headshot_rate * attacker_distance_adv)),
                    spray_control=attacker_profile.spray_control,
                    first_shot_discipline=attacker_profile.first_shot_discipline,
                    counter_strafe_skill=attacker_profile.counter_strafe_skill,
                    clutch_factor=attacker_profile.clutch_factor,
                    peek_aggression=attacker_profile.peek_aggression
                )
            if defender_distance_adv != 1.0:
                defender_profile = PlayerCombatProfile(
                    base_reaction_ms=defender_profile.base_reaction_ms,
                    reaction_variance=defender_profile.reaction_variance,
                    crosshair_placement=defender_profile.crosshair_placement * defender_distance_adv,
                    headshot_rate=max(0.05, min(0.40, defender_profile.headshot_rate * defender_distance_adv)),
                    spray_control=defender_profile.spray_control,
                    first_shot_discipline=defender_profile.first_shot_discipline,
                    counter_strafe_skill=defender_profile.counter_strafe_skill,
                    clutch_factor=defender_profile.clutch_factor,
                    peek_aggression=defender_profile.peek_aggression
                )

            # VCT Hold Angle Integration: Boost defender if holding a known angle
            # Uses actual angle direction from VCT data to calculate pre-aim quality
            if not defender.moved_this_tick:
                data_loader = get_data_loader()
                # Convert normalized position to VCT coordinates (approx 100m map = 10000 units)
                defender_x_vct = defender.x * 10000
                defender_y_vct = defender.y * 10000
                defender_zone = data_loader.get_zone_for_position(
                    self.map_name, defender_x_vct, defender_y_vct
                )
                if defender_zone:
                    hold_angle = data_loader.get_hold_angle(self.map_name, defender_zone)
                    if hold_angle and hold_angle.get('samples', 0) >= 5:
                        # Calculate angle from defender to attacker
                        dx = attacker.x - defender.x
                        dy = attacker.y - defender.y
                        angle_to_attacker = math.atan2(dy, dx)

                        # Compare to VCT hold angle (mean_angle is in radians)
                        vct_hold_angle = hold_angle.get('mean_angle', 0)
                        angle_std = hold_angle.get('std_angle', 1.0)

                        # Calculate angle difference (wrapped to -pi to pi)
                        angle_diff = abs(angle_to_attacker - vct_hold_angle)
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff

                        # Crosshair boost based on how close attacker is to the hold angle
                        # If within 1 std deviation: high boost
                        # If within 2 std: moderate boost
                        # Beyond 2 std: minimal boost (attacker from unexpected angle)
                        if angle_diff <= angle_std:
                            angle_quality = 1.0  # Perfect pre-aim
                        elif angle_diff <= 2 * angle_std:
                            angle_quality = 0.6  # Good pre-aim
                        else:
                            angle_quality = 0.2  # Attacker from unexpected angle

                        # Sample confidence (more samples = more reliable data)
                        sample_confidence = min(1.0, hold_angle['samples'] / 50)

                        # Final crosshair boost: 5-15% based on angle quality and confidence
                        crosshair_boost = 0.05 + (0.10 * angle_quality * sample_confidence)

                        defender_profile = PlayerCombatProfile(
                            base_reaction_ms=defender_profile.base_reaction_ms,
                            reaction_variance=defender_profile.reaction_variance,
                            crosshair_placement=min(0.98, defender_profile.crosshair_placement + crosshair_boost),
                            headshot_rate=defender_profile.headshot_rate,
                            spray_control=defender_profile.spray_control,
                            first_shot_discipline=defender_profile.first_shot_discipline,
                            counter_strafe_skill=defender_profile.counter_strafe_skill,
                            clutch_factor=defender_profile.clutch_factor,
                            peek_aggression=defender_profile.peek_aggression
                        )

            # Determine movement states
            attacker_movement = self._get_movement_state(attacker)
            defender_movement = self._get_movement_state(defender)

            # Determine engagement type
            attacker_peeking = attacker.moved_this_tick and not defender.moved_this_tick
            defender_peeking = defender.moved_this_tick and not attacker.moved_this_tick

            if attacker_peeking and not defender_peeking:
                engagement_type = EngagementType.PEEK_VS_HOLD
            elif defender_peeking and not attacker_peeking:
                engagement_type = EngagementType.HOLD_VS_PEEK
            elif attacker_peeking and defender_peeking:
                engagement_type = EngagementType.BOTH_PEEKING
            else:
                engagement_type = EngagementType.BOTH_HOLDING

            # Get weapon names
            attacker_weapon_obj = attacker.weapon or WeaponDatabase.WEAPONS['classic']
            defender_weapon_obj = defender.weapon or WeaponDatabase.WEAPONS['classic']
            attacker_weapon_name = attacker_weapon_obj.name.lower()
            defender_weapon_name = defender_weapon_obj.name.lower()

            # Simulate the combat
            result = combat_model.simulate_engagement(
                player_a_id=attacker.player_id,
                player_a_profile=attacker_profile,
                player_a_weapon=attacker_weapon_name,
                player_a_health=attacker.health,
                player_a_armor=attacker.shield,
                player_a_movement=attacker_movement,
                player_b_id=defender.player_id,
                player_b_profile=defender_profile,
                player_b_weapon=defender_weapon_name,
                player_b_health=defender.health,
                player_b_armor=defender.shield,
                player_b_movement=defender_movement,
                distance_meters=distance_meters,
                engagement_type=engagement_type,
                a_is_flashed=attacker.is_flashed,
                b_is_flashed=defender.is_flashed
            )

            # Determine winner/loser
            if result.winner_id == attacker.player_id:
                killer = attacker
                victim = defender
                killer_weapon = attacker_weapon_obj
            else:
                killer = defender
                victim = attacker
                killer_weapon = defender_weapon_obj

            is_headshot = result.headshot_kill

            # Apply lethal damage since the model determined the winner
            health_dmg = victim.health
            shield_dmg = victim.shield
            new_shield = 0

        # Apply damage
        victim.shield = new_shield
        victim.health = 0  # Lethal engagement

        # P0 FIX: Taking damage interrupts plant/defuse action
        if victim.is_planting:
            victim.is_planting = False
            victim.action_start_ms = 0
            self.events.append(SimulationEvent(
                timestamp_ms=time_ms,
                event_type='plant_cancelled',
                player_id=victim.player_id,
                position_x=victim.x,
                position_y=victim.y,
                details={'reason': 'damage', 'damage_taken': health_dmg + shield_dmg}
            ))
        if victim.is_defusing:
            # Save progress if past half-defuse checkpoint
            if (time_ms - victim.action_start_ms) >= self.HALF_DEFUSE_MS:
                victim.defuse_progress_ms = self.HALF_DEFUSE_MS
            victim.is_defusing = False
            victim.action_start_ms = 0
            self.events.append(SimulationEvent(
                timestamp_ms=time_ms,
                event_type='defuse_cancelled',
                player_id=victim.player_id,
                position_x=victim.x,
                position_y=victim.y,
                details={
                    'reason': 'damage',
                    'damage_taken': health_dmg + shield_dmg,
                    'progress_saved_ms': victim.defuse_progress_ms
                }
            ))

        if victim.health <= 0:
            victim.health = 0
            victim.is_alive = False
            victim.deaths += 1
            killer.kills += 1

            # Record kill in round state
            kill_event = self.round_state.record_kill(
                time_ms=time_ms,
                killer_id=killer.player_id,
                killer_team=killer.side,
                victim_id=victim.player_id,
                victim_team=victim.side,
                position=(victim.x, victim.y),
                weapon=killer_weapon.name,
                is_headshot=is_headshot
            )

            # Record simulation event
            self.events.append(SimulationEvent(
                timestamp_ms=time_ms,
                event_type='kill',
                player_id=victim.player_id,
                target_id=victim.player_id,
                position_x=victim.x,
                position_y=victim.y,
                details={
                    'killer_id': killer.player_id,
                    'headshot': is_headshot,
                    'weapon': killer_weapon.name,
                    'distance': distance,
                    'is_first_blood': kill_event.is_first_blood,
                    'is_trade': kill_event.is_trade,
                    'damage_dealt': health_dmg + shield_dmg,
                }
            ))

            # P3 FIX: Grant ultimate points on kill/death
            killer_ult_ready = killer.grant_ultimate_point("kill")
            victim_ult_ready = victim.grant_ultimate_point("death")

            # Record ultimate events if someone's ult became ready
            if killer_ult_ready:
                self.events.append(SimulationEvent(
                    timestamp_ms=time_ms,
                    event_type='ultimate_ready',
                    player_id=killer.player_id,
                    position_x=killer.x,
                    position_y=killer.y,
                    details={'agent': killer.agent, 'reason': 'kill'}
                ))
            if victim_ult_ready:
                self.events.append(SimulationEvent(
                    timestamp_ms=time_ms,
                    event_type='ultimate_ready',
                    player_id=victim.player_id,
                    position_x=victim.x,
                    position_y=victim.y,
                    details={'agent': victim.agent, 'reason': 'death'}
                ))

            # P2 FIX: Set up pending trade opportunity
            self.pending_trade = {
                'kill_time_ms': time_ms,
                'kill_position': (victim.x, victim.y),
                'killer_id': killer.player_id,
                'killer_position': (killer.x, killer.y),
                'victim_team': victim.side,  # Team that needs to trade
            }

            # Transfer spike if victim had it
            if victim.has_spike:
                victim.has_spike = False
                # Find nearest alive attacker to give spike
                if victim.side == 'attack':
                    alive_attackers = [
                        p for p in self.players.values()
                        if p.side == 'attack' and p.is_alive
                    ]
                    if alive_attackers:
                        # Give to nearest attacker
                        nearest = min(
                            alive_attackers,
                            key=lambda p: math.sqrt(
                                (p.x - victim.x) ** 2 + (p.y - victim.y) ** 2
                            )
                        )
                        nearest.has_spike = True

    def _check_site_execute_decision(self, time_ms: int):
        """SPIKE FLOW FIX: Decide when attackers should execute on a site.

        In real VALORANT, teams don't just wander into sites - they make a
        strategic decision to execute, use utility, and take site control.

        This method models:
        1. Execute timing decision (based on time pressure, man advantage, info)
        2. Site selection (based on current positions, defender locations)
        3. Site control assessment (are attackers in position to plant?)

        Pro match statistics (source: THESPIKE.GG, VLR.gg):
        - Average plant time: 56 seconds
        - Fast execute: 45 seconds
        - Slow default: 70 seconds
        - ~35% of rounds end in spike outcomes (plant→defuse or plant→explode)
        """
        # Skip if spike already planted or execute already in progress
        if self.spike_planted:
            return

        # Count alive players
        alive_attack = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        alive_defense = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        # Can't execute without players
        if alive_attack == 0:
            return

        # Get pacing parameters
        pacing = RoundPacingParams()

        # DECISION 1: Should we START an execute?
        if not self.site_execute_active:
            # Base probability to start execute
            execute_prob = 0.0

            # Time-based execute probability
            # Target: ~40-50% of rounds should result in plant
            # Pro stats: avg plant time 56s, 35% rounds end in spike outcomes
            # Probabilities tuned to achieve ~40-50% execute rate with ~56s avg plant time
            if time_ms < 30000:
                # Opening phase - very rare fast exec
                execute_prob = 0.0003
            elif time_ms < 45000:
                # Early-mid - starting to consider execute
                execute_prob = 0.0015
            elif time_ms < 60000:
                # Around average plant time - most executes start here
                execute_prob = 0.0040
            elif time_ms < 75000:
                # Late - should execute soon
                execute_prob = 0.008
            else:
                # Very late round - urgent
                execute_prob = 0.015

            # Man advantage modifier
            if alive_attack > alive_defense:
                execute_prob *= 1.5  # More likely with advantage
            elif alive_attack < alive_defense:
                execute_prob *= 0.7  # More cautious when down

            # Decide to execute
            if random.random() < execute_prob:
                self.site_execute_active = True
                self.execute_start_time = time_ms

                # Select target site based on attacker positions
                sites = self._get_map_sites()
                spike_carrier = next(
                    (p for p in self.players.values()
                     if p.side == 'attack' and p.is_alive and p.has_spike),
                    None
                )

                if spike_carrier and sites:
                    # Find closest site to spike carrier
                    closest_site = min(
                        sites.items(),
                        key=lambda s: math.sqrt(
                            (spike_carrier.x - s[1]['center'][0])**2 +
                            (spike_carrier.y - s[1]['center'][1])**2
                        )
                    )
                    self.target_site = closest_site[0]
                else:
                    self.target_site = 'A'  # Default

                # Record execute event
                self.events.append(SimulationEvent(
                    timestamp_ms=time_ms,
                    event_type='execute_start',
                    player_id=spike_carrier.player_id if spike_carrier else 'team',
                    position_x=spike_carrier.x if spike_carrier else 0.5,
                    position_y=spike_carrier.y if spike_carrier else 0.5,
                    details={
                        'target_site': self.target_site,
                        'attackers_alive': alive_attack,
                        'defenders_alive': alive_defense
                    }
                ))
                return

        # DECISION 2: Check site control during active execute
        if self.site_execute_active and not self.site_control_achieved:
            sites = self._get_map_sites()
            if self.target_site and self.target_site in sites:
                site_data = sites[self.target_site]
                site_center = site_data['center']
                site_radius = site_data.get('radius', 0.08)

                # Count attackers near site
                attackers_on_site = sum(
                    1 for p in self.players.values()
                    if p.side == 'attack' and p.is_alive and
                    math.sqrt((p.x - site_center[0])**2 + (p.y - site_center[1])**2) < site_radius * 2
                )

                # Count defenders near site
                defenders_on_site = sum(
                    1 for p in self.players.values()
                    if p.side == 'defense' and p.is_alive and
                    math.sqrt((p.x - site_center[0])**2 + (p.y - site_center[1])**2) < site_radius * 2
                )

                # Site control achieved when attackers have cleared the site
                # This should be harder to achieve - requires winning fights
                time_since_execute = time_ms - (self.execute_start_time or time_ms)

                # More stringent conditions for site control
                # Combat should happen during execute, so not every execute succeeds
                control_achieved = (
                    # Multiple attackers on site AND no defenders
                    (attackers_on_site >= 2 and defenders_on_site == 0) or
                    # OR more attackers than defenders AND some time has passed (utility cleared)
                    (attackers_on_site > defenders_on_site + 1 and time_since_execute > 5000) or
                    # OR long execute with presence (utility should have cleared by now)
                    (time_since_execute > 15000 and attackers_on_site >= 1 and defenders_on_site == 0)
                )

                if control_achieved:
                    self.site_control_achieved = True
                    self.events.append(SimulationEvent(
                        timestamp_ms=time_ms,
                        event_type='site_control',
                        player_id='team',
                        position_x=site_center[0],
                        position_y=site_center[1],
                        details={
                            'site': self.target_site,
                            'attackers_on_site': attackers_on_site,
                            'defenders_on_site': defenders_on_site
                        }
                    ))

    def _check_spike_plant(self, time_ms: int):
        """Check spike plant progress - P0 FIX: 4-second duration with immobility.

        Plant mechanics (real VALORANT):
        - Takes 4 seconds to complete
        - Player CANNOT move, shoot, or look while planting
        - Taking damage cancels the plant
        - Player must be on a bombsite
        """
        # P1 FIX: Use map-specific site positions (supports 2-site and 3-site maps)
        sites = self._get_map_sites()

        for player in self.players.values():
            if player.side != 'attack' or not player.is_alive or not player.has_spike:
                continue

            # CASE 1: Player is currently planting - check for completion
            if player.is_planting:
                plant_progress = time_ms - player.action_start_ms
                if plant_progress >= self.PLANT_DURATION_MS:
                    # Plant complete!
                    player.is_planting = False
                    player.action_start_ms = 0
                    self.spike_planted = True
                    self.spike_plant_time = time_ms

                    # Determine which site based on player position
                    for site_name, site_data in sites.items():
                        site_x, site_y = site_data['center']
                        site_radius = site_data.get('radius', 0.1)
                        distance = math.sqrt((player.x - site_x) ** 2 + (player.y - site_y) ** 2)
                        if distance < site_radius + 0.05:  # Use map-specific radius
                            self.spike_site = site_name
                            break
                    else:
                        self.spike_site = 'A'  # Default

                    # Update round state
                    self.round_state.plant_spike(time_ms, self.spike_site)

                    self.events.append(SimulationEvent(
                        timestamp_ms=time_ms,
                        event_type='spike_plant',
                        player_id=player.player_id,
                        position_x=player.x,
                        position_y=player.y,
                        details={
                            'site': self.spike_site,
                            'plant_duration_ms': self.PLANT_DURATION_MS
                        }
                    ))

                    # P3 FIX: Grant ultimate point for successful plant
                    player.grant_ultimate_point("plant")

                    return  # Only one plant per tick
                continue  # Still planting, don't start new plant

            # CASE 2: Player can START planting
            # Check if player is on a site (P1 FIX: use map-specific site radius)
            for site_name, site_data in sites.items():
                site_x, site_y = site_data['center']
                site_radius = site_data.get('radius', 0.1)
                distance = math.sqrt((player.x - site_x) ** 2 + (player.y - site_y) ** 2)
                if distance < site_radius:  # Within site's plant radius
                    # SPIKE FLOW FIX: Plant probability based on site execute state
                    # Pro statistics: ~35% of rounds end in spike outcomes
                    # Average plant time: 56 seconds

                    # Base probability (no execute active - wandered onto site)
                    pacing = RoundPacingParams()
                    plant_prob = 0.005  # Very low - not intentionally planting

                    # CASE 2a: Site control achieved - PLANT IMMEDIATELY
                    if self.site_control_achieved and self.target_site == site_name:
                        # Site is clear, plant ASAP
                        plant_prob = 0.40  # 40% per tick = plant within ~3 ticks

                    # CASE 2b: Executing on this site but no control yet
                    elif self.site_execute_active and self.target_site == site_name:
                        # Executing but site not fully clear - risky plant
                        plant_prob = 0.08  # May try to plant while teammates hold

                        # Increase if low on time
                        if time_ms > 70000:
                            plant_prob = 0.20  # Urgent - plant even if risky

                    # CASE 2c: Not executing (default/fallback behavior)
                    else:
                        # Time-based probability using validated parameters
                        if time_ms < 30000:
                            plant_prob = pacing.PLANT_PROB_EARLY  # 0.001
                        elif time_ms < 50000:
                            plant_prob = pacing.PLANT_PROB_MID  # 0.003
                        elif time_ms < 70000:
                            plant_prob = pacing.PLANT_PROB_LATE  # 0.010
                        elif time_ms < 90000:
                            plant_prob = pacing.PLANT_PROB_URGENT  # 0.025
                        else:
                            plant_prob = pacing.PLANT_PROB_CRITICAL  # 0.050

                    # Behavior modifier (aggressive players plant faster)
                    behavior = player.behavior_modifiers
                    if behavior and behavior.aggression > 0:
                        plant_prob *= 1.2

                    # Check nearby defenders - don't plant with defenders in face
                    nearby_defenders = sum(
                        1 for p in self.players.values()
                        if p.side == 'defense' and p.is_alive and
                        math.sqrt((p.x - player.x)**2 + (p.y - player.y)**2) < 0.10
                    )
                    if nearby_defenders > 0 and not self.site_control_achieved:
                        plant_prob *= 0.2  # Much less likely with defender nearby

                    # Minimum time before first plant (5 seconds for fast exec)
                    if time_ms > 5000 and random.random() < plant_prob:
                        # START planting (doesn't complete instantly)
                        player.is_planting = True
                        player.action_start_ms = time_ms

                        self.events.append(SimulationEvent(
                            timestamp_ms=time_ms,
                            event_type='plant_start',
                            player_id=player.player_id,
                            position_x=player.x,
                            position_y=player.y,
                            details={
                                'site': site_name,
                                'duration_required_ms': self.PLANT_DURATION_MS,
                                'site_control': self.site_control_achieved,
                                'execute_active': self.site_execute_active
                            }
                        ))
                        return  # Only one plant attempt per tick

    def _check_spike_defuse(self, time_ms: int):
        """Check spike defuse progress - P0 FIX: 7-second duration with half-defuse checkpoint.

        Defuse mechanics (real VALORANT):
        - Takes 7 seconds to complete
        - Player CANNOT move, shoot, or look while defusing
        - Taking damage cancels the defuse
        - Half-defuse checkpoint at 3.5 seconds (progress saved)
        - Player must be near the planted spike
        """
        if not self.spike_planted or not self.spike_site:
            return

        # P1 FIX: Use map-specific site positions (supports 3-site maps)
        sites = self._get_map_sites()
        site_data = sites.get(self.spike_site, {'center': (0.5, 0.3), 'radius': 0.08})
        spike_pos = site_data['center']

        for player in self.players.values():
            if player.side != 'defense' or not player.is_alive:
                continue

            # CASE 1: Player is currently defusing - check for completion
            if player.is_defusing:
                # Calculate total progress including saved progress
                current_progress = (time_ms - player.action_start_ms) + player.defuse_progress_ms
                if current_progress >= self.DEFUSE_DURATION_MS:
                    # Defuse complete!
                    player.is_defusing = False
                    player.action_start_ms = 0
                    player.defuse_progress_ms = 0

                    # Update round state
                    self.round_state.defuse_spike(time_ms)

                    self.events.append(SimulationEvent(
                        timestamp_ms=time_ms,
                        event_type='spike_defuse',
                        player_id=player.player_id,
                        position_x=player.x,
                        position_y=player.y,
                        details={
                            'site': self.spike_site,
                            'defuse_duration_ms': self.DEFUSE_DURATION_MS
                        }
                    ))
                    return  # Defuse complete
                continue  # Still defusing

            # CASE 2: Player can START defusing
            distance_to_spike = math.sqrt(
                (player.x - spike_pos[0]) ** 2 + (player.y - spike_pos[1]) ** 2
            )

            if distance_to_spike < 0.08:  # Within 8% of map to spike
                # Check time remaining on spike
                time_since_plant = time_ms - (self.spike_plant_time or time_ms)
                spike_time_remaining = self.SPIKE_TIME_MS - time_since_plant

                # Calculate time needed to defuse (accounting for saved progress)
                time_needed = self.DEFUSE_DURATION_MS - player.defuse_progress_ms

                # SPIKE FLOW FIX: Use validated defuse probabilities
                # Pro statistics: 31% defuse success rate
                pacing = RoundPacingParams()

                # Count alive attackers (affects defuse decision)
                alive_attackers = sum(
                    1 for p in self.players.values()
                    if p.side == 'attack' and p.is_alive
                )

                # Count nearby attackers (very dangerous to defuse)
                nearby_attackers = sum(
                    1 for p in self.players.values()
                    if p.side == 'attack' and p.is_alive and
                    math.sqrt((p.x - spike_pos[0])**2 + (p.y - spike_pos[1])**2) < 0.15
                )

                # Base defuse probability based on time remaining
                if spike_time_remaining > 30000:  # > 30 seconds
                    defuse_prob = pacing.DEFUSE_PROB_SAFE  # 0.002
                elif spike_time_remaining > 15000:  # 15-30 seconds
                    defuse_prob = pacing.DEFUSE_PROB_MODERATE  # 0.005
                elif spike_time_remaining > 7000:  # 7-15 seconds
                    defuse_prob = pacing.DEFUSE_PROB_URGENT  # 0.015
                else:  # < 7 seconds (must defuse now!)
                    defuse_prob = pacing.DEFUSE_PROB_CRITICAL  # 0.030

                # MAJOR MODIFIER: Adjust based on attacker presence
                # Note: Even if attackers are dead, defender must reach spike first
                # Pro defuse success rate: 31% - most post-plants don't get defused
                if alive_attackers == 0:
                    defuse_prob = 0.15  # 15% per tick - still need to reach spike

                # If attackers alive but not nearby, more likely to try
                elif alive_attackers > 0 and nearby_attackers == 0:
                    defuse_prob *= 1.5  # 50% boost - attackers far away

                # If attackers nearby, very risky to defuse
                elif nearby_attackers > 0:
                    defuse_prob *= 0.2  # Much less likely with attackers near

                # Don't start if not enough time (even with half-defuse)
                if spike_time_remaining < time_needed:
                    if alive_attackers == 0:
                        # Still try - might get lucky with half-defuse
                        defuse_prob *= 0.8
                    else:
                        # Can't complete defuse in time, might fake
                        defuse_prob *= 0.2

                # Behavior modifier - less aggressive players more likely to tap
                behavior = player.behavior_modifiers
                if behavior:
                    if behavior.aggression < 0.3:
                        defuse_prob *= 1.3  # Passive players more likely to defuse

                if random.random() < defuse_prob:
                    # START defusing
                    player.is_defusing = True
                    player.action_start_ms = time_ms

                    self.events.append(SimulationEvent(
                        timestamp_ms=time_ms,
                        event_type='defuse_start',
                        player_id=player.player_id,
                        position_x=player.x,
                        position_y=player.y,
                        details={
                            'site': self.spike_site,
                            'progress_ms': player.defuse_progress_ms,
                            'time_needed_ms': time_needed,
                            'spike_time_remaining_ms': spike_time_remaining
                        }
                    ))
                    return  # Only one defuse attempt per tick

    # =========================================================================
    # AI SYSTEM: Information and Decision Methods
    # =========================================================================

    def _update_information_state(self, time_ms: int):
        """AI SYSTEM: Update what each player knows about the game state.

        This replaces omniscient knowledge with realistic fog of war.
        Players only know what they've seen or heard.
        """
        if not self.ai_behavior:
            return

        # Get active smoke positions for vision blocking
        # get_active_smoke_positions returns [(position, radius), ...], extract just positions
        active_smoke_data = self.ability_system.get_active_smoke_positions(time_ms)
        active_smokes = [pos for pos, radius in active_smoke_data]

        # Prepare target list for vision checks
        all_players = [
            {
                'id': p.player_id,
                'x': p.x,
                'y': p.y,
                'team': p.side,
                'is_alive': p.is_alive,
                'is_running': p.is_running,
                'facing': math.atan2(p.y - p.prev_y, p.x - p.prev_x) if p.moved_this_tick else 0
            }
            for p in self.players.values()
            if p.is_alive
        ]

        # Update vision for each player
        for player in self.players.values():
            if not player.is_alive:
                continue

            # Calculate facing direction based on movement or last direction
            if player.moved_this_tick:
                facing = math.atan2(player.y - player.prev_y, player.x - player.prev_x)
            else:
                # Default to facing forward based on team
                facing = 0 if player.side == 'attack' else math.pi

            # Update vision-based knowledge
            self.info_manager.update_vision(
                observer_id=player.player_id,
                observer_pos=(player.x, player.y),
                observer_facing=facing,
                targets=all_players,
                time_ms=time_ms,
                smoke_positions=active_smokes
            )

            # Propagate running footstep sounds
            if player.is_running and player.moved_this_tick:
                self.info_manager.propagate_sound(
                    source_pos=(player.x, player.y),
                    sound_type=InfoSource.SOUND_FOOTSTEP,
                    source_team=player.side,
                    all_players=all_players,
                    time_ms=time_ms,
                    source_id=player.player_id
                )

        # Update info manager tick (decay confidence, process callouts)
        self.info_manager.update_tick(time_ms)

    def _build_neural_state(self, player: SimulatedPlayer, time_ms: int, phase: str,
                            knowledge: PlayerKnowledge, sites: Dict) -> GameStateFeatures:
        """Build GameStateFeatures for neural AI from player state."""
        # Get teammate info
        teammates_info = []
        for p in self.players.values():
            if p.side == player.side and p.is_alive and p.player_id != player.player_id:
                dist = math.sqrt((player.x - p.x)**2 + (player.y - p.y)**2)
                angle = math.atan2(p.y - player.y, p.x - player.x)
                teammates_info.append({
                    'distance': dist,
                    'angle': angle,
                    'health': p.health,
                    'has_util': True,  # Simplified
                    'is_alive': p.is_alive
                })

        # Get nearby enemy info from knowledge
        enemies_info = []
        for enemy_id, enemy_info in knowledge.enemies.items():
            if enemy_info and enemy_info.last_known_x is not None:
                dist = math.sqrt((player.x - enemy_info.last_known_x)**2 +
                               (player.y - enemy_info.last_known_y)**2)
                angle = math.atan2(enemy_info.last_known_y - player.y,
                                  enemy_info.last_known_x - player.x)
                confidence_map = {
                    InfoConfidence.EXACT: 1.0, InfoConfidence.HIGH: 0.8,
                    InfoConfidence.MEDIUM: 0.5, InfoConfidence.LOW: 0.3, InfoConfidence.STALE: 0.1
                }
                enemies_info.append({
                    'distance': dist,
                    'angle': angle,
                    'confidence': confidence_map.get(enemy_info.confidence, 0.5),
                    'visible': enemy_info.confidence == InfoConfidence.EXACT,
                    'time_since_seen': time_ms - enemy_info.last_seen_ms
                })

        # Calculate distances to sites
        dist_a = 1.0
        dist_b = 1.0
        dist_c = 1.0
        if 'A' in sites:
            dist_a = math.sqrt((player.x - sites['A']['center'][0])**2 +
                              (player.y - sites['A']['center'][1])**2)
        if 'B' in sites:
            dist_b = math.sqrt((player.x - sites['B']['center'][0])**2 +
                              (player.y - sites['B']['center'][1])**2)
        if 'C' in sites:
            dist_c = math.sqrt((player.x - sites['C']['center'][0])**2 +
                              (player.y - sites['C']['center'][1])**2)

        # Agent utilities
        agent_lower = player.agent.lower() if player.agent else ""
        has_flash = agent_lower in ['phoenix', 'reyna', 'yoru', 'kayo', 'skye', 'breach', 'gekko']
        has_smoke = agent_lower in ['omen', 'brimstone', 'astra', 'viper', 'harbor', 'clove']
        has_molly = agent_lower in ['brimstone', 'viper', 'killjoy', 'kayo', 'phoenix', 'gekko']

        return GameStateFeatures(
            position=(player.x, player.y),
            velocity=(player.x - player.prev_x, player.y - player.prev_y),
            health=player.health,
            armor=player.shield,
            time_ms=time_ms,
            round_time_ms=self.ROUND_TIME_MS,
            role=player.role.value if player.role else 'support',
            team=player.side,
            has_spike=player.has_spike,
            spike_planted=self.spike_planted,
            is_planting=player.is_planting,
            is_defusing=player.is_defusing,
            enemies_seen=knowledge.get_known_enemy_count(),
            enemies_confirmed_dead=len(knowledge.confirmed_dead),
            uncertainty=1.0 - knowledge.get_known_enemy_count() / 5.0,
            threat_level=knowledge.get_site_threat_level(player.x, player.y, 0.15, time_ms),
            nearby_enemies=enemies_info[:3],
            nearby_teammates=teammates_info[:2],
            distance_to_site_a=dist_a,
            distance_to_site_b=dist_b,
            distance_to_site_c=dist_c,
            has_flash=has_flash,
            has_smoke=has_smoke,
            has_molly=has_molly,
            is_running=player.is_running,
            is_moving=player.moved_this_tick,
            phase=phase,
            time_since_last_action_ms=0
        )

    def _build_utility_context(self, player: SimulatedPlayer, time_ms: int,
                               knowledge: PlayerKnowledge, sites: Dict,
                               target_site_pos: Optional[Tuple[float, float]]) -> DecisionContext:
        """Build DecisionContext for utility AI from player state."""
        # Get teammate positions
        teammates = {
            p.player_id: (p.x, p.y)
            for p in self.players.values()
            if p.side == player.side and p.is_alive and p.player_id != player.player_id
        }

        # Calculate spike time remaining
        spike_time_remaining = None
        if self.spike_planted and self.spike_plant_time:
            spike_time_remaining = self.SPIKE_TIME_MS - (time_ms - self.spike_plant_time)

        # Derive utility availability from agent type
        agent_lower = player.agent.lower() if player.agent else ""
        has_flash = agent_lower in ['phoenix', 'reyna', 'yoru', 'kayo', 'skye', 'breach', 'gekko']
        has_smoke = agent_lower in ['omen', 'brimstone', 'astra', 'viper', 'harbor', 'clove']
        has_molly = agent_lower in ['brimstone', 'viper', 'killjoy', 'kayo', 'phoenix', 'gekko']

        return DecisionContext(
            player_id=player.player_id,
            team=player.side,
            role=player.role.value if player.role else 'support',
            position=(player.x, player.y),
            time_ms=time_ms,
            round_time_ms=self.ROUND_TIME_MS,
            has_spike=player.has_spike,
            is_alive=player.is_alive,
            health=player.health,
            armor=player.shield,
            has_flash=has_flash,
            has_smoke=has_smoke,
            has_molly=has_molly,
            has_util=has_flash or has_smoke or has_molly,
            knowledge=knowledge,
            teammates_alive=len(teammates) + 1,
            teammates_positions=teammates,
            spike_planted=self.spike_planted,
            spike_site=self.spike_site,
            spike_time_remaining_ms=spike_time_remaining,
            target_site=self.target_site,
            site_position=target_site_pos,
            site_radius=sites.get(self.target_site, {}).get('radius', 0.08) if self.target_site else 0.08
        )

    def _update_ai_decisions(self, time_ms: int, phase: str):
        """AI SYSTEM: Update AI decisions for all players.

        This replaces hardcoded probabilities with emergent decision making
        based on what each player knows.

        Supports two modes:
        - Neural AI (use_neural_ai=True): Uses trained neural network
        - Utility AI (use_neural_ai=False): Uses utility-based decisions
        """
        if not self.ai_behavior:
            return

        sites = self._get_map_sites()
        target_site_pos = None
        if self.target_site and self.target_site in sites:
            target_site_pos = sites[self.target_site]['center']

        # Build decision contexts for all players
        attack_contexts = []
        for player in self.players.values():
            if not player.is_alive:
                continue

            knowledge = self.info_manager.get_knowledge(player.player_id)
            if not knowledge:
                continue

            # NEURAL AI MODE: Use learned neural network
            if self.use_neural_ai:
                neural_state = self._build_neural_state(player, time_ms, phase, knowledge, sites)
                action, probs = self.neural_ai.make_decision(player.player_id, neural_state)

                # Convert neural action to legacy decision format for movement system
                action_name = self.neural_ai.convert_to_legacy_decision(action)

                # Store decision in ai_behavior for movement integration
                from .ai_decision_system import DecisionResult
                decision = DecisionResult(
                    decision=Decision[action_name.upper()],
                    confidence=max(probs),
                    utility=max(probs),
                    target_position=target_site_pos if action_name in ['advance', 'execute', 'plant'] else (player.x, player.y),
                    reasoning=f"Neural AI: {action_name} ({max(probs):.2f})"
                )
                self.ai_behavior.current_decisions[player.player_id] = decision

                if player.side == 'attack':
                    # Still need utility context for team execute decisions
                    attack_contexts.append(self._build_utility_context(player, time_ms, knowledge, sites, target_site_pos))

            else:
                # UTILITY AI MODE: Use rule-based utility system
                ctx = self._build_utility_context(player, time_ms, knowledge, sites, target_site_pos)

                # Make decision
                decision = self.ai_behavior.update_player_decision(ctx)

                # Collect attack contexts for team execute decision
                if player.side == 'attack':
                    attack_contexts.append(ctx)

        # Team execute decision (replaces hardcoded execute_prob)
        if attack_contexts and not self.spike_planted and not self.site_execute_active:
            should_exec, target, confidence = self.ai_system.should_team_execute(
                attack_contexts, self.info_manager
            )

            if should_exec:
                self.site_execute_active = True
                self.execute_start_time = time_ms
                self.target_site = target

                # Record execute event
                spike_carrier = next(
                    (p for p in self.players.values()
                     if p.side == 'attack' and p.is_alive and p.has_spike),
                    None
                )
                self.events.append(SimulationEvent(
                    timestamp_ms=time_ms,
                    event_type='execute_start',
                    player_id=spike_carrier.player_id if spike_carrier else 'team',
                    position_x=spike_carrier.x if spike_carrier else 0.5,
                    position_y=spike_carrier.y if spike_carrier else 0.5,
                    details={
                        'target_site': self.target_site,
                        'confidence': confidence,
                        'ai_decision': True
                    }
                ))

    def _should_plant_ai(self, player: 'SimulatedPlayer', time_ms: int) -> bool:
        """AI SYSTEM: Check if player should plant based on AI decision."""
        if not self.ai_behavior:
            return False
        return self.ai_behavior.should_plant(player.player_id)

    def _should_defuse_ai(self, player: 'SimulatedPlayer', time_ms: int) -> bool:
        """AI SYSTEM: Check if player should defuse based on AI decision."""
        if not self.ai_behavior:
            return False
        return self.ai_behavior.should_defuse(player.player_id)

    def _check_round_end(self, time_ms: int) -> bool:
        """Check if the round should end."""
        alive_attack = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        alive_defense = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        # Team eliminated
        if alive_attack == 0 or alive_defense == 0:
            return True

        # Time expired
        if not self.spike_planted and time_ms >= self.ROUND_TIME_MS:
            return True

        # Spike exploded
        if self.spike_planted and self.spike_plant_time:
            if time_ms >= self.spike_plant_time + self.SPIKE_TIME_MS:
                return True

        # P0 FIX: Spike defused
        if self.round_state.spike_defused:
            return True

        return False

    async def create_snapshot(self, session: SimulationSession) -> str:
        """Create a snapshot of current state for what-if scenarios."""
        snapshot_id = str(uuid4())

        snapshot = SimulationSnapshot(
            id=snapshot_id,
            time_ms=session.current_time_ms,
            phase=self._get_current_phase(session.current_time_ms),
            players=[
                {
                    'player_id': p.player_id,
                    'team_id': p.team_id,
                    'side': p.side,
                    'x': p.x,
                    'y': p.y,
                    'is_alive': p.is_alive,
                    'health': p.health,
                    'shield': p.shield,
                    'has_spike': p.has_spike,
                    'weapon_name': p.weapon_name,
                    'armor_name': p.armor_name,
                    'loadout_value': p.loadout_value,
                }
                for p in self.players.values()
            ],
            events=[e.model_dump() for e in self.events],
            spike_planted=self.spike_planted,
            spike_site=self.spike_site,
        )

        self.snapshots.append(snapshot)

        # Update session
        existing_snapshots = session.snapshots or []
        existing_snapshots.append(snapshot.__dict__)
        session.snapshots = existing_snapshots

        return snapshot_id

    async def run_what_if(
        self,
        session: SimulationSession,
        scenario: WhatIfScenario
    ) -> SimulationState:
        """Run a what-if scenario from a snapshot."""
        # Find the snapshot
        snapshot = None
        for s in self.snapshots:
            if s.time_ms <= scenario.snapshot_time_ms:
                snapshot = s
            else:
                break

        if not snapshot:
            # Use current state
            snapshot = SimulationSnapshot(
                id=str(uuid4()),
                time_ms=session.current_time_ms,
                phase=self._get_current_phase(session.current_time_ms),
                players=[
                    {
                        'player_id': p.player_id,
                        'team_id': p.team_id,
                        'side': p.side,
                        'x': p.x,
                        'y': p.y,
                        'is_alive': p.is_alive,
                        'health': p.health,
                        'shield': p.shield,
                        'has_spike': p.has_spike,
                        'weapon_name': p.weapon_name,
                        'armor_name': p.armor_name,
                    }
                    for p in self.players.values()
                ],
                events=[e.model_dump() for e in self.events],
                spike_planted=self.spike_planted,
                spike_site=self.spike_site,
            )

        # Restore state from snapshot
        self.players = {}
        for p_data in snapshot.players:
            weapon = WeaponDatabase.get_weapon(p_data.get('weapon_name', 'Classic'))
            armor = WeaponDatabase.get_armor(p_data.get('armor_name', 'none'))

            self.players[p_data['player_id']] = SimulatedPlayer(
                player_id=p_data['player_id'],
                team_id=p_data['team_id'],
                side=p_data['side'],
                x=p_data['x'],
                y=p_data['y'],
                is_alive=p_data['is_alive'],
                health=p_data['health'],
                shield=p_data.get('shield', 0),
                has_spike=p_data['has_spike'],
                weapon=weapon,
                armor=armor,
            )

        self.events = [SimulationEvent(**e) for e in snapshot.events]
        self.spike_planted = snapshot.spike_planted
        self.spike_site = snapshot.spike_site

        # Apply modifications
        for player_id, mods in scenario.modifications.items():
            if player_id in self.players:
                player = self.players[player_id]
                if 'x' in mods:
                    player.x = mods['x']
                if 'y' in mods:
                    player.y = mods['y']
                if 'is_alive' in mods:
                    player.is_alive = mods['is_alive']
                if 'health' in mods:
                    player.health = mods['health']
                if 'shield' in mods:
                    player.shield = mods['shield']

        # Run simulation forward
        return await self.advance(session, ticks=100)

    async def analyze(self, session: SimulationSession) -> Dict[str, Any]:
        """Analyze simulation and provide improvement suggestions."""
        alive_attack = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        alive_defense = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        # Determine winner
        if self.spike_planted and alive_attack > 0:
            winner = 'attack'
        elif alive_attack == 0:
            winner = 'defense'
        elif alive_defense == 0:
            winner = 'attack'
        else:
            winner = 'defense'  # Time expired without plant

        # Collect kills
        kills = [e for e in self.events if e.event_type == 'kill']

        # Calculate win probability at current state
        win_prob = WinProbabilityCalculator.calculate_win_probability(
            self.round_state, session.current_time_ms
        )

        # Generate suggestions based on simulation results
        suggestions = []

        # Attack suggestions
        if winner == 'defense':
            if not self.spike_planted:
                suggestions.append("Consider faster site takes - spike was never planted")
            if len([k for k in kills if k.details.get('killer_id', '').startswith(session.defense_team_id)]) > 2:
                suggestions.append("Multiple early deaths suggest entry routes may be predictable")

            # Check economy impact
            attack_loadout = sum(p.loadout_value for p in self.players.values() if p.side == 'attack')
            defense_loadout = sum(p.loadout_value for p in self.players.values() if p.side == 'defense')
            if attack_loadout < defense_loadout * 0.7:
                suggestions.append("Economy disadvantage may have impacted combat - consider eco/force timing")

        # Defense suggestions
        if winner == 'attack':
            if self.spike_planted:
                plant_time = next((e.timestamp_ms for e in self.events if e.event_type == 'spike_plant'), 0)
                if plant_time < 30000:
                    suggestions.append("Early spike plant - consider faster rotations")
            suggestions.append("Review crossfire positions to catch entry fraggers")

        # General suggestions
        if len(kills) < 3:
            suggestions.append("Low kill count - consider more aggressive positioning")

        # First blood analysis
        if self.round_state.first_blood_team:
            fb_kills = [k for k in kills if k.details.get('is_first_blood', False)]
            if fb_kills:
                fb_time = fb_kills[0].timestamp_ms
                if fb_time < 15000:
                    suggestions.append(f"Early first blood at {fb_time}ms - review opening duels")

        # Trade analysis
        trades = [k for k in kills if k.details.get('is_trade', False)]
        if len(trades) > 0:
            trade_rate = len(trades) / max(1, len(kills) - 1)
            if trade_rate < 0.3:
                suggestions.append("Low trade rate - improve positioning for refrag opportunities")

        return {
            'winner': winner,
            'total_duration_ms': session.current_time_ms,
            'kills': [k.model_dump() for k in kills],
            'spike_planted': self.spike_planted,
            'spike_site': self.spike_site,
            'attack_alive': alive_attack,
            'defense_alive': alive_defense,
            'win_probability': win_prob,
            'first_blood_team': self.round_state.first_blood_team,
            'first_blood_time_ms': self.round_state.first_blood_time_ms,
            'improvement_suggestions': suggestions,
            'key_moments': [
                {'time_ms': e.timestamp_ms, 'type': e.event_type, 'details': e.details}
                for e in self.events
                if e.event_type in ['kill', 'spike_plant', 'spike_defuse', 'ability']
            ],
        }

    def _build_state(self, session: SimulationSession, current_time: int) -> SimulationState:
        """Build current simulation state response."""
        positions = [
            PlayerPosition(
                player_id=p.player_id,
                team_id=p.team_id,
                x=p.x,
                y=p.y,
                is_alive=p.is_alive,
                health=p.health,
                agent=p.agent,
                side=p.side,
                shield=p.shield,
                weapon_name=p.weapon_name,
                armor_name=p.armor_name,
                loadout_value=p.loadout_value,
                is_flashed=p.is_flashed,
                is_slowed=p.is_slowed,
                is_revealed=p.is_revealed,
                role=p.role.value if p.role else None,
            )
            for p in self.players.values()
        ]

        # Calculate win probability
        win_prob = WinProbabilityCalculator.calculate_win_probability(
            self.round_state, current_time
        )

        # Calculate alive counts
        attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        # Calculate loadout values
        attack_loadout = sum(p.loadout_value for p in self.players.values() if p.side == 'attack')
        defense_loadout = sum(p.loadout_value for p in self.players.values() if p.side == 'defense')

        return SimulationState(
            session_id=session.id,
            current_time_ms=current_time,
            phase=self._get_current_phase(current_time),
            status=session.status,
            positions=positions,
            events=self.events,
            spike_planted=self.spike_planted,
            spike_site=self.spike_site,
            win_probability=win_prob,
            first_blood_team=self.round_state.first_blood_team,
            first_blood_time_ms=self.round_state.first_blood_time_ms if self.round_state.first_blood_team else None,
            attack_alive=attack_alive,
            defense_alive=defense_alive,
            attack_buy_type=self.round_state.attack_buy_type,
            defense_buy_type=self.round_state.defense_buy_type,
            attack_loadout_value=attack_loadout,
            defense_loadout_value=defense_loadout,
            attack_strategy=self.attack_strategy.name if self.attack_strategy else None,
            defense_strategy=self.defense_strategy.name if self.defense_strategy else None,
        )
