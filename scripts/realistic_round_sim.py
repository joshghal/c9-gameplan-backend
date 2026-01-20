#!/usr/bin/env python3
"""
Realistic Round Simulation with Proper Phases

Phases:
1. SETUP (0-15s): Teams position, no engagements
2. MAP_CONTROL (15-40s): Info gathering, early picks possible
3. EXECUTE (40-60s): Site take, main engagements
4. POST_PLANT (after plant): Retake scenario
5. CLUTCH (when 1vX): Final moments
"""

import asyncio
import random
import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer, PlayerTendencies
from app.services.weapon_system import WeaponDatabase
from app.services.data_loader import get_data_loader
from app.services.validated_parameters import RoundPhase, PHASE_PARAMS
from app.services.trade_system import TradeSystem, ReadinessState
from app.services.behavior_adaptation import BehaviorAdapter, SpikeCarrierTendencies


# =============================================================================
# MAP COORDINATE SYSTEMS (from cloud9-webapp/src/config/maps.ts)
# =============================================================================
# Formula: minimapX = gameY * xMultiplier + xScalarToAdd
#          minimapY = gameX * yMultiplier + yScalarToAdd
# Note: X and Y are swapped in the formula

MAP_TRANSFORMS = {
    'ascent': {'xMult': 7e-05, 'yMult': -7e-05, 'xAdd': 0.813895, 'yAdd': 0.573242},
    'bind': {'xMult': 5.9e-05, 'yMult': -5.9e-05, 'xAdd': 0.576941, 'yAdd': 0.967566},
    'haven': {'xMult': 7.5e-05, 'yMult': -7.5e-05, 'xAdd': 1.09345, 'yAdd': 0.642728},
    'split': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.842188, 'yAdd': 0.697578},
    'icebox': {'xMult': 7.2e-05, 'yMult': -7.2e-05, 'xAdd': 0.460214, 'yAdd': 0.304687},
    'breeze': {'xMult': 7e-05, 'yMult': -7e-05, 'xAdd': 0.465123, 'yAdd': 0.833078},
    'fracture': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.556952, 'yAdd': 1.155886},
    'pearl': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.480469, 'yAdd': 0.916016},
    'lotus': {'xMult': 7.2e-05, 'yMult': -7.2e-05, 'xAdd': 0.454789, 'yAdd': 0.917752},
    'sunset': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.5, 'yAdd': 0.515625},
    'abyss': {'xMult': 8.1e-05, 'yMult': -8.1e-05, 'xAdd': 0.5, 'yAdd': 0.5},
}

# Current map for simulation
CURRENT_MAP = 'ascent'

# VCT GRID walkable boundaries (from 33 pro match JSONL files)
# These are more accurate for pro-level positioning
MAP_WALKABLE_BOUNDS = {
    'abyss': {'x': (0.057, 0.943), 'y': (0.044, 0.984)},
    'ascent': {'x': (0.071, 0.897), 'y': (0.096, 0.874)},
    'bind': {'x': (0.178, 0.868), 'y': (0.077, 0.917)},
    'corrode': {'x': (0.394, 0.988), 'y': (0.179, 0.975)},
    'fracture': {'x': (0.042, 0.904), 'y': (0.226, 0.871)},
    'haven': {'x': (0.082, 0.951), 'y': (0.107, 0.889)},
    'icebox': {'x': (0.100, 0.952), 'y': (0.085, 0.895)},
    'lotus': {'x': (0.071, 0.920), 'y': (0.141, 0.876)},
    'pearl': {'x': (0.053, 0.920), 'y': (0.050, 0.934)},
    'split': {'x': (0.107, 0.905), 'y': (0.076, 0.944)},
    'sunset': {'x': (0.062, 0.940), 'y': (0.071, 0.952)},
}


def clamp_to_walkable(x: float, y: float, map_name: str = None) -> tuple:
    """Clamp position to VCT-validated walkable area."""
    map_name = map_name or CURRENT_MAP
    bounds = MAP_WALKABLE_BOUNDS.get(map_name, MAP_WALKABLE_BOUNDS['ascent'])
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    return (
        max(x_min, min(x_max, x)),
        max(y_min, min(y_max, y))
    )


def game_to_minimap(game_x: float, game_y: float, map_name: str = None) -> tuple:
    """Convert game coordinates to minimap normalized (0-1) coordinates.

    Uses exact formula from cloud9-webapp:
    minimapX = gameY * xMultiplier + xScalarToAdd
    minimapY = gameX * yMultiplier + yScalarToAdd
    """
    map_name = map_name or CURRENT_MAP
    t = MAP_TRANSFORMS.get(map_name, MAP_TRANSFORMS['ascent'])

    minimap_x = game_y * t['xMult'] + t['xAdd']
    minimap_y = game_x * t['yMult'] + t['yAdd']

    # Clamp to VCT-validated walkable area (not just 0-1)
    return clamp_to_walkable(minimap_x, minimap_y, map_name)


# Ascent named locations (game coordinates from VCT GRID data)
# VCT Ascent bounds: (-4850, -11100) to (7550, 2200)
ASCENT_COORDS = {
    # Sites (from VCT plant positions)
    'a_site': (6100, -6600),           # A Site center
    'b_site': (-2300, -7500),          # B Site center

    # Spawns (from VCT round start positions)
    'attacker_spawn': (100, 1100),     # VCT: ~(109, 1086)
    'defender_spawn': (1900, -10500),  # VCT: ~(1882, -10498)

    # A Side approaches
    'a_lobby': (3500, -2500),          # A Lobby
    'a_main': (5000, -4500),           # A Main
    'a_short': (4500, -5500),          # A Short/Tree
    'a_heaven': (6500, -7500),         # A Heaven (CT side)

    # B Side approaches
    'b_lobby': (-1500, -2000),         # B Lobby
    'b_main': (-2000, -5500),          # B Main
    'b_ct': (-3500, -8500),            # B CT/Defender side

    # Mid
    'mid_top': (1500, -1500),          # Mid Top (attacker)
    'mid_bottom': (2000, -5500),       # Mid Bottom (catwalk)
    'mid_cubby': (1000, -4000),        # Mid Cubby
    'mid_market': (-500, -4500),       # Market
}

def get_ascent_pos(location: str) -> tuple:
    """Get minimap position for a named Ascent location."""
    if location in ASCENT_COORDS:
        return game_to_minimap(*ASCENT_COORDS[location])
    return (0.5, 0.5)  # Default to center


# Engagement zones - rectangular areas where fights can happen (minimap coords)
# Players can only engage if they're in the SAME zone or ADJACENT zones
ENGAGEMENT_ZONES = {
    'attacker_spawn': {'bounds': (0.75, 0.45, 0.95, 0.70), 'adjacent': ['mid_top']},
    'defender_spawn': {'bounds': (0.02, 0.35, 0.15, 0.50), 'adjacent': ['mid_bottom', 'a_ct', 'b_ct']},
    'mid_top': {'bounds': (0.55, 0.35, 0.75, 0.55), 'adjacent': ['attacker_spawn', 'mid_bottom', 'a_lobby', 'b_lobby']},
    'mid_bottom': {'bounds': (0.35, 0.40, 0.55, 0.60), 'adjacent': ['mid_top', 'a_main', 'b_main', 'market']},
    'a_lobby': {'bounds': (0.50, 0.20, 0.70, 0.40), 'adjacent': ['mid_top', 'a_main']},
    'a_main': {'bounds': (0.35, 0.15, 0.55, 0.35), 'adjacent': ['a_lobby', 'a_site', 'mid_bottom']},
    'a_site': {'bounds': (0.25, 0.05, 0.45, 0.25), 'adjacent': ['a_main', 'a_ct', 'a_heaven']},
    'a_ct': {'bounds': (0.15, 0.10, 0.30, 0.30), 'adjacent': ['a_site', 'defender_spawn', 'a_heaven']},
    'a_heaven': {'bounds': (0.20, 0.02, 0.35, 0.15), 'adjacent': ['a_site', 'a_ct']},
    'b_lobby': {'bounds': (0.50, 0.55, 0.70, 0.75), 'adjacent': ['mid_top', 'b_main']},
    'b_main': {'bounds': (0.30, 0.55, 0.50, 0.80), 'adjacent': ['b_lobby', 'b_site', 'market']},
    'b_site': {'bounds': (0.20, 0.65, 0.40, 0.85), 'adjacent': ['b_main', 'b_ct']},
    'b_ct': {'bounds': (0.10, 0.55, 0.25, 0.75), 'adjacent': ['b_site', 'defender_spawn']},
    'market': {'bounds': (0.25, 0.45, 0.40, 0.60), 'adjacent': ['mid_bottom', 'b_main']},
}


def get_player_zone(x: float, y: float) -> Optional[str]:
    """Get which engagement zone a player is in."""
    for zone_name, zone_data in ENGAGEMENT_ZONES.items():
        x_min, y_min, x_max, y_max = zone_data['bounds']
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone_name
    return None


def can_players_engage(p1_x: float, p1_y: float, p2_x: float, p2_y: float) -> bool:
    """Check if two players can engage (same zone or adjacent zones)."""
    zone1 = get_player_zone(p1_x, p1_y)
    zone2 = get_player_zone(p2_x, p2_y)

    # Both must be in a zone
    if zone1 is None or zone2 is None:
        return False

    # Same zone = can engage
    if zone1 == zone2:
        return True

    # Adjacent zones = can engage
    if zone2 in ENGAGEMENT_ZONES[zone1]['adjacent']:
        return True

    return False


@dataclass
class RoundEvent:
    time_ms: int
    event_type: str  # kill, trade, plant, defuse, ability
    actor: str
    target: Optional[str]
    details: Dict


class RealisticRoundSimulator:
    """Simulates rounds with proper tactical phases.

    Player names and agents are configurable - defaults are for testing.
    All timing/engagement parameters come from validated_parameters.py
    """

    # Default players for testing - in real use these come from scenario config
    DEFAULT_ATTACK_PLAYERS = ['OXY', 'xeppaa', 'v1c', 'Jakee', 'mitch']
    DEFAULT_DEFENSE_PLAYERS = ['aspas', 'Sato', 'kiNgg', 'C0M', 'tex']
    DEFAULT_AGENTS_ATTACK = ['jett', 'raze', 'sova', 'omen', 'killjoy']
    DEFAULT_AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'kayo', 'sage']

    # All timing from validated_parameters.py
    SETUP_END = PHASE_PARAMS.SETUP_END_MS
    CONTROL_END = PHASE_PARAMS.CONTROL_END_MS
    EXECUTE_END = PHASE_PARAMS.EXECUTE_END_MS
    ROUND_END = PHASE_PARAMS.ROUND_TIME_LIMIT_MS
    PLANT_TIME = 4000   # ✅ RIOT: 4s plant time
    DEFUSE_TIME = 7000  # ✅ RIOT: 7s defuse (4s with half)

    # Engagement rates from validated_parameters.py
    ENGAGEMENT_RATES = {
        RoundPhase.SETUP: PHASE_PARAMS.ENGAGEMENT_RATE_SETUP,
        RoundPhase.MAP_CONTROL: PHASE_PARAMS.ENGAGEMENT_RATE_CONTROL,
        RoundPhase.EXECUTE: PHASE_PARAMS.ENGAGEMENT_RATE_EXECUTE,
        RoundPhase.POST_PLANT: PHASE_PARAMS.ENGAGEMENT_RATE_POST_PLANT,
        RoundPhase.CLUTCH: PHASE_PARAMS.ENGAGEMENT_RATE_CLUTCH,
    }

    # Trade params - now uses readiness-based system from TradeSystem
    # These are kept for backwards compatibility but readiness-based is preferred
    TRADE_WINDOW = 3000  # Extended to 3s (VCT: 79.3% within 3s)
    TRADE_DISTANCE = 0.10  # Only used as secondary factor now
    TRADE_BOOST = 2.0  # Less important with readiness-based

    def __init__(
        self,
        attack_players: Optional[List[str]] = None,
        defense_players: Optional[List[str]] = None,
        attack_agents: Optional[List[str]] = None,
        defense_agents: Optional[List[str]] = None,
    ):
        """Initialize simulator with optional player/agent configuration.

        Args:
            attack_players: List of attacking player names (uses VCT data if available)
            defense_players: List of defending player names
            attack_agents: List of agents for attackers
            defense_agents: List of agents for defenders
        """
        self.data_loader = get_data_loader()
        self.events: List[RoundEvent] = []
        self.players: Dict[str, SimulatedPlayer] = {}
        self.spike_planted = False
        self.spike_plant_time = 0
        self.spike_site = None
        self.last_kill_time = 0
        self.last_kill_victim_pos = None

        # NEW: Readiness-based trade system
        self.trade_system = TradeSystem()

        # Use provided players or defaults
        self.attack_players = attack_players or self.DEFAULT_ATTACK_PLAYERS
        self.defense_players = defense_players or self.DEFAULT_DEFENSE_PLAYERS
        self.attack_agents = attack_agents or self.DEFAULT_AGENTS_ATTACK
        self.defense_agents = defense_agents or self.DEFAULT_AGENTS_DEFENSE

    def _get_phase_by_time(self, time_ms: int) -> RoundPhase:
        """Get phase based on time only (for visualization)."""
        if time_ms < self.SETUP_END:
            return RoundPhase.SETUP
        elif time_ms < self.CONTROL_END:
            return RoundPhase.MAP_CONTROL
        elif time_ms < self.EXECUTE_END:
            return RoundPhase.EXECUTE
        else:
            return RoundPhase.POST_PLANT

    def _get_phase(self, time_ms: int) -> RoundPhase:
        """Determine current round phase."""
        attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        # Check for clutch
        if attack_alive == 1 or defense_alive == 1:
            return RoundPhase.CLUTCH

        # Check for post-plant
        if self.spike_planted:
            return RoundPhase.POST_PLANT

        # Time-based phases
        if time_ms < self.SETUP_END:
            return RoundPhase.SETUP
        elif time_ms < self.CONTROL_END:
            return RoundPhase.MAP_CONTROL
        else:
            return RoundPhase.EXECUTE

    def _get_phase_positions(self, phase: RoundPhase, side: str) -> List[Tuple[float, float]]:
        """Get tactical positions for each phase.

        VCT SPAWN DATA (verified):
        - ATK spawn: minimap ~(0.86-0.88, 0.53-0.57) = RIGHT side of map
        - DEF spawn: minimap ~(0.08-0.10, 0.44-0.47) = LEFT side of map
        - A Site: minimap ~(0.35, 0.15) = TOP-LEFT
        - B Site: minimap ~(0.29, 0.73) = BOTTOM-LEFT

        Teams move TOGETHER, not spread randomly.
        """
        if side == 'attack':
            if phase == RoundPhase.SETUP:
                # ALL attackers clustered at spawn (RIGHT side)
                # VCT spawn: minimap ~(0.85, 0.57), so game ~(50, 500)
                return [
                    game_to_minimap(50, 520),       # Spawn cluster
                    game_to_minimap(100, 550),
                    game_to_minimap(0, 480),
                    game_to_minimap(150, 580),
                    game_to_minimap(70, 510),
                ]
            elif phase == RoundPhase.MAP_CONTROL:
                # Team pushing together toward A (mid area)
                # Moving from spawn (right) toward A site (top-left)
                return [
                    game_to_minimap(1500, -1500),   # Lead - mid top
                    game_to_minimap(1800, -1200),   # Support
                    game_to_minimap(1200, -1800),   # Trail
                    game_to_minimap(2000, -1000),   # Lurk
                    game_to_minimap(1600, -1600),   # Core
                ]
            elif phase == RoundPhase.EXECUTE:
                # Executing onto A site - converging
                # A site at game ~(6100, -6600)
                return [
                    game_to_minimap(4500, -5000),   # Entry
                    game_to_minimap(4800, -5300),   # Second
                    game_to_minimap(5200, -5600),   # Site approach
                    game_to_minimap(4200, -4800),   # Support
                    game_to_minimap(5000, -5500),   # Trade
                ]
            elif phase == RoundPhase.POST_PLANT:
                # Holding post-plant around A site
                return [
                    game_to_minimap(5500, -5800),   # Close angle
                    game_to_minimap(5800, -6000),   # Site
                    game_to_minimap(5200, -5500),   # Main peek
                    game_to_minimap(5600, -6200),   # Back site
                    game_to_minimap(5000, -5600),   # Deep
                ]
            else:  # Clutch
                return [
                    game_to_minimap(5500, -5900),
                    game_to_minimap(5600, -6000),
                    game_to_minimap(5400, -5800),
                    game_to_minimap(5700, -6100),
                    game_to_minimap(5300, -5700),
                ]
        else:  # Defense
            if phase == RoundPhase.SETUP:
                # ALL defenders clustered at spawn (LEFT side)
                # VCT spawn: minimap ~(0.08, 0.40), so game ~(2475, -10484)
                return [
                    game_to_minimap(2450, -10500),  # Spawn cluster
                    game_to_minimap(2500, -10450),
                    game_to_minimap(2400, -10550),
                    game_to_minimap(2550, -10400),
                    game_to_minimap(2480, -10480),
                ]
            elif phase == RoundPhase.MAP_CONTROL:
                # Defenders spread to hold sites and mid
                # 2 on A, 1 mid, 2 on B
                return [
                    game_to_minimap(5500, -6000),   # A site anchor
                    game_to_minimap(5000, -5500),   # A main peek
                    game_to_minimap(2000, -7000),   # Mid hold
                    game_to_minimap(-1500, -7500),  # B site anchor
                    game_to_minimap(-1000, -7000),  # B main peek
                ]
            elif phase in [RoundPhase.EXECUTE, RoundPhase.POST_PLANT]:
                # Retaking/defending A site
                return [
                    game_to_minimap(6000, -6500),   # A CT
                    game_to_minimap(5500, -6200),   # A site
                    game_to_minimap(4500, -5500),   # A main
                    game_to_minimap(5800, -6800),   # Garden
                    game_to_minimap(5200, -5800),   # Rotate
                ]
            else:  # Clutch
                return [
                    game_to_minimap(5800, -6300),
                    game_to_minimap(5500, -6000),
                    game_to_minimap(5200, -5700),
                    game_to_minimap(6000, -6500),
                    game_to_minimap(5600, -6100),
                ]

    def _setup_players(self):
        """Initialize players with real stats from VCT data when available.

        FIX: Aggressive players (duelists) spawn forward for early contact.
        This enables 15% early kills target.
        """
        self.players = {}

        # Attack team - spawn in attacker area using Ascent coordinates
        # Duelists pushed forward, support back at spawn
        attack_spawns = [
            game_to_minimap(2500, -1200),   # Duelist 1 - forward
            game_to_minimap(3000, -1500),   # Duelist 2 - A lobby approach
            game_to_minimap(1500, -800),    # Support - spawn
            game_to_minimap(2000, -600),    # Support - spawn
            game_to_minimap(0, -1000),      # Support - B lobby approach
        ]
        for i, name in enumerate(self.attack_players):
            profile = self.data_loader.get_player_profile(name)

            # Use real VCT stats if available, otherwise defaults
            hs_rate = profile.headshot_rate if profile and profile.headshot_rate > 0 else 0.25
            aggression = profile.aggression if profile else 0.5
            clutch = profile.clutch_potential if profile else 0.5

            agent_name = self.attack_agents[i % len(self.attack_agents)]
            player = SimulatedPlayer(
                player_id=f'atk_{name}',
                team_id='attack',
                side='attack',
                x=attack_spawns[i % len(attack_spawns)][0],
                y=attack_spawns[i % len(attack_spawns)][1],
                agent=agent_name,
                has_spike=(i == 0),
                headshot_rate=hs_rate,
                tendencies=PlayerTendencies(
                    base_aggression=aggression,
                    clutch_factor=clutch,
                    trade_awareness=0.6
                )
            )
            player.weapon = WeaponDatabase.get_weapon('vandal')
            player.shield = 50

            # Initialize spike carrier tendencies based on role and VCT profile
            role = self._get_role_from_agent(agent_name)
            player_profile_dict = None
            if profile:
                player_profile_dict = {
                    'aggression': profile.aggression,
                    'first_kill_rate': profile.first_kill_rate,
                    'clutch_potential': profile.clutch_potential,
                    'first_death_rate': getattr(profile, 'first_death_rate', 0.1),
                }
            player.spike_carrier_tendencies = BehaviorAdapter.create_spike_carrier_tendencies(
                role, player_profile_dict
            )

            self.players[player.player_id] = player

        # Defense team - using Ascent coordinates
        # One aggressive, rest hold sites and mid
        defense_spawns = [
            game_to_minimap(2000, -3500),   # Aggressive - mid push
            get_ascent_pos('a_site'),       # A site anchor
            get_ascent_pos('b_site'),       # B site anchor
            game_to_minimap(1000, -5000),   # Mid bottom hold
            game_to_minimap(0, -5500),      # Rotate position
        ]
        for i, name in enumerate(self.defense_players):
            profile = self.data_loader.get_player_profile(name)

            # Use real VCT stats if available, otherwise defaults
            hs_rate = profile.headshot_rate if profile and profile.headshot_rate > 0 else 0.25
            aggression = profile.aggression if profile else 0.5
            clutch = profile.clutch_potential if profile else 0.5

            agent_name = self.defense_agents[i % len(self.defense_agents)]
            player = SimulatedPlayer(
                player_id=f'def_{name}',
                team_id='defense',
                side='defense',
                x=defense_spawns[i % len(defense_spawns)][0],
                y=defense_spawns[i % len(defense_spawns)][1],
                agent=agent_name,
                headshot_rate=hs_rate,
                tendencies=PlayerTendencies(
                    base_aggression=aggression,
                    clutch_factor=clutch,
                    trade_awareness=0.6
                )
            )
            player.weapon = WeaponDatabase.get_weapon('vandal')
            player.shield = 50

            # Initialize spike carrier tendencies for defense too (for spike transfer scenarios)
            role = self._get_role_from_agent(agent_name)
            player_profile_dict = None
            if profile:
                player_profile_dict = {
                    'aggression': profile.aggression,
                    'first_kill_rate': profile.first_kill_rate,
                    'clutch_potential': profile.clutch_potential,
                    'first_death_rate': getattr(profile, 'first_death_rate', 0.1),
                }
            player.spike_carrier_tendencies = BehaviorAdapter.create_spike_carrier_tendencies(
                role, player_profile_dict
            )

            self.players[player.player_id] = player

    def _move_players(self, phase: RoundPhase, time_ms: int):
        """Move players based on phase."""
        attack_targets = self._get_phase_positions(phase, 'attack')
        defense_targets = self._get_phase_positions(phase, 'defense')

        attack_idx = 0
        defense_idx = 0

        # Calculate team sizes for retreat behavior
        attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        for pid, player in self.players.items():
            if not player.is_alive:
                continue

            # === RETREAT BEHAVIOR: Losing team moves to safety ===
            # This is KEY for realistic wipe rate - teams disengage when losing
            should_retreat = False
            retreat_target = None

            if player.side == 'attack':
                diff = defense_alive - attack_alive
                players_lost = 5 - attack_alive
                # Retreat toward attacker spawn (using Ascent coords)
                retreat_target = game_to_minimap(2000, -1000)  # Attacker spawn area
            else:
                diff = attack_alive - defense_alive
                players_lost = 5 - defense_alive
                # Retreat toward defender spawn (using Ascent coords)
                retreat_target = game_to_minimap(3000, -7000)  # Defender spawn area

            # Retreat conditions (less aggressive than before):
            # 1. Down 3+ players (heavy disadvantage) - always retreat
            # 2. Clutch 1vX - retreat unless forced to fight
            # 3. Post-plant attacker - spread out but stay near site
            if diff >= 3:
                should_retreat = True
            elif phase == RoundPhase.CLUTCH and player.side == 'attack' and not self.spike_planted:
                # 1vX attacker without plant - try to escape
                should_retreat = True
            elif phase == RoundPhase.POST_PLANT and player.side == 'attack':
                # VCT PATTERN: Attackers stay CLUSTERED near spike for crossfires
                # Don't retreat far - hold angles close to trade each other
                should_retreat = False  # Stay in crossfire position, don't run
                # If repositioning needed, stay near spike (using Ascent A site coords)
                a_site = ASCENT_COORDS['a_site']
                hide_spots = [
                    game_to_minimap(a_site[0] - 200, a_site[1]),
                    game_to_minimap(a_site[0] + 200, a_site[1]),
                    game_to_minimap(a_site[0], a_site[1] - 200),
                    game_to_minimap(a_site[0], a_site[1] + 200),
                ]
                retreat_target = random.choice(hide_spots)

            # Get normal target position
            if player.side == 'attack':
                target = attack_targets[attack_idx % len(attack_targets)]
                attack_idx += 1
            else:
                target = defense_targets[defense_idx % len(defense_targets)]
                defense_idx += 1

            # Override target if retreating
            if should_retreat and retreat_target:
                target = retreat_target

            # Movement speed based on phase and aggression
            aggr = player.tendencies.base_aggression if player.tendencies else 0.5

            if phase == RoundPhase.SETUP:
                speed = 0.001  # Slow positioning
            elif phase == RoundPhase.EXECUTE:
                speed = 0.004 * (0.8 + aggr * 0.4)  # Fast execute
            elif should_retreat:
                speed = 0.003  # Fast retreat
            else:
                speed = 0.002 * (0.7 + aggr * 0.6)  # Normal

            # Move toward target
            dx = target[0] - player.x
            dy = target[1] - player.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.02:
                player.x += (dx / dist) * speed
                player.y += (dy / dist) * speed

            # Add some randomness
            player.x += random.uniform(-0.005, 0.005)
            player.y += random.uniform(-0.005, 0.005)

            # Clamp to VCT-validated walkable area
            player.x, player.y = clamp_to_walkable(player.x, player.y)

    def _check_engagements(self, time_ms: int, phase: RoundPhase):
        """Check for engagements based on phase.

        Key behaviors for realistic wipe rate:
        - POST_PLANT: Attackers HIDE to run down clock (low engagement)
        - SETUP: No engagements until 12s (teams positioning, no contact)

        Uses READINESS-BASED trade system:
        - Trades based on readiness state (pre-aimed, normal, repositioning)
        - NOT based on distance (VCT shows distance has no correlation with trade time)
        """
        # No engagements during setup/positioning phase (first 12 seconds)
        # In competitive Valorant, contact before ~12s is extremely rare
        # Teams are still positioning and taking map control
        if time_ms < 12000:
            return

        engagement_rate = self.ENGAGEMENT_RATES[phase]

        # === NEW: Check for readiness-based trade opportunities ===
        # Get potential traders (alive players with their roles)
        potential_traders = []
        for pid, p in self.players.items():
            if p.is_alive:
                # Determine role from agent
                role = self._get_role_from_agent(p.agent)
                potential_traders.append((pid, (p.x, p.y), p.side, role))

        # Check and attempt trades using new readiness-based system
        trade_results = self.trade_system.check_and_attempt_trade(
            time_ms, potential_traders, has_los_func=None  # Simplified LOS
        )

        # Process any successful trades
        for trade_result in trade_results:
            # Find the killer (target of the trade)
            killer_id = trade_result.target_id
            if killer_id in self.players and self.players[killer_id].is_alive:
                killer = self.players[killer_id]
                # Find the trader
                trader_id = trade_result.trader_id
                if trader_id in self.players:
                    trader = self.players[trader_id]
                    # Execute the trade kill
                    self._execute_trade_kill(trader, killer, time_ms, trade_result)
                    return  # One engagement per tick

        for pid, player in list(self.players.items()):
            if not player.is_alive:
                continue

            for eid, enemy in self.players.items():
                if enemy.side == player.side or not enemy.is_alive:
                    continue

                # === ZONE-BASED ENGAGEMENT CHECK ===
                # Players can only fight if they're in the same zone or adjacent zones
                # This prevents "wallhack" kills across the map
                if not can_players_engage(player.x, player.y, enemy.x, enemy.y):
                    continue

                dx = enemy.x - player.x
                dy = enemy.y - player.y
                dist = math.sqrt(dx*dx + dy*dy)

                # Additional distance check within zone (close combat)
                if dist > 0.20:
                    continue

                # Engagement probability
                engage_prob = engagement_rate

                # === WIPE RATE FIX: Balanced Save Behavior ===
                # VCT: Only 18% of rounds end in elimination (wipe)
                # 82% end via spike outcomes (plant/defuse/timeout)
                # Balance: teams should save when losing but still fight sometimes

                attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
                defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

                # === ASYMMETRIC SAVE BEHAVIOR ===
                # Attackers must be more aggressive (need to plant to win)
                # Defenders can save more easily (timeout = win)
                if player.side == 'attack':
                    players_lost = 5 - attack_alive
                    diff = defense_alive - attack_alive
                else:
                    players_lost = 5 - defense_alive
                    diff = attack_alive - defense_alive

                # Attackers: Less save behavior (must fight to plant)
                # Defenders: More save behavior (can win by timeout)
                if player.side == 'attack':
                    # Attackers only save when heavily down
                    if players_lost >= 3:
                        engage_prob *= 0.5  # 3+ dead = 50% reduction
                    elif players_lost >= 2:
                        engage_prob *= 0.7  # 2 dead = 30% reduction
                    # No reduction for 1 player down - must fight
                else:
                    # Defenders save more aggressively
                    if players_lost >= 1:
                        save_factor = 0.65 ** players_lost
                        engage_prob *= save_factor

                # Disadvantage penalty (both sides)
                if diff >= 2:
                    engage_prob *= 0.5  # 50% reduction when 2+ down
                elif diff >= 1 and player.side == 'defense':
                    engage_prob *= 0.7  # Defenders save at 1 down

                # === POST-PLANT DYNAMICS ===
                # Attackers defend but don't hide completely
                # Defenders push for defuse
                if phase == RoundPhase.POST_PLANT:
                    if player.side == 'attack':
                        # Attackers defend spike - watching angles
                        engage_prob *= 0.4  # 60% reduction - defensive but ready

                        # Fight if enemy approaches spike
                        if enemy.y < 0.45:
                            engage_prob *= 2.5  # Defend spike!
                    else:
                        # Defenders push - they need kills to defuse
                        engage_prob *= 0.8  # Only 20% reduction - must push

                # === COOLDOWN AFTER KILLS ===
                # Brief cooldown for repositioning
                time_since_kill = time_ms - self.last_kill_time
                if time_since_kill < 3000:  # 3 second cooldown
                    # Decaying cooldown
                    cooldown_factor = 0.3 + (time_since_kill / 3000) * 0.5  # 0.3 to 0.8
                    engage_prob *= cooldown_factor

                # === CLUTCH BEHAVIOR ===
                # 1vX situations are cautious but decisive
                if phase == RoundPhase.CLUTCH:
                    if player.side == 'attack':
                        if self.spike_planted:
                            # Spike planted - hide but peek to delay
                            engage_prob *= 0.2  # 80% reduction - hiding
                        else:
                            # No spike - must make a play
                            engage_prob *= 0.5  # 50% reduction
                    else:
                        # Defense in clutch - hunt actively
                        if self.spike_planted:
                            engage_prob *= 1.0  # Full aggression - must find attacker
                        else:
                            engage_prob *= 0.6  # Can wait for timeout

                # === FIX 2: EARLY AGGRESSION ===
                # Some players push aggressively in SETUP for info picks
                if phase == RoundPhase.SETUP:
                    # High aggression players more likely to push
                    player_aggr = player.tendencies.base_aggression if player.tendencies else 0.5
                    if player_aggr > 0.6:  # Aggressive player
                        engage_prob *= 2.5  # More likely to take early fights (reduced from 3.0)

                    # Duelists especially aggressive early
                    if player.agent in ['jett', 'raze', 'reyna', 'phoenix', 'neon']:
                        engage_prob *= 1.8  # Reduced from 2.0

                # === SPIKE CARRIER PROTECTION (Per-Player) ===
                # Spike carrier behavior varies by role and player tendencies
                # Replaces fixed 0.3/0.5 modifiers with VCT-derived per-player values
                if player.side == 'attack' and player.has_spike and not self.spike_planted:
                    # Get spike carrier tendencies (initialized during player creation)
                    tendencies = getattr(player, 'spike_carrier_tendencies', None)
                    if tendencies is None:
                        # Fallback to role-based defaults if not initialized
                        role = self._get_role_from_agent(player.agent)
                        tendencies = BehaviorAdapter.create_spike_carrier_tendencies(role)

                    # Calculate man advantage for context
                    attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
                    defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)
                    man_advantage = attack_alive - defense_alive

                    # Get per-player engagement modifier based on phase, man advantage, and time
                    carrier_mod = tendencies.get_engagement_modifier(phase.value, man_advantage, time_ms)
                    engage_prob *= carrier_mod

                if random.random() < engage_prob:
                    self._resolve_engagement(player, enemy, dist, time_ms, phase)
                    return  # One engagement per tick

    def _get_role_from_agent(self, agent: str) -> str:
        """Map agent to role for trade system."""
        duelists = ['jett', 'raze', 'reyna', 'phoenix', 'neon', 'yoru', 'iso']
        initiators = ['sova', 'skye', 'breach', 'kayo', 'fade', 'gekko']
        controllers = ['omen', 'brimstone', 'astra', 'viper', 'harbor', 'clove']
        sentinels = ['killjoy', 'cypher', 'sage', 'chamber', 'deadlock', 'vyse']

        if agent.lower() in duelists:
            return 'duelist'
        elif agent.lower() in initiators:
            return 'initiator'
        elif agent.lower() in controllers:
            return 'controller'
        elif agent.lower() in sentinels:
            return 'sentinel'
        return 'initiator'  # Default

    def _execute_trade_kill(self, trader: SimulatedPlayer, target: SimulatedPlayer,
                           time_ms: int, trade_result):
        """Execute a trade kill based on readiness-based trade system."""
        target.is_alive = False
        target.health = 0
        trader.kills += 1
        target.deaths += 1

        # === SPIKE TRANSFER ===
        # When spike carrier dies, spike transfers to nearest alive teammate
        if target.has_spike and not self.spike_planted:
            target.has_spike = False
            closest_teammate = None
            closest_dist = float('inf')
            for p in self.players.values():
                if p.side == target.side and p.is_alive and p.player_id != target.player_id:
                    d = math.sqrt((p.x - target.x)**2 + (p.y - target.y)**2)
                    if d < closest_dist:
                        closest_dist = d
                        closest_teammate = p
            if closest_teammate:
                closest_teammate.has_spike = True

        is_headshot = random.random() < trader.headshot_rate

        # Calculate distance for logging
        dist = math.sqrt(
            (trader.x - target.x)**2 + (trader.y - target.y)**2
        )

        self.events.append(RoundEvent(
            time_ms=time_ms,
            event_type='trade',
            actor=trader.player_id.split('_')[1],
            target=target.player_id.split('_')[1],
            details={
                'headshot': is_headshot,
                'distance': dist,
                'phase': self._get_phase(time_ms).value,
                'killer_team': 'C9' if trader.side == 'attack' else 'LOUD',
                'victim_team': 'C9' if target.side == 'attack' else 'LOUD',
                'readiness_state': trade_result.readiness_state.value,
                'trade_time_ms': trade_result.trade_time_ms,
            }
        ))

        # Update last kill tracking
        self.last_kill_time = time_ms
        self.last_kill_victim_pos = (target.x, target.y)

        # Record this kill for potential future trades
        self.trade_system.record_kill(
            timestamp_ms=time_ms,
            killer_id=trader.player_id,
            killer_team=trader.side,
            killer_pos=(trader.x, trader.y),
            victim_id=target.player_id,
            victim_team=target.side,
            victim_pos=(target.x, target.y),
        )

    def _resolve_engagement(self, player1: SimulatedPlayer, player2: SimulatedPlayer,
                           dist: float, time_ms: int, phase: RoundPhase):
        """Resolve combat between two players."""
        # Calculate win probability based on stats
        p1_skill = player1.headshot_rate + 0.1
        p2_skill = player2.headshot_rate + 0.1

        # Aggression bonus in execute phase
        if phase == RoundPhase.EXECUTE:
            p1_aggr = player1.tendencies.base_aggression if player1.tendencies else 0.5
            p2_aggr = player2.tendencies.base_aggression if player2.tendencies else 0.5
            p1_skill *= (1.0 + p1_aggr * 0.2)
            p2_skill *= (1.0 + p2_aggr * 0.2)

        # Clutch factor in clutch phase
        if phase == RoundPhase.CLUTCH:
            p1_clutch = player1.tendencies.clutch_factor if player1.tendencies else 0.5
            p2_clutch = player2.tendencies.clutch_factor if player2.tendencies else 0.5
            p1_skill *= (0.8 + p1_clutch * 0.4)
            p2_skill *= (0.8 + p2_clutch * 0.4)

        # === PHASE-BASED ADVANTAGES ===
        # Tuned to achieve ~47% attack win rate (VCT target)

        # SETUP: Defenders hold angles, slight advantage
        if phase == RoundPhase.SETUP:
            if player2.side == 'defense':
                p2_skill *= 1.08  # Reduced (was 1.10)
            elif player1.side == 'defense':
                p1_skill *= 1.08

        # MAP_CONTROL: Attackers have utility advantage (flashes, smokes)
        # This helps attackers survive to plant (increases plant rate)
        if phase == RoundPhase.MAP_CONTROL:
            if player1.side == 'attack':
                p1_skill *= 1.25  # Boosted (was 1.15)
            elif player2.side == 'attack':
                p2_skill *= 1.25

        # EXECUTE: Attackers coordinate for site take
        if phase == RoundPhase.EXECUTE:
            if player1.side == 'attack':
                p1_skill *= 1.30  # Boosted (was 1.20)
            elif player2.side == 'attack':
                p2_skill *= 1.30

        # POST_PLANT: Attackers have strong positional advantage
        # VCT data shows ~65% post-plant win rate for attackers
        if phase == RoundPhase.POST_PLANT:
            if player1.side == 'attack':
                p1_skill *= 1.50  # Boosted (was 1.40)
            elif player2.side == 'attack':
                p2_skill *= 1.50

        p1_win = p1_skill / (p1_skill + p2_skill)

        if random.random() < p1_win:
            winner, loser = player1, player2
        else:
            winner, loser = player2, player1

        loser.is_alive = False
        loser.health = 0
        winner.kills += 1
        loser.deaths += 1

        # === SPIKE TRANSFER ===
        # When spike carrier dies, spike transfers to nearest alive teammate
        if loser.has_spike and not self.spike_planted:
            loser.has_spike = False
            # Find closest alive teammate
            closest_teammate = None
            closest_dist = float('inf')
            for p in self.players.values():
                if p.side == loser.side and p.is_alive and p.player_id != loser.player_id:
                    d = math.sqrt((p.x - loser.x)**2 + (p.y - loser.y)**2)
                    if d < closest_dist:
                        closest_dist = d
                        closest_teammate = p
            if closest_teammate:
                closest_teammate.has_spike = True

        is_headshot = random.random() < winner.headshot_rate

        # Record this kill as a regular kill (not a trade)
        # Trades are now handled by the readiness-based trade system
        self.events.append(RoundEvent(
            time_ms=time_ms,
            event_type='kill',
            actor=winner.player_id.split('_')[1],
            target=loser.player_id.split('_')[1],
            details={
                'headshot': is_headshot,
                'distance': dist,
                'phase': phase.value,
                'killer_team': 'C9' if winner.side == 'attack' else 'LOUD',
                'victim_team': 'C9' if loser.side == 'attack' else 'LOUD',
            }
        ))

        # Update last kill tracking
        self.last_kill_time = time_ms
        self.last_kill_victim_pos = (loser.x, loser.y)

        # NEW: Record this kill in the trade system so it can be traded
        # The readiness-based system will check for trade opportunities
        self.trade_system.record_kill(
            timestamp_ms=time_ms,
            killer_id=winner.player_id,
            killer_team=winner.side,
            killer_pos=(winner.x, winner.y),
            victim_id=loser.player_id,
            victim_team=loser.side,
            victim_pos=(loser.x, loser.y),
        )

    def _check_plant(self, time_ms: int, phase: RoundPhase):
        """Check if attackers can plant.

        VCT-derived targets (from extracted GRID data):
        - Plant rate: ~68% of attack rounds
        - Avg plant time: ~74.5s (most plants in EXECUTE phase 40-60s, many late)
        """
        if self.spike_planted:
            return

        # Only allow plants in EXECUTE phase (40s+) or very late MAP_CONTROL (38s+)
        # This matches VCT where plants typically happen 50-80s into round
        if phase == RoundPhase.SETUP:
            return
        if phase == RoundPhase.MAP_CONTROL and time_ms < 38000:
            return

        # Find spike carrier
        spike_carrier = None
        for p in self.players.values():
            if p.side == 'attack' and p.is_alive and p.has_spike:
                spike_carrier = p
                break

        if not spike_carrier:
            return

        # Site positions in minimap coordinates (from Ascent game coords)
        a_site_pos = get_ascent_pos('a_site')  # ~(0.350, 0.142)
        b_site_pos = get_ascent_pos('b_site')  # ~(0.285, 0.737)

        # Check distance to sites (player must be within ~0.15 normalized distance)
        # This represents roughly 15m radius around each site
        dist_to_a = math.sqrt((spike_carrier.x - a_site_pos[0])**2 + (spike_carrier.y - a_site_pos[1])**2)
        dist_to_b = math.sqrt((spike_carrier.x - b_site_pos[0])**2 + (spike_carrier.y - b_site_pos[1])**2)

        plant_radius = 0.15  # ~15m radius around site center
        in_a_site = dist_to_a < plant_radius
        in_b_site = dist_to_b < plant_radius

        if in_a_site or in_b_site:
            # Plant probability based on situation
            attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
            defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

            # VCT-tuned plant probability - targeting ~68% plant rate, avg 48-52s plant time
            # Base probability is LOW early, increases over time
            # Balance: enough plants to get ~68% rate, bias towards 48-55s

            # Time-based scaling: moderate early, peaks at 50-60s
            if time_ms < 40000:  # Before EXECUTE
                plant_prob = 0.007  # Rare early plants
            elif time_ms < 46000:  # Early EXECUTE (40-46s)
                plant_prob = 0.014  # Building up
            elif time_ms < 50000:  # Mid EXECUTE (46-50s)
                plant_prob = 0.030  # Good probability
            elif time_ms < 56000:  # Prime time (50-56s) - sweet spot
                plant_prob = 0.052  # Higher probability
            elif time_ms < 62000:  # Late EXECUTE (56-62s)
                plant_prob = 0.058  # Still high
            elif time_ms < 72000:  # Post-execute (62-72s)
                plant_prob = 0.08   # High probability
            elif time_ms < 82000:  # Late round (72-82s)
                plant_prob = 0.15   # Very high to avoid timeout
            else:  # Urgent (82s+)
                plant_prob = 0.25   # Must plant now

            # Man advantage modifier (±30%)
            if attack_alive > defense_alive:
                plant_prob *= 1.3
            elif attack_alive < defense_alive:
                plant_prob *= 0.7  # More cautious when outnumbered

            if random.random() < plant_prob:
                self.spike_planted = True
                self.spike_plant_time = time_ms
                self.spike_site = 'A' if in_a_site else 'B'

                # Move spike carrier TO the site when planting
                site_pos = a_site_pos if in_a_site else b_site_pos
                spike_carrier.x = site_pos[0]
                spike_carrier.y = site_pos[1]

                # Move other alive attackers near the planted site (post-plant positions)
                for p in self.players.values():
                    if p.side == 'attack' and p.is_alive and p != spike_carrier:
                        # Position around site for crossfires
                        offset_x = random.uniform(-0.08, 0.08)
                        offset_y = random.uniform(-0.08, 0.08)
                        p.x, p.y = clamp_to_walkable(site_pos[0] + offset_x, site_pos[1] + offset_y)

                self.events.append(RoundEvent(
                    time_ms=time_ms,
                    event_type='plant',
                    actor=spike_carrier.player_id.split('_')[1],
                    target=None,
                    details={'site': self.spike_site}
                ))

    def _check_defuse(self, time_ms: int, phase: RoundPhase) -> Optional[str]:
        """Check if defenders can defuse. Returns defuser name if successful."""
        if not self.spike_planted:
            return None

        # Find defender near spike (using Ascent site coordinates)
        spike_pos = get_ascent_pos('a_site') if self.spike_site == 'A' else get_ascent_pos('b_site')

        for p in self.players.values():
            if p.side != 'defense' or not p.is_alive:
                continue

            # Distance to spike
            dist = math.sqrt((p.x - spike_pos[0])**2 + (p.y - spike_pos[1])**2)

            if dist < 0.08:  # Close enough to defuse
                # Defuse probability based on time remaining
                time_since_plant = time_ms - self.spike_plant_time
                time_remaining = 45000 - time_since_plant

                attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)

                # Base defuse probability - tuned for realistic outcomes
                # In VCT, ~31% of planted rounds end in defuse
                # But defuse requires: reaching spike (risky) + 7s defuse time
                if time_remaining > 30000:
                    defuse_prob = 0.0002  # Very risky early - attackers watching
                elif time_remaining > 15000:
                    defuse_prob = 0.0005  # Might attempt if clear
                elif time_remaining > 7000:
                    defuse_prob = 0.001   # Must try soon
                else:
                    defuse_prob = 0.003   # Last chance - desperation

                # More likely if attackers eliminated or few remaining
                if attack_alive == 0:
                    defuse_prob = 0.02  # Free defuse
                elif attack_alive == 1:
                    defuse_prob *= 2.0  # Easier with only 1 defender

                if random.random() < defuse_prob:
                    self.events.append(RoundEvent(
                        time_ms=time_ms,
                        event_type='defuse',
                        actor=p.player_id.split('_')[1],
                        target=None,
                        details={'time_remaining': time_remaining}
                    ))
                    return p.player_id.split('_')[1]

        return None

    async def run_round(self) -> Dict:
        """Run a complete round simulation."""
        self._setup_players()
        self.events = []
        self.spike_planted = False
        self.last_kill_time = -10000
        self.last_kill_victim_pos = None
        self.trade_system.reset()  # Reset trade system for new round

        # Track position history at key timestamps for visualization
        self.position_history = {}
        snapshot_times = [0, 15000, 40000, 45000, 50000, 55000, 60000, 70000, 80000, 90000]

        time_ms = 0
        tick_ms = 128

        while time_ms < self.ROUND_END:
            # Record position snapshots at key times
            for snap_time in snapshot_times:
                if snap_time not in self.position_history and time_ms >= snap_time:
                    self.position_history[snap_time] = {
                        pid: (p.x, p.y, p.is_alive)
                        for pid, p in self.players.items()
                    }
            attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
            defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

            # Check win conditions
            if attack_alive == 0:
                return self._build_result('LOUD', time_ms, 'elimination')
            if defense_alive == 0:
                return self._build_result('C9', time_ms, 'elimination')

            # Check spike timer
            if self.spike_planted and (time_ms - self.spike_plant_time) >= 45000:
                return self._build_result('C9', time_ms, 'detonation')

            phase = self._get_phase(time_ms)

            # Move players
            self._move_players(phase, time_ms)

            # Check for plant
            self._check_plant(time_ms, phase)

            # Check for defuse
            defuser = self._check_defuse(time_ms, phase)
            if defuser:
                return self._build_result('LOUD', time_ms, 'defuse')

            # Check engagements
            self._check_engagements(time_ms, phase)

            time_ms += tick_ms

        # Time ran out
        if self.spike_planted:
            return self._build_result('C9', time_ms, 'timeout_planted')
        else:
            return self._build_result('LOUD', time_ms, 'timeout')

    def _build_result(self, winner: str, duration: int, win_condition: str) -> Dict:
        """Build round result."""
        attack_alive = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        defense_alive = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        # Count trades
        trades = sum(1 for e in self.events if e.event_type == 'trade')

        # Kill timing distribution
        kill_times = [e.time_ms for e in self.events if e.event_type in ['kill', 'trade']]

        return {
            'winner': winner,
            'win_condition': win_condition,
            'duration': duration,
            'attack_alive': attack_alive,
            'defense_alive': defense_alive,
            'spike_planted': self.spike_planted,
            'events': self.events,
            'trades': trades,
            'kill_times': kill_times,
            'player_stats': {
                pid.split('_')[1]: {'kills': p.kills, 'deaths': p.deaths, 'alive': p.is_alive}
                for pid, p in self.players.items()
            },
            'position_history': self.position_history,
        }


async def main():
    sim = RealisticRoundSimulator()

    print('='*90)
    print('REALISTIC ROUND SIMULATION - Phased Combat')
    print('='*90)
    print()
    print('Phases: SETUP (0-15s) -> MAP_CONTROL (15-40s) -> EXECUTE (40-60s) -> POST_PLANT/CLUTCH')
    print()

    # Run single detailed round
    print('DETAILED ROUND:')
    print('-'*90)

    result = await sim.run_round()

    for event in result['events']:
        phase_str = f"[{event.details.get('phase', '?'):8}]"

        if event.event_type == 'plant':
            print(f"[{event.time_ms/1000:5.1f}s] {phase_str} PLANT: {event.actor} plants spike on {event.details['site']}")
        elif event.event_type == 'defuse':
            print(f"[{event.time_ms/1000:5.1f}s] {phase_str} DEFUSE: {event.actor} defuses spike")
        elif event.event_type == 'trade':
            hs = '(HS)' if event.details.get('headshot') else ''
            print(f"[{event.time_ms/1000:5.1f}s] {phase_str} TRADE: {event.actor} [{event.details.get('killer_team', '?')}] -> {event.target} [{event.details.get('victim_team', '?')}] {hs}")
        else:
            hs = '(HS)' if event.details.get('headshot') else ''
            print(f"[{event.time_ms/1000:5.1f}s] {phase_str} KILL:  {event.actor} [{event.details.get('killer_team', '?')}] -> {event.target} [{event.details.get('victim_team', '?')}] {hs}")

    print()
    print(f"Result: {result['winner']} wins by {result['win_condition']}")
    print(f"Score: {result['attack_alive']}v{result['defense_alive']} | Duration: {result['duration']/1000:.1f}s | Trades: {result['trades']}")
    print()

    # Run 50 rounds for stats
    print('='*90)
    print('50 ROUND ANALYSIS')
    print('='*90)

    results = []
    for _ in range(50):
        r = await sim.run_round()
        results.append(r)

    # Aggregate stats
    c9_wins = sum(1 for r in results if r['winner'] == 'C9')
    avg_duration = sum(r['duration'] for r in results) / len(results)
    avg_trades = sum(r['trades'] for r in results) / len(results)

    all_kills = []
    for r in results:
        all_kills.extend(r['kill_times'])

    total_kills = len(all_kills)
    avg_kills = total_kills / len(results)

    early = sum(1 for t in all_kills if t < 20000) / max(1, total_kills)
    mid = sum(1 for t in all_kills if 20000 <= t < 60000) / max(1, total_kills)
    late = sum(1 for t in all_kills if t >= 60000) / max(1, total_kills)

    # Score distribution
    wipes = sum(1 for r in results if r['attack_alive'] == 0 or r['defense_alive'] == 0)
    close = sum(1 for r in results if abs(r['attack_alive'] - r['defense_alive']) <= 1 and r['attack_alive'] > 0 and r['defense_alive'] > 0)

    plants = sum(1 for r in results if r['spike_planted'])

    print()
    print('SIMULATION RESULTS vs VCT TARGET')
    print('-'*50)
    print(f"{'Metric':<25} {'Ours':>10} {'VCT':>10}")
    print('-'*50)
    print(f"{'Attack Win Rate':<25} {c9_wins/50*100:>9.0f}% {47:>9}%")
    print(f"{'Avg Duration':<25} {avg_duration/1000:>9.1f}s {65:>9}s")
    print(f"{'Avg Kills/Round':<25} {avg_kills:>10.1f} {7.5:>10}")
    print(f"{'Avg Trades/Round':<25} {avg_trades:>10.1f} {1.9:>10}")
    print(f"{'Trade Rate':<25} {avg_trades/max(1,avg_kills)*100:>9.0f}% {25:>9}%")
    print(f"{'Plant Rate':<25} {plants/50*100:>9.0f}% {60:>9}%")
    print(f"{'Wipe Rate':<25} {wipes/50*100:>9.0f}% {18:>9}%")
    print()
    print('Kill Timing:')
    print(f"  Early (0-20s):   {early*100:5.0f}% (target: 15%)")
    print(f"  Mid (20-60s):    {mid*100:5.0f}% (target: 55%)")
    print(f"  Late (60s+):     {late*100:5.0f}% (target: 30%)")

    # Win condition breakdown
    print()
    print('Win Conditions:')
    detonations = sum(1 for r in results if r['win_condition'] == 'detonation')
    defuses = sum(1 for r in results if r['win_condition'] == 'defuse')
    eliminations_atk = sum(1 for r in results if r['win_condition'] == 'elimination' and r['winner'] == 'C9')
    eliminations_def = sum(1 for r in results if r['win_condition'] == 'elimination' and r['winner'] == 'LOUD')
    timeouts = sum(1 for r in results if 'timeout' in r['win_condition'])
    print(f"  Detonation (ATK):      {detonations:2} ({detonations*2}%)")
    print(f"  Elimination (ATK):     {eliminations_atk:2} ({eliminations_atk*2}%)")
    print(f"  Defuse (DEF):          {defuses:2} ({defuses*2}%)")
    print(f"  Elimination (DEF):     {eliminations_def:2} ({eliminations_def*2}%)")
    print(f"  Timeout (DEF):         {timeouts:2} ({timeouts*2}%)")


if __name__ == '__main__':
    asyncio.run(main())
