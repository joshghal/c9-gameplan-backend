"""Ability System for VALORANT tactical simulations.

Manages agent abilities, their effects, and timing patterns based on
observed data from professional matches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import random
import math


class AbilityCategory(Enum):
    """Categories of abilities by their tactical function."""
    SMOKE = "smoke"         # Vision blocking
    FLASH = "flash"         # Blinding enemies
    MOLLY = "molly"         # Area denial/damage
    RECON = "recon"         # Information gathering
    HEAL = "heal"           # Healing/revive
    MOVEMENT = "movement"   # Mobility abilities
    WALL = "wall"           # Physical barriers
    TRAP = "trap"           # Tripwires, alarms
    ULTIMATE = "ultimate"   # Ultimate abilities


@dataclass
class AbilityStats:
    """Statistics for an ability."""
    ability_id: str
    agent: str
    name: str
    category: AbilityCategory
    slot: str  # 'c', 'q', 'e', 'x' (ultimate)
    cost: int  # 0 for signature, ult points for ultimates
    max_charges: int = 1
    duration_ms: int = 0  # How long effect lasts
    cooldown_ms: int = 0  # Cooldown between uses (for signature)
    radius: float = 0.0   # Effect radius (normalized map units)

    # Effect parameters
    vision_block: bool = False
    flash_duration_ms: int = 0
    damage_per_second: float = 0.0
    heal_per_second: float = 0.0
    reveals_enemies: bool = False
    slows_enemies: bool = False

    # Repositioning parameters (for mollies, snake bite, etc.)
    forces_reposition: bool = False  # True if ability forces enemies to move
    unsafe_radius: float = 0.0       # Radius within which players should reposition


@dataclass
class ActiveAbility:
    """An ability that is currently active in the simulation."""
    ability: AbilityStats
    owner_id: str
    owner_team: str
    position: Tuple[float, float]
    start_time_ms: int
    end_time_ms: int
    affected_players: Set[str] = field(default_factory=set)


@dataclass
class PlayerAbilityState:
    """Tracks a player's ability usage state."""
    agent: str
    available_abilities: Dict[str, int]  # ability_id -> charges remaining
    cooldowns: Dict[str, int]  # ability_id -> cooldown end time
    ultimate_points: int = 0
    ultimate_ready: bool = False


class AbilityDatabase:
    """Database of all agent abilities."""

    ABILITIES: Dict[str, Dict[str, AbilityStats]] = {
        # DUELISTS
        'jett': {
            'cloudburst': AbilityStats(
                ability_id='jett_smoke', agent='jett', name='Cloudburst',
                category=AbilityCategory.SMOKE, slot='c', cost=200,
                max_charges=2, duration_ms=4500, radius=0.06, vision_block=True
            ),
            'updraft': AbilityStats(
                ability_id='jett_updraft', agent='jett', name='Updraft',
                category=AbilityCategory.MOVEMENT, slot='q', cost=150,
                max_charges=2, duration_ms=0
            ),
            'tailwind': AbilityStats(
                ability_id='jett_dash', agent='jett', name='Tailwind',
                category=AbilityCategory.MOVEMENT, slot='e', cost=0,
                max_charges=1, cooldown_ms=12000
            ),
            'blade_storm': AbilityStats(
                ability_id='jett_ult', agent='jett', name='Blade Storm',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                duration_ms=0
            ),
        },
        'reyna': {
            'leer': AbilityStats(
                ability_id='reyna_flash', agent='reyna', name='Leer',
                category=AbilityCategory.FLASH, slot='c', cost=250,
                max_charges=2, duration_ms=2000, flash_duration_ms=2000, radius=0.15
            ),
            'devour': AbilityStats(
                ability_id='reyna_heal', agent='reyna', name='Devour',
                category=AbilityCategory.HEAL, slot='q', cost=0,
                duration_ms=3000, heal_per_second=33.3
            ),
            'dismiss': AbilityStats(
                ability_id='reyna_dismiss', agent='reyna', name='Dismiss',
                category=AbilityCategory.MOVEMENT, slot='e', cost=0,
                duration_ms=2000
            ),
            'empress': AbilityStats(
                ability_id='reyna_ult', agent='reyna', name='Empress',
                category=AbilityCategory.ULTIMATE, slot='x', cost=6,
                duration_ms=30000
            ),
        },
        'phoenix': {
            'blaze': AbilityStats(
                ability_id='phoenix_wall', agent='phoenix', name='Blaze',
                category=AbilityCategory.WALL, slot='c', cost=200,
                duration_ms=8000, vision_block=True, damage_per_second=30
            ),
            'curveball': AbilityStats(
                ability_id='phoenix_flash', agent='phoenix', name='Curveball',
                category=AbilityCategory.FLASH, slot='q', cost=250,
                max_charges=2, flash_duration_ms=1100, radius=0.1
            ),
            'hot_hands': AbilityStats(
                ability_id='phoenix_molly', agent='phoenix', name='Hot Hands',
                category=AbilityCategory.MOLLY, slot='e', cost=0,
                duration_ms=4000, damage_per_second=60, radius=0.08, cooldown_ms=45000,
                forces_reposition=True, unsafe_radius=0.10
            ),
            'run_it_back': AbilityStats(
                ability_id='phoenix_ult', agent='phoenix', name='Run It Back',
                category=AbilityCategory.ULTIMATE, slot='x', cost=6,
                duration_ms=10000
            ),
        },
        'raze': {
            'boom_bot': AbilityStats(
                ability_id='raze_bot', agent='raze', name='Boom Bot',
                category=AbilityCategory.RECON, slot='c', cost=400,
                duration_ms=10000, reveals_enemies=True
            ),
            'blast_pack': AbilityStats(
                ability_id='raze_satchel', agent='raze', name='Blast Pack',
                category=AbilityCategory.MOVEMENT, slot='q', cost=200,
                max_charges=2, duration_ms=5000
            ),
            'paint_shells': AbilityStats(
                ability_id='raze_grenade', agent='raze', name='Paint Shells',
                category=AbilityCategory.MOLLY, slot='e', cost=0,
                damage_per_second=0, radius=0.12, cooldown_ms=45000,  # Instant damage
                forces_reposition=True, unsafe_radius=0.14
            ),
            'showstopper': AbilityStats(
                ability_id='raze_ult', agent='raze', name='Showstopper',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                radius=0.15
            ),
        },

        # CONTROLLERS
        'omen': {
            'shrouded_step': AbilityStats(
                ability_id='omen_tp', agent='omen', name='Shrouded Step',
                category=AbilityCategory.MOVEMENT, slot='c', cost=150,
                max_charges=2
            ),
            'paranoia': AbilityStats(
                ability_id='omen_flash', agent='omen', name='Paranoia',
                category=AbilityCategory.FLASH, slot='q', cost=300,
                flash_duration_ms=2200
            ),
            'dark_cover': AbilityStats(
                ability_id='omen_smoke', agent='omen', name='Dark Cover',
                category=AbilityCategory.SMOKE, slot='e', cost=0,
                max_charges=2, duration_ms=15000, radius=0.08, vision_block=True, cooldown_ms=40000
            ),
            'from_the_shadows': AbilityStats(
                ability_id='omen_ult', agent='omen', name='From the Shadows',
                category=AbilityCategory.ULTIMATE, slot='x', cost=7
            ),
        },
        'brimstone': {
            'stim_beacon': AbilityStats(
                ability_id='brim_stim', agent='brimstone', name='Stim Beacon',
                category=AbilityCategory.HEAL, slot='c', cost=200,
                duration_ms=12000, radius=0.1
            ),
            'incendiary': AbilityStats(
                ability_id='brim_molly', agent='brimstone', name='Incendiary',
                category=AbilityCategory.MOLLY, slot='q', cost=250,
                duration_ms=8000, damage_per_second=60, radius=0.1,
                forces_reposition=True, unsafe_radius=0.12
            ),
            'sky_smoke': AbilityStats(
                ability_id='brim_smoke', agent='brimstone', name='Sky Smoke',
                category=AbilityCategory.SMOKE, slot='e', cost=0,
                max_charges=3, duration_ms=19500, radius=0.08, vision_block=True
            ),
            'orbital_strike': AbilityStats(
                ability_id='brim_ult', agent='brimstone', name='Orbital Strike',
                category=AbilityCategory.ULTIMATE, slot='x', cost=7,
                duration_ms=3000, damage_per_second=60, radius=0.15,
                forces_reposition=True, unsafe_radius=0.18
            ),
        },
        'viper': {
            'snake_bite': AbilityStats(
                ability_id='viper_molly', agent='viper', name='Snake Bite',
                category=AbilityCategory.MOLLY, slot='c', cost=200,
                max_charges=2, duration_ms=8000, damage_per_second=25, radius=0.08,
                forces_reposition=True, unsafe_radius=0.10
            ),
            'poison_cloud': AbilityStats(
                ability_id='viper_smoke', agent='viper', name='Poison Cloud',
                category=AbilityCategory.SMOKE, slot='q', cost=200,
                duration_ms=15000, radius=0.1, vision_block=True
            ),
            'toxic_screen': AbilityStats(
                ability_id='viper_wall', agent='viper', name='Toxic Screen',
                category=AbilityCategory.WALL, slot='e', cost=0,
                duration_ms=15000, vision_block=True
            ),
            'vipers_pit': AbilityStats(
                ability_id='viper_ult', agent='viper', name="Viper's Pit",
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                duration_ms=0, radius=0.25, vision_block=True  # Lasts until viper leaves
            ),
        },
        'astra': {
            'gravity_well': AbilityStats(
                ability_id='astra_pull', agent='astra', name='Gravity Well',
                category=AbilityCategory.MOLLY, slot='c', cost=200,
                duration_ms=2750, slows_enemies=True, radius=0.1
            ),
            'nova_pulse': AbilityStats(
                ability_id='astra_stun', agent='astra', name='Nova Pulse',
                category=AbilityCategory.FLASH, slot='q', cost=250,
                duration_ms=1100, radius=0.1
            ),
            'nebula': AbilityStats(
                ability_id='astra_smoke', agent='astra', name='Nebula',
                category=AbilityCategory.SMOKE, slot='e', cost=0,
                duration_ms=15000, radius=0.08, vision_block=True
            ),
            'cosmic_divide': AbilityStats(
                ability_id='astra_ult', agent='astra', name='Cosmic Divide',
                category=AbilityCategory.ULTIMATE, slot='x', cost=7,
                duration_ms=21000, vision_block=True
            ),
        },

        # INITIATORS
        'sova': {
            'owl_drone': AbilityStats(
                ability_id='sova_drone', agent='sova', name='Owl Drone',
                category=AbilityCategory.RECON, slot='c', cost=400,
                duration_ms=10000, reveals_enemies=True
            ),
            'shock_bolt': AbilityStats(
                ability_id='sova_shock', agent='sova', name='Shock Bolt',
                category=AbilityCategory.MOLLY, slot='q', cost=150,
                max_charges=2, damage_per_second=0, radius=0.06  # Instant damage
            ),
            'recon_bolt': AbilityStats(
                ability_id='sova_recon', agent='sova', name='Recon Bolt',
                category=AbilityCategory.RECON, slot='e', cost=0,
                duration_ms=3000, radius=0.2, reveals_enemies=True, cooldown_ms=40000
            ),
            'hunters_fury': AbilityStats(
                ability_id='sova_ult', agent='sova', name="Hunter's Fury",
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                reveals_enemies=True
            ),
        },
        'breach': {
            'aftershock': AbilityStats(
                ability_id='breach_aftershock', agent='breach', name='Aftershock',
                category=AbilityCategory.MOLLY, slot='c', cost=200,
                damage_per_second=0, radius=0.06
            ),
            'flashpoint': AbilityStats(
                ability_id='breach_flash', agent='breach', name='Flashpoint',
                category=AbilityCategory.FLASH, slot='q', cost=250,
                max_charges=2, flash_duration_ms=2000, radius=0.15
            ),
            'fault_line': AbilityStats(
                ability_id='breach_stun', agent='breach', name='Fault Line',
                category=AbilityCategory.FLASH, slot='e', cost=0,
                duration_ms=3000, slows_enemies=True, cooldown_ms=40000
            ),
            'rolling_thunder': AbilityStats(
                ability_id='breach_ult', agent='breach', name='Rolling Thunder',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                flash_duration_ms=6000
            ),
        },
        'skye': {
            'regrowth': AbilityStats(
                ability_id='skye_heal', agent='skye', name='Regrowth',
                category=AbilityCategory.HEAL, slot='c', cost=200,
                duration_ms=5000, heal_per_second=20, radius=0.1
            ),
            'trailblazer': AbilityStats(
                ability_id='skye_dog', agent='skye', name='Trailblazer',
                category=AbilityCategory.RECON, slot='q', cost=300,
                duration_ms=6000, reveals_enemies=True
            ),
            'guiding_light': AbilityStats(
                ability_id='skye_flash', agent='skye', name='Guiding Light',
                category=AbilityCategory.FLASH, slot='e', cost=0,
                max_charges=2, flash_duration_ms=2250, cooldown_ms=40000
            ),
            'seekers': AbilityStats(
                ability_id='skye_ult', agent='skye', name='Seekers',
                category=AbilityCategory.ULTIMATE, slot='x', cost=7,
                reveals_enemies=True
            ),
        },
        'fade': {
            'prowler': AbilityStats(
                ability_id='fade_prowler', agent='fade', name='Prowler',
                category=AbilityCategory.RECON, slot='c', cost=300,
                max_charges=2, duration_ms=3000, reveals_enemies=True
            ),
            'seize': AbilityStats(
                ability_id='fade_seize', agent='fade', name='Seize',
                category=AbilityCategory.MOLLY, slot='q', cost=200,
                duration_ms=5000, slows_enemies=True, radius=0.08
            ),
            'haunt': AbilityStats(
                ability_id='fade_haunt', agent='fade', name='Haunt',
                category=AbilityCategory.RECON, slot='e', cost=0,
                duration_ms=2000, reveals_enemies=True, cooldown_ms=40000
            ),
            'nightfall': AbilityStats(
                ability_id='fade_ult', agent='fade', name='Nightfall',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                duration_ms=12000, slows_enemies=True, reveals_enemies=True
            ),
        },
        'kayo': {
            'frag': AbilityStats(
                ability_id='kayo_grenade', agent='kayo', name='FRAG/ment',
                category=AbilityCategory.MOLLY, slot='c', cost=200,
                duration_ms=4000, damage_per_second=60, radius=0.1,
                forces_reposition=True, unsafe_radius=0.12
            ),
            'flash_drive': AbilityStats(
                ability_id='kayo_flash', agent='kayo', name='FLASH/drive',
                category=AbilityCategory.FLASH, slot='q', cost=250,
                max_charges=2, flash_duration_ms=2200
            ),
            'zero_point': AbilityStats(
                ability_id='kayo_suppress', agent='kayo', name='ZERO/point',
                category=AbilityCategory.RECON, slot='e', cost=0,
                duration_ms=8000, radius=0.15, reveals_enemies=True, cooldown_ms=40000
            ),
            'null_cmd': AbilityStats(
                ability_id='kayo_ult', agent='kayo', name='NULL/cmd',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                duration_ms=15000, radius=0.25
            ),
        },

        # SENTINELS
        'sage': {
            'barrier_orb': AbilityStats(
                ability_id='sage_wall', agent='sage', name='Barrier Orb',
                category=AbilityCategory.WALL, slot='c', cost=400,
                duration_ms=40000
            ),
            'slow_orb': AbilityStats(
                ability_id='sage_slow', agent='sage', name='Slow Orb',
                category=AbilityCategory.MOLLY, slot='q', cost=200,
                max_charges=2, duration_ms=7000, slows_enemies=True, radius=0.12
            ),
            'healing_orb': AbilityStats(
                ability_id='sage_heal', agent='sage', name='Healing Orb',
                category=AbilityCategory.HEAL, slot='e', cost=0,
                heal_per_second=60, cooldown_ms=45000
            ),
            'resurrection': AbilityStats(
                ability_id='sage_ult', agent='sage', name='Resurrection',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8
            ),
        },
        'cypher': {
            'trapwire': AbilityStats(
                ability_id='cypher_trip', agent='cypher', name='Trapwire',
                category=AbilityCategory.TRAP, slot='c', cost=200,
                max_charges=2, reveals_enemies=True
            ),
            'cyber_cage': AbilityStats(
                ability_id='cypher_cage', agent='cypher', name='Cyber Cage',
                category=AbilityCategory.SMOKE, slot='q', cost=100,
                max_charges=2, duration_ms=7000, radius=0.06, vision_block=True
            ),
            'spycam': AbilityStats(
                ability_id='cypher_cam', agent='cypher', name='Spycam',
                category=AbilityCategory.RECON, slot='e', cost=0,
                reveals_enemies=True, cooldown_ms=45000
            ),
            'neural_theft': AbilityStats(
                ability_id='cypher_ult', agent='cypher', name='Neural Theft',
                category=AbilityCategory.ULTIMATE, slot='x', cost=6,
                reveals_enemies=True
            ),
        },
        'killjoy': {
            'nanoswarm': AbilityStats(
                ability_id='kj_swarm', agent='killjoy', name='Nanoswarm',
                category=AbilityCategory.MOLLY, slot='c', cost=200,
                max_charges=2, duration_ms=4500, damage_per_second=45, radius=0.08,
                forces_reposition=True, unsafe_radius=0.10
            ),
            'alarmbot': AbilityStats(
                ability_id='kj_bot', agent='killjoy', name='Alarmbot',
                category=AbilityCategory.TRAP, slot='q', cost=200,
                reveals_enemies=True
            ),
            'turret': AbilityStats(
                ability_id='kj_turret', agent='killjoy', name='Turret',
                category=AbilityCategory.TRAP, slot='e', cost=0,
                reveals_enemies=True, cooldown_ms=45000
            ),
            'lockdown': AbilityStats(
                ability_id='kj_ult', agent='killjoy', name='Lockdown',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                duration_ms=13000, slows_enemies=True, radius=0.35
            ),
        },
        'chamber': {
            'trademark': AbilityStats(
                ability_id='chamber_trip', agent='chamber', name='Trademark',
                category=AbilityCategory.TRAP, slot='c', cost=200,
                reveals_enemies=True, slows_enemies=True
            ),
            'headhunter': AbilityStats(
                ability_id='chamber_pistol', agent='chamber', name='Headhunter',
                category=AbilityCategory.ULTIMATE, slot='q', cost=150,
                max_charges=8
            ),
            'rendezvous': AbilityStats(
                ability_id='chamber_tp', agent='chamber', name='Rendezvous',
                category=AbilityCategory.MOVEMENT, slot='e', cost=0,
                cooldown_ms=20000
            ),
            'tour_de_force': AbilityStats(
                ability_id='chamber_ult', agent='chamber', name='Tour De Force',
                category=AbilityCategory.ULTIMATE, slot='x', cost=8,
                duration_ms=0
            ),
        },
    }

    @classmethod
    def get_agent_abilities(cls, agent: str) -> Dict[str, AbilityStats]:
        """Get all abilities for an agent."""
        return cls.ABILITIES.get(agent.lower(), {})

    @classmethod
    def get_ability(cls, agent: str, ability_id: str) -> Optional[AbilityStats]:
        """Get a specific ability."""
        agent_abilities = cls.ABILITIES.get(agent.lower(), {})
        return agent_abilities.get(ability_id)


class AbilitySystem:
    """Manages ability usage and effects during simulation."""

    # P3 FIX: Enhanced timing patterns from GRID data analysis (milliseconds from round start)
    # Based on analysis of 2,297 ability events from pro matches
    ABILITY_TIMING_PATTERNS = {
        'opening': {
            AbilityCategory.SMOKE: [(3000, 8000), (5000, 12000), (8000, 15000)],
            AbilityCategory.FLASH: [(15000, 25000), (18000, 28000)],
            AbilityCategory.RECON: [(5000, 15000), (10000, 20000)],
            AbilityCategory.WALL: [(2000, 8000)],
            AbilityCategory.TRAP: [(1000, 5000)],  # Sentinels set up early
        },
        'mid_round': {
            AbilityCategory.SMOKE: [(25000, 40000), (35000, 50000)],
            AbilityCategory.FLASH: [(30000, 50000), (40000, 60000)],
            AbilityCategory.MOLLY: [(30000, 45000)],
            AbilityCategory.RECON: [(25000, 40000)],
        },
        'post_plant': {
            AbilityCategory.MOLLY: [(0, 5000), (5000, 15000), (10000, 25000)],
            AbilityCategory.SMOKE: [(0, 8000), (10000, 20000)],
            AbilityCategory.FLASH: [(5000, 20000)],
        },
        'retake': {
            AbilityCategory.FLASH: [(0, 5000), (5000, 15000)],
            AbilityCategory.RECON: [(0, 8000)],
            AbilityCategory.SMOKE: [(5000, 15000)],
            AbilityCategory.MOLLY: [(10000, 25000)],  # Clear plant spot
        },
    }

    # P3 FIX: Map-specific ability timing adjustments (GRID analysis)
    # Some maps have faster/slower executes based on layout
    MAP_TIMING_MODIFIERS = {
        'ascent': 1.0,    # Baseline timing
        'bind': 0.9,      # Faster rotates, earlier abilities
        'split': 1.1,     # Slower mid control battles
        'icebox': 1.0,    # Standard timing
        'breeze': 1.2,    # Larger map, slower pacing
        'fracture': 0.85, # Fast-paced with dual spawns
        'pearl': 1.0,     # Standard timing
        'sunset': 0.95,   # Slightly faster pace
        'abyss': 1.1,     # Slower, more methodical
        'haven': 1.05,    # 3 sites = more spreading out
        'lotus': 0.9,     # Fast rotates with doors
    }

    # P3 FIX: Agent-specific ability usage rates (from GRID analysis)
    # How likely each agent is to use abilities per phase
    AGENT_ABILITY_USAGE_RATES = {
        # Controllers - high smoke usage
        'omen': {'smoke': 0.9, 'flash': 0.7, 'recon': 0.0},
        'brimstone': {'smoke': 0.95, 'molly': 0.6, 'recon': 0.0},
        'viper': {'smoke': 0.85, 'molly': 0.7, 'wall': 0.6},
        'astra': {'smoke': 0.9, 'stun': 0.5, 'recon': 0.0},
        'harbor': {'smoke': 0.85, 'wall': 0.7, 'slow': 0.5},
        'clove': {'smoke': 0.85, 'heal': 0.4, 'recon': 0.0},
        # Initiators - high info usage
        'sova': {'recon': 0.9, 'flash': 0.0, 'shock': 0.4},
        'breach': {'flash': 0.85, 'stun': 0.6, 'recon': 0.0},
        'skye': {'flash': 0.8, 'recon': 0.7, 'heal': 0.5},
        'kayo': {'flash': 0.85, 'recon': 0.6, 'suppress': 0.4},
        'fade': {'recon': 0.85, 'flash': 0.0, 'slow': 0.6},
        'gekko': {'flash': 0.75, 'recon': 0.7, 'molly': 0.5},
        # Duelists - aggressive ability usage
        'jett': {'smoke': 0.6, 'dash': 0.8, 'updraft': 0.5},
        'raze': {'molly': 0.7, 'flash': 0.0, 'satchel': 0.8},
        'phoenix': {'flash': 0.85, 'molly': 0.5, 'wall': 0.4},
        'reyna': {'flash': 0.9, 'heal': 0.7, 'recon': 0.0},
        'yoru': {'flash': 0.8, 'teleport': 0.6, 'decoy': 0.3},
        'neon': {'wall': 0.5, 'stun': 0.6, 'dash': 0.85},
        'iso': {'shield': 0.7, 'wall': 0.5, 'recon': 0.0},
        # Sentinels - setup-focused
        'killjoy': {'trap': 0.95, 'molly': 0.7, 'turret': 0.9},
        'cypher': {'trap': 0.95, 'recon': 0.8, 'cage': 0.6},
        'sage': {'wall': 0.8, 'slow': 0.7, 'heal': 0.9},
        'chamber': {'trap': 0.85, 'teleport': 0.6, 'recon': 0.0},
        'deadlock': {'trap': 0.9, 'wall': 0.6, 'recon': 0.0},
        'vyse': {'trap': 0.9, 'wall': 0.7, 'recon': 0.0},
    }

    # P2 FIX: Execute timing - smokes should precede site entry by 2-3 seconds
    # Common smoke positions for site executes per map
    EXECUTE_SMOKE_POSITIONS = {
        'A': [
            (0.28, 0.35),  # A main entrance smoke
            (0.22, 0.28),  # A site cross smoke
            (0.35, 0.25),  # A heaven/high ground smoke
        ],
        'B': [
            (0.72, 0.35),  # B main entrance smoke
            (0.78, 0.28),  # B site cross smoke
            (0.65, 0.25),  # B heaven/high ground smoke
        ],
        'C': [  # For 3-site maps (Haven, Lotus)
            (0.85, 0.32),  # C main entrance
            (0.82, 0.25),  # C site smoke
        ],
    }
    SMOKE_BEFORE_ENTRY_MS = 2500  # Deploy smokes 2.5 seconds before entry

    def __init__(self):
        self.active_abilities: List[ActiveAbility] = []
        self.player_states: Dict[str, PlayerAbilityState] = {}
        # P2 FIX: Execute timing tracking
        self.execute_called: bool = False
        self.execute_site: Optional[str] = None
        self.execute_time_ms: int = 0

    def initialize_player(self, player_id: str, agent: str):
        """Initialize ability state for a player."""
        abilities = AbilityDatabase.get_agent_abilities(agent)

        available = {}
        for ability_id, ability in abilities.items():
            if ability.category != AbilityCategory.ULTIMATE:
                available[ability_id] = ability.max_charges

        self.player_states[player_id] = PlayerAbilityState(
            agent=agent,
            available_abilities=available,
            cooldowns={},
            ultimate_points=0,
            ultimate_ready=False
        )

    def call_execute(self, site: str, time_ms: int):
        """P2 FIX: Signal that attackers are executing onto a site.

        This triggers coordinated smoke timing - smokes should deploy
        approximately SMOKE_BEFORE_ENTRY_MS before the execute call.
        """
        self.execute_called = True
        self.execute_site = site.upper()
        self.execute_time_ms = time_ms

    def reset_execute(self):
        """Reset execute state (e.g., after plant or round end)."""
        self.execute_called = False
        self.execute_site = None
        self.execute_time_ms = 0

    def check_execute_smoke_timing(
        self,
        player_id: str,
        time_ms: int,
        player_position: Tuple[float, float],
        site_positions: Dict[str, Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """P2 FIX: Check if a smoke should be deployed for execute timing.

        Smokes are deployed when attackers are close to a site and preparing to execute.
        Returns the smoke target position if smoke should be used, None otherwise.
        """
        state = self.player_states.get(player_id)
        if not state:
            return None

        # Check if player has smoke ability with charges
        abilities = AbilityDatabase.get_agent_abilities(state.agent)
        smoke_ability = None
        for ability_id, charges in state.available_abilities.items():
            if charges <= 0:
                continue
            ability = abilities.get(ability_id)
            if ability and ability.category == AbilityCategory.SMOKE:
                smoke_ability = ability
                break

        if not smoke_ability:
            return None

        # Determine which site attackers are approaching
        target_site = None
        min_dist_to_site = float('inf')

        for site_name, site_pos in site_positions.items():
            dist = math.sqrt(
                (player_position[0] - site_pos[0]) ** 2 +
                (player_position[1] - site_pos[1]) ** 2
            )
            if dist < min_dist_to_site:
                min_dist_to_site = dist
                target_site = site_name

        # If close enough to a site (within 25% of map), consider smoking
        if min_dist_to_site > 0.25:
            return None

        # Get execute smoke positions for this site
        smoke_positions = self.EXECUTE_SMOKE_POSITIONS.get(target_site, [])
        if not smoke_positions:
            return None

        # 20% chance per tick to deploy execute smoke when in range
        if random.random() < 0.20:
            # Pick a smoke position (prioritize first ones - main chokes)
            if random.random() < 0.6:
                target = smoke_positions[0]  # Main entrance
            else:
                target = random.choice(smoke_positions)

            # Add slight variance
            return (
                target[0] + random.uniform(-0.02, 0.02),
                target[1] + random.uniform(-0.02, 0.02)
            )

        return None

    def should_use_ability(
        self,
        player_id: str,
        time_ms: int,
        phase: str,
        round_state: 'RoundState',
        player_position: Tuple[float, float],
        team: str,
        attacker_positions: Optional[List[Tuple[float, float]]] = None
    ) -> Optional[Tuple[AbilityStats, Tuple[float, float]]]:
        """Determine if a player should use an ability.

        Context-aware ability usage based on phase and game state:
        - Retake: Smokes block attacker sightlines, mollies target spike
        - Post-plant attackers: Delay utility, save for retake denial
        - Early round: Setup and info gathering

        Args:
            player_id: Player to check
            time_ms: Current round time
            phase: Current game phase
            round_state: Current round state
            player_position: Player's position
            team: 'attack' or 'defense'
            attacker_positions: Known attacker positions (for smart targeting)

        Returns:
            Tuple of (ability to use, target position) or None
        """
        state = self.player_states.get(player_id)
        if not state:
            return None

        abilities = AbilityDatabase.get_agent_abilities(state.agent)

        # Context-aware ability usage for retakes
        if phase == 'retake' and team == 'defense':
            return self._check_retake_ability(
                state, abilities, time_ms, round_state,
                player_position, attacker_positions
            )

        # P2 FIX: Check for execute smoke timing (attackers approaching site)
        if team == 'attack' and phase in ('opening', 'mid_round'):
            # Use generic site positions - these will be overridden by map data if available
            default_sites = {
                'A': (0.3, 0.3),
                'B': (0.7, 0.3),
                'C': (0.85, 0.25),  # For 3-site maps
            }
            execute_smoke_pos = self.check_execute_smoke_timing(
                player_id, time_ms, player_position, default_sites
            )
            if execute_smoke_pos:
                # Find the smoke ability to use
                for ability_id, charges in state.available_abilities.items():
                    if charges <= 0:
                        continue
                    ability = abilities.get(ability_id)
                    if ability and ability.category == AbilityCategory.SMOKE:
                        return (ability, execute_smoke_pos)

        # Get timing patterns for current phase
        patterns = self.ABILITY_TIMING_PATTERNS.get(phase, {})

        # Check each available ability
        for ability_id, charges in state.available_abilities.items():
            if charges <= 0:
                continue

            # Check cooldown
            cooldown_end = state.cooldowns.get(ability_id, 0)
            if time_ms < cooldown_end:
                continue

            ability = abilities.get(ability_id)
            if not ability:
                continue

            # Check if timing is appropriate
            timing_windows = patterns.get(ability.category, [])
            should_use = False

            for window_start, window_end in timing_windows:
                if window_start <= time_ms <= window_end:
                    # Random chance to use (prevents all abilities at exact times)
                    if random.random() < 0.15:  # 15% chance per tick in window
                        should_use = True
                        break

            if not should_use:
                continue

            # Determine target position based on ability type
            target_pos = self._get_ability_target(
                ability, player_position, team, round_state
            )

            return (ability, target_pos)

        return None

    def _check_retake_ability(
        self,
        state: 'PlayerAbilityState',
        abilities: Dict[str, AbilityStats],
        time_ms: int,
        round_state: 'RoundState',
        player_position: Tuple[float, float],
        attacker_positions: Optional[List[Tuple[float, float]]]
    ) -> Optional[Tuple[AbilityStats, Tuple[float, float]]]:
        """Context-aware ability usage for retake scenarios.

        Retake utility flow:
        1. Recon (0-8s): Gather info on attacker positions
        2. Smoke (5-15s): Block attacker sightlines to entry points
        3. Flash (0-15s): Create entry window
        4. Molly (10-25s): Force attackers off spike

        Args:
            state: Player's ability state
            abilities: Available abilities
            time_ms: Current round time
            round_state: Round state with spike info
            player_position: Player's position
            attacker_positions: Known attacker positions

        Returns:
            Tuple of (ability, target) or None
        """
        if not round_state.spike_planted or not round_state.spike_site:
            return None

        # Get spike position based on site
        spike_positions = {
            'A': (0.3, 0.3),
            'B': (0.7, 0.3),
            'C': (0.85, 0.25),
        }
        spike_pos = spike_positions.get(round_state.spike_site, (0.5, 0.3))

        # Calculate time since plant for phase
        plant_elapsed = time_ms - (round_state.spike_plant_time_ms if hasattr(round_state, 'spike_plant_time_ms') else 0)

        for ability_id, charges in state.available_abilities.items():
            if charges <= 0:
                continue

            cooldown_end = state.cooldowns.get(ability_id, 0)
            if time_ms < cooldown_end:
                continue

            ability = abilities.get(ability_id)
            if not ability:
                continue

            # Recon first (0-8s): Gather info
            if ability.category == AbilityCategory.RECON:
                if 0 <= plant_elapsed <= 8000:
                    if random.random() < 0.25:  # 25% chance per tick
                        # Target the spike site to find attackers
                        target = (
                            spike_pos[0] + random.uniform(-0.05, 0.05),
                            spike_pos[1] + random.uniform(-0.05, 0.05)
                        )
                        return (ability, target)

            # Smoke to block attacker sightlines (5-15s)
            if ability.category == AbilityCategory.SMOKE:
                if 5000 <= plant_elapsed <= 15000:
                    if random.random() < 0.20:  # 20% chance per tick
                        # If we know attacker positions, smoke between them and us
                        if attacker_positions:
                            # Find average attacker position and smoke between
                            avg_x = sum(p[0] for p in attacker_positions) / len(attacker_positions)
                            avg_y = sum(p[1] for p in attacker_positions) / len(attacker_positions)
                            target = (
                                (avg_x + player_position[0]) / 2 + random.uniform(-0.02, 0.02),
                                (avg_y + player_position[1]) / 2 + random.uniform(-0.02, 0.02)
                            )
                        else:
                            # Default: smoke common entry choke
                            target = (
                                spike_pos[0] + 0.1 + random.uniform(-0.02, 0.02),
                                spike_pos[1] + random.uniform(-0.05, 0.05)
                            )
                        return (ability, target)

            # Flash for entry (0-15s)
            if ability.category == AbilityCategory.FLASH:
                if 0 <= plant_elapsed <= 15000:
                    if random.random() < 0.15:  # 15% chance per tick
                        # Flash toward site
                        target = (
                            spike_pos[0] + random.uniform(-0.05, 0.05),
                            spike_pos[1] + random.uniform(-0.05, 0.05)
                        )
                        return (ability, target)

            # Molly to clear spike (10-25s) - HIGH PRIORITY
            if ability.category == AbilityCategory.MOLLY:
                if 10000 <= plant_elapsed <= 25000:
                    if random.random() < 0.30:  # 30% chance - high priority for retake
                        # Target spike directly
                        target = (
                            spike_pos[0] + random.uniform(-0.03, 0.03),
                            spike_pos[1] + random.uniform(-0.03, 0.03)
                        )
                        return (ability, target)

        return None

    def _get_ability_target(
        self,
        ability: AbilityStats,
        player_position: Tuple[float, float],
        team: str,
        round_state: 'RoundState'
    ) -> Tuple[float, float]:
        """Get target position for an ability."""
        # Default: slightly ahead of player
        if team == 'attack':
            # Target toward defense side
            target = (
                player_position[0] + random.uniform(0.05, 0.15),
                player_position[1] - random.uniform(0.0, 0.1)
            )
        else:
            # Target toward attack side
            target = (
                player_position[0] - random.uniform(0.05, 0.15),
                player_position[1] + random.uniform(0.0, 0.1)
            )

        # Smokes target common spots
        if ability.category == AbilityCategory.SMOKE:
            common_smoke_spots = [
                (0.3, 0.35), (0.35, 0.3),  # A site entrances
                (0.65, 0.35), (0.7, 0.3),  # B site entrances
                (0.5, 0.45), (0.5, 0.5),   # Mid
            ]
            if random.random() < 0.7:  # 70% chance to use common spot
                target = random.choice(common_smoke_spots)
                # Add slight variance
                target = (
                    target[0] + random.uniform(-0.02, 0.02),
                    target[1] + random.uniform(-0.02, 0.02)
                )

        # Post-plant mollies target spike
        if round_state.spike_planted and ability.category == AbilityCategory.MOLLY:
            if round_state.spike_site == 'A':
                target = (0.3 + random.uniform(-0.03, 0.03), 0.3 + random.uniform(-0.03, 0.03))
            else:
                target = (0.7 + random.uniform(-0.03, 0.03), 0.3 + random.uniform(-0.03, 0.03))

        return target

    def use_ability(
        self,
        player_id: str,
        ability: AbilityStats,
        position: Tuple[float, float],
        time_ms: int,
        team: str
    ) -> ActiveAbility:
        """Use an ability and create its active effect."""
        state = self.player_states.get(player_id)
        if state:
            # Decrease charges
            if ability.ability_id in state.available_abilities:
                state.available_abilities[ability.ability_id] -= 1

            # Set cooldown
            if ability.cooldown_ms > 0:
                state.cooldowns[ability.ability_id] = time_ms + ability.cooldown_ms

        active = ActiveAbility(
            ability=ability,
            owner_id=player_id,
            owner_team=team,
            position=position,
            start_time_ms=time_ms,
            end_time_ms=time_ms + ability.duration_ms
        )

        self.active_abilities.append(active)
        return active

    def update_effects(
        self,
        time_ms: int,
        players: Dict[str, 'SimulatedPlayer']
    ) -> Dict[str, Dict[str, any]]:
        """Update ability effects and return effects applied to players.

        Args:
            time_ms: Current time
            players: Dict of player_id -> SimulatedPlayer

        Returns:
            Dict of player_id -> effects applied
        """
        effects = {}

        # Remove expired abilities
        self.active_abilities = [
            a for a in self.active_abilities
            if time_ms < a.end_time_ms
        ]

        # Apply active ability effects
        for active in self.active_abilities:
            ability = active.ability

            for player_id, player in players.items():
                if not player.is_alive:
                    continue

                # Calculate distance to ability
                dx = player.x - active.position[0]
                dy = player.y - active.position[1]
                distance = math.sqrt(dx * dx + dy * dy)

                # Check if in radius
                if distance > ability.radius:
                    continue

                # Initialize effects dict for player
                if player_id not in effects:
                    effects[player_id] = {
                        'is_flashed': False,
                        'flash_end_ms': 0,
                        'is_slowed': False,
                        'should_reposition': False,  # Molly forces repositioning
                        'reposition_away_from': None,  # Position to move away from
                        'in_smoke': False,
                        'revealed': False,
                        'damage_taken': 0,
                        'healing': 0,
                    }

                # Apply effects based on ability type
                if ability.flash_duration_ms > 0:
                    # Flash enemies only
                    if player.team_id != active.owner_team:
                        effects[player_id]['is_flashed'] = True
                        effects[player_id]['flash_end_ms'] = time_ms + ability.flash_duration_ms

                if ability.vision_block:
                    effects[player_id]['in_smoke'] = True

                if ability.slows_enemies and player.team_id != active.owner_team:
                    effects[player_id]['is_slowed'] = True

                if ability.reveals_enemies and player.team_id != active.owner_team:
                    effects[player_id]['revealed'] = True

                if ability.damage_per_second > 0 and player.team_id != active.owner_team:
                    # Damage per tick (128ms)
                    damage = ability.damage_per_second * 0.128
                    effects[player_id]['damage_taken'] += damage

                if ability.heal_per_second > 0 and player.team_id == active.owner_team:
                    heal = ability.heal_per_second * 0.128
                    effects[player_id]['healing'] += heal

                # Check for repositioning effects (mollies, snake bite, etc.)
                if ability.forces_reposition and player.team_id != active.owner_team:
                    # Only force reposition if within unsafe radius
                    if distance <= ability.unsafe_radius:
                        effects[player_id]['should_reposition'] = True
                        effects[player_id]['reposition_away_from'] = active.position

        return effects

    def find_safe_position(
        self,
        current_pos: Tuple[float, float],
        danger_pos: Tuple[float, float],
        unsafe_radius: float,
        min_safe_distance: float = 0.15
    ) -> Tuple[float, float]:
        """Find a safe position away from a danger zone (molly, etc.).

        Args:
            current_pos: Player's current position
            danger_pos: Center of the danger zone
            unsafe_radius: Radius of the danger zone
            min_safe_distance: Minimum distance to move away

        Returns:
            Safe position tuple (x, y)
        """
        # Calculate direction away from danger
        dx = current_pos[0] - danger_pos[0]
        dy = current_pos[1] - danger_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            # Directly on danger, pick random direction
            angle = random.random() * 2 * math.pi
            dx = math.cos(angle)
            dy = math.sin(angle)
            distance = 1.0

        # Normalize direction
        dx /= distance
        dy /= distance

        # Move to safe distance outside the danger zone
        safe_distance = unsafe_radius + min_safe_distance + random.uniform(0, 0.05)
        safe_x = danger_pos[0] + dx * safe_distance
        safe_y = danger_pos[1] + dy * safe_distance

        # Clamp to valid map bounds
        safe_x = max(0.05, min(0.95, safe_x))
        safe_y = max(0.05, min(0.95, safe_y))

        return (safe_x, safe_y)

    def is_position_smoked(
        self,
        position: Tuple[float, float],
        time_ms: int
    ) -> bool:
        """Check if a position is inside an active smoke."""
        for active in self.active_abilities:
            if not active.ability.vision_block:
                continue
            if time_ms >= active.end_time_ms:
                continue

            dx = position[0] - active.position[0]
            dy = position[1] - active.position[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= active.ability.radius:
                return True

        return False

    def get_active_smokes(self, time_ms: int) -> List[Tuple[Tuple[float, float], float]]:
        """Get list of active smoke positions and radii."""
        smokes = []
        for active in self.active_abilities:
            if active.ability.vision_block and time_ms < active.end_time_ms:
                smokes.append((active.position, active.ability.radius))
        return smokes

    def get_active_smoke_positions(self, time_ms: int) -> List[Tuple[Tuple[float, float], float]]:
        """Alias for get_active_smokes - returns list of (position, radius) tuples."""
        return self.get_active_smokes(time_ms)
