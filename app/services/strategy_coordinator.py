"""Strategy Coordinator for VALORANT tactical simulations.

Manages team strategies, role assignments, and coordinated movements.
Adds variability to prevent predictable patterns like "player X always goes to site A at time Y".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import random
import math


class Role(Enum):
    """Player roles within a strategy."""
    ENTRY = "entry"       # First in, takes fights
    SUPPORT = "support"   # Flashes/utility for entry
    LURK = "lurk"         # Flanks, plays off-angles
    ANCHOR = "anchor"     # Holds site, passive angles
    FLEX = "flex"         # Adapts based on situation
    IGL = "igl"           # In-game leader, calls plays
    OP = "op"             # Operator player


class StrategyPhase(Enum):
    """Phases of strategy execution."""
    SETUP = "setup"           # Initial positioning
    DEFAULT = "default"       # Default positions, gathering info
    EXECUTE = "execute"       # Committing to a site
    ROTATE = "rotate"         # Rotating to different site
    POST_PLANT = "post_plant" # After spike plant
    RETAKE = "retake"         # Retaking a site


@dataclass
class StrategyWaypoint:
    """A waypoint in a strategy execution."""
    position: Tuple[float, float]
    time_window_ms: Tuple[int, int]  # (min_time, max_time) to reach
    is_optional: bool = False
    requires_clear: bool = False  # Wait until area is cleared
    ability_trigger: Optional[str] = None  # Ability to use here


@dataclass
class RoleAssignment:
    """Assignment of a role to a player with specific waypoints."""
    player_id: str
    role: Role
    waypoints: List[StrategyWaypoint]
    target_site: Optional[str] = None


@dataclass
class Strategy:
    """A complete team strategy."""
    strategy_id: str
    name: str
    side: str  # 'attack' or 'defense'
    description: str

    # Role requirements
    required_roles: List[Role]
    preferred_agent_roles: Dict[Role, List[str]]  # Role -> agent types

    # Timing
    default_phase: StrategyPhase = StrategyPhase.DEFAULT
    execute_time_window_ms: Tuple[int, int] = (25000, 45000)

    # Conditions
    min_economy: int = 0  # Minimum team economy to run this strat
    rotation_triggers: List[str] = field(default_factory=list)  # Conditions to rotate

    # Site preferences (for attack)
    primary_site: Optional[str] = None
    secondary_site: Optional[str] = None

    # Variation parameters
    timing_variance_ms: int = 5000  # Random variance in execution timing
    position_variance: float = 0.05  # Random variance in positions (map %)


class StrategyDatabase:
    """Database of available strategies."""

    ATTACK_STRATEGIES: Dict[str, Strategy] = {
        'default': Strategy(
            strategy_id='atk_default',
            name='Default',
            side='attack',
            description='Standard default, gather info then execute',
            required_roles=[Role.ENTRY, Role.SUPPORT, Role.LURK, Role.FLEX, Role.IGL],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['initiator', 'controller'],
                Role.LURK: ['sentinel', 'duelist'],
                Role.FLEX: ['initiator', 'sentinel'],
                Role.IGL: ['controller', 'sentinel'],
            },
            execute_time_window_ms=(30000, 50000),
            timing_variance_ms=8000,
        ),
        'a_execute': Strategy(
            strategy_id='atk_a_exec',
            name='A Execute',
            side='attack',
            description='Fast A site execute with utility',
            required_roles=[Role.ENTRY, Role.SUPPORT, Role.SUPPORT, Role.FLEX, Role.LURK],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['initiator', 'controller'],
                Role.LURK: ['sentinel'],
                Role.FLEX: ['initiator'],
            },
            execute_time_window_ms=(20000, 35000),
            primary_site='A',
            timing_variance_ms=5000,
        ),
        'b_execute': Strategy(
            strategy_id='atk_b_exec',
            name='B Execute',
            side='attack',
            description='Fast B site execute with utility',
            required_roles=[Role.ENTRY, Role.SUPPORT, Role.SUPPORT, Role.FLEX, Role.LURK],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['initiator', 'controller'],
                Role.LURK: ['sentinel'],
                Role.FLEX: ['initiator'],
            },
            execute_time_window_ms=(20000, 35000),
            primary_site='B',
            timing_variance_ms=5000,
        ),
        'split': Strategy(
            strategy_id='atk_split',
            name='Split',
            side='attack',
            description='Split attack from multiple angles',
            required_roles=[Role.ENTRY, Role.ENTRY, Role.SUPPORT, Role.LURK, Role.FLEX],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['controller'],
                Role.LURK: ['sentinel', 'initiator'],
                Role.FLEX: ['initiator'],
            },
            execute_time_window_ms=(35000, 55000),
            timing_variance_ms=6000,
        ),
        'slow_default': Strategy(
            strategy_id='atk_slow',
            name='Slow Default',
            side='attack',
            description='Slow methodical approach, heavy info gathering',
            required_roles=[Role.ENTRY, Role.SUPPORT, Role.LURK, Role.ANCHOR, Role.IGL],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['initiator'],
                Role.LURK: ['sentinel'],
                Role.ANCHOR: ['controller'],
                Role.IGL: ['controller'],
            },
            execute_time_window_ms=(50000, 75000),
            timing_variance_ms=10000,
        ),
        'rush': Strategy(
            strategy_id='atk_rush',
            name='Rush',
            side='attack',
            description='Fast rush to overwhelm defenders',
            required_roles=[Role.ENTRY, Role.ENTRY, Role.SUPPORT, Role.SUPPORT, Role.FLEX],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['initiator', 'controller'],
                Role.FLEX: ['duelist', 'initiator'],
            },
            execute_time_window_ms=(5000, 15000),
            min_economy=0,  # Good for ecos
            timing_variance_ms=3000,
        ),
    }

    DEFENSE_STRATEGIES: Dict[str, Strategy] = {
        'default': Strategy(
            strategy_id='def_default',
            name='Default',
            side='defense',
            description='Standard 2-1-2 or 2-2-1 setup',
            required_roles=[Role.ANCHOR, Role.ANCHOR, Role.FLEX, Role.LURK, Role.OP],
            preferred_agent_roles={
                Role.ANCHOR: ['sentinel', 'controller'],
                Role.FLEX: ['initiator'],
                Role.LURK: ['duelist', 'initiator'],
                Role.OP: ['sentinel', 'duelist'],
            },
            timing_variance_ms=3000,
            position_variance=0.08,
        ),
        'stack_a': Strategy(
            strategy_id='def_stack_a',
            name='Stack A',
            side='defense',
            description='Heavy presence on A site',
            required_roles=[Role.ANCHOR, Role.ANCHOR, Role.ANCHOR, Role.FLEX, Role.LURK],
            preferred_agent_roles={
                Role.ANCHOR: ['sentinel', 'controller'],
                Role.FLEX: ['initiator'],
                Role.LURK: ['duelist'],
            },
            primary_site='A',
            timing_variance_ms=2000,
        ),
        'stack_b': Strategy(
            strategy_id='def_stack_b',
            name='Stack B',
            side='defense',
            description='Heavy presence on B site',
            required_roles=[Role.ANCHOR, Role.ANCHOR, Role.ANCHOR, Role.FLEX, Role.LURK],
            preferred_agent_roles={
                Role.ANCHOR: ['sentinel', 'controller'],
                Role.FLEX: ['initiator'],
                Role.LURK: ['duelist'],
            },
            primary_site='B',
            timing_variance_ms=2000,
        ),
        'aggro_mid': Strategy(
            strategy_id='def_aggro',
            name='Aggressive Mid',
            side='defense',
            description='Push for information and early picks',
            required_roles=[Role.ENTRY, Role.SUPPORT, Role.ANCHOR, Role.ANCHOR, Role.LURK],
            preferred_agent_roles={
                Role.ENTRY: ['duelist'],
                Role.SUPPORT: ['initiator'],
                Role.ANCHOR: ['sentinel', 'controller'],
                Role.LURK: ['initiator'],
            },
            timing_variance_ms=4000,
        ),
        'retake': Strategy(
            strategy_id='def_retake',
            name='Retake Setup',
            side='defense',
            description='Light site presence, heavy retake utility',
            required_roles=[Role.ANCHOR, Role.FLEX, Role.FLEX, Role.SUPPORT, Role.LURK],
            preferred_agent_roles={
                Role.ANCHOR: ['sentinel'],
                Role.FLEX: ['controller', 'initiator'],
                Role.SUPPORT: ['initiator'],
                Role.LURK: ['duelist'],
            },
            timing_variance_ms=3000,
        ),
    }


class StrategyCoordinator:
    """Coordinates team strategies and role assignments."""

    # Agent type mappings
    AGENT_TYPES: Dict[str, str] = {
        # Duelists
        'jett': 'duelist', 'reyna': 'duelist', 'phoenix': 'duelist',
        'raze': 'duelist', 'yoru': 'duelist', 'neon': 'duelist', 'iso': 'duelist',
        # Initiators
        'sova': 'initiator', 'breach': 'initiator', 'skye': 'initiator',
        'kayo': 'initiator', 'fade': 'initiator', 'gekko': 'initiator',
        # Controllers
        'brimstone': 'controller', 'omen': 'controller', 'viper': 'controller',
        'astra': 'controller', 'harbor': 'controller', 'clove': 'controller',
        # Sentinels
        'sage': 'sentinel', 'cypher': 'sentinel', 'killjoy': 'sentinel',
        'chamber': 'sentinel', 'deadlock': 'sentinel', 'vyse': 'sentinel',
    }

    def __init__(self):
        self.current_strategy: Optional[Strategy] = None
        self.role_assignments: Dict[str, RoleAssignment] = {}
        self.strategy_history: List[str] = []
        self.round_number: int = 0

    def select_strategy(
        self,
        team_id: str,
        map_name: str,
        side: str,
        round_type: str,  # 'pistol', 'eco', 'force', 'half', 'full'
        team_credits: int,
        round_number: int,
        opponent_tendencies: Optional[Dict[str, float]] = None
    ) -> Strategy:
        """Select a strategy based on current conditions.

        Args:
            team_id: Team identifier
            map_name: Current map
            side: 'attack' or 'defense'
            round_type: Type of buy round
            team_credits: Total team credits
            round_number: Current round number
            opponent_tendencies: Optional info about opponent patterns

        Returns:
            Selected Strategy
        """
        self.round_number = round_number
        strategies = (
            StrategyDatabase.ATTACK_STRATEGIES if side == 'attack'
            else StrategyDatabase.DEFENSE_STRATEGIES
        )

        # Filter by economy requirements
        viable = [
            s for s in strategies.values()
            if s.min_economy <= team_credits
        ]

        if not viable:
            viable = list(strategies.values())

        # Weight strategies based on conditions
        weights = []
        for strat in viable:
            weight = 1.0

            # Prefer default in most situations
            if strat.strategy_id.endswith('default'):
                weight *= 1.5

            # Eco/force rounds prefer rush or aggressive
            if round_type in ['eco', 'force']:
                if 'rush' in strat.name.lower() or 'aggro' in strat.name.lower():
                    weight *= 2.0

            # Avoid repeating recent strategies
            recent = self.strategy_history[-3:]
            if strat.strategy_id in recent:
                weight *= 0.3  # Reduce probability of repetition

            # Map-specific weights could be added here
            weights.append(weight)

        # Normalize and select
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        strategy = random.choices(viable, weights=weights, k=1)[0]
        self.current_strategy = strategy
        self.strategy_history.append(strategy.strategy_id)

        # Keep history bounded
        if len(self.strategy_history) > 10:
            self.strategy_history = self.strategy_history[-10:]

        return strategy

    def assign_roles(
        self,
        players: List[Dict[str, str]],  # List of {player_id, agent}
        strategy: Strategy,
        custom_assignments: Optional[Dict[str, Role]] = None
    ) -> Dict[str, RoleAssignment]:
        """Assign roles to players based on strategy and agents.

        Args:
            players: List of player info dicts with player_id and agent
            strategy: Strategy to assign roles for
            custom_assignments: Optional forced role assignments

        Returns:
            Dict mapping player_id to RoleAssignment
        """
        assignments: Dict[str, RoleAssignment] = {}
        assigned_roles: Set[Role] = set()

        # Apply custom assignments first
        if custom_assignments:
            for player_id, role in custom_assignments.items():
                player = next((p for p in players if p['player_id'] == player_id), None)
                if player:
                    assignments[player_id] = RoleAssignment(
                        player_id=player_id,
                        role=role,
                        waypoints=self._generate_waypoints(role, strategy)
                    )
                    assigned_roles.add(role)

        # Get required roles still needed
        needed_roles = []
        for role in strategy.required_roles:
            if role not in assigned_roles:
                needed_roles.append(role)
                assigned_roles.add(role)

        # Match remaining players to roles based on agent type
        unassigned = [p for p in players if p['player_id'] not in assignments]

        for role in needed_roles:
            if not unassigned:
                break

            # Find best player for this role
            preferred_types = strategy.preferred_agent_roles.get(role, [])
            best_player = None
            best_score = -1

            for player in unassigned:
                agent = player.get('agent', 'unknown').lower()
                agent_type = self.AGENT_TYPES.get(agent, 'unknown')

                score = 0
                if agent_type in preferred_types:
                    score = len(preferred_types) - preferred_types.index(agent_type)

                if score > best_score:
                    best_score = score
                    best_player = player

            if best_player:
                assignments[best_player['player_id']] = RoleAssignment(
                    player_id=best_player['player_id'],
                    role=role,
                    waypoints=self._generate_waypoints(role, strategy)
                )
                unassigned.remove(best_player)

        # Assign remaining players as FLEX
        for player in unassigned:
            assignments[player['player_id']] = RoleAssignment(
                player_id=player['player_id'],
                role=Role.FLEX,
                waypoints=self._generate_waypoints(Role.FLEX, strategy)
            )

        self.role_assignments = assignments
        return assignments

    def _generate_waypoints(
        self,
        role: Role,
        strategy: Strategy
    ) -> List[StrategyWaypoint]:
        """Generate waypoints for a role within a strategy.

        This adds randomization to prevent deterministic movement patterns.
        """
        waypoints = []

        # Add timing variance
        variance = strategy.timing_variance_ms
        pos_variance = strategy.position_variance

        if strategy.side == 'attack':
            waypoints = self._generate_attack_waypoints(role, strategy, variance, pos_variance)
        else:
            waypoints = self._generate_defense_waypoints(role, strategy, variance, pos_variance)

        return waypoints

    def _generate_attack_waypoints(
        self,
        role: Role,
        strategy: Strategy,
        time_variance: int,
        pos_variance: float
    ) -> List[StrategyWaypoint]:
        """Generate attack-side waypoints with randomization."""
        waypoints = []

        # Random variance helper
        def vary_pos(pos: Tuple[float, float]) -> Tuple[float, float]:
            return (
                pos[0] + random.uniform(-pos_variance, pos_variance),
                pos[1] + random.uniform(-pos_variance, pos_variance)
            )

        def vary_time(base: int) -> Tuple[int, int]:
            delta = random.randint(-time_variance, time_variance)
            return (max(0, base + delta - 2000), base + delta + 2000)

        # Role-specific waypoints
        if role == Role.ENTRY:
            # Entry goes toward site first
            if strategy.primary_site == 'A' or random.random() < 0.5:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.35, 0.5)),
                    time_window_ms=vary_time(15000)
                ))
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.35, 0.35)),
                    time_window_ms=vary_time(30000),
                    requires_clear=True
                ))
            else:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.5, 0.65)),
                    time_window_ms=vary_time(15000)
                ))
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.65, 0.35)),
                    time_window_ms=vary_time(30000),
                    requires_clear=True
                ))

        elif role == Role.SUPPORT:
            # Support follows entry
            if strategy.primary_site == 'A' or random.random() < 0.5:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.3, 0.55)),
                    time_window_ms=vary_time(18000)
                ))
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.32, 0.4)),
                    time_window_ms=vary_time(32000),
                    ability_trigger='flash'
                ))
            else:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.45, 0.6)),
                    time_window_ms=vary_time(18000)
                ))
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.6, 0.4)),
                    time_window_ms=vary_time(32000),
                    ability_trigger='flash'
                ))

        elif role == Role.LURK:
            # Lurk takes opposite site or flank
            if strategy.primary_site == 'A':
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.6, 0.7)),
                    time_window_ms=vary_time(25000)
                ))
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.7, 0.5)),
                    time_window_ms=vary_time(45000),
                    is_optional=True
                ))
            else:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.25, 0.6)),
                    time_window_ms=vary_time(25000)
                ))
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.3, 0.4)),
                    time_window_ms=vary_time(45000),
                    is_optional=True
                ))

        elif role in [Role.FLEX, Role.IGL]:
            # Flex plays mid or supports main push
            waypoints.append(StrategyWaypoint(
                position=vary_pos((0.4, 0.55)),
                time_window_ms=vary_time(20000)
            ))
            waypoints.append(StrategyWaypoint(
                position=vary_pos((0.45, 0.45)),
                time_window_ms=vary_time(35000)
            ))

        return waypoints

    def _generate_defense_waypoints(
        self,
        role: Role,
        strategy: Strategy,
        time_variance: int,
        pos_variance: float
    ) -> List[StrategyWaypoint]:
        """Generate defense-side waypoints with randomization."""
        waypoints = []

        def vary_pos(pos: Tuple[float, float]) -> Tuple[float, float]:
            return (
                pos[0] + random.uniform(-pos_variance, pos_variance),
                pos[1] + random.uniform(-pos_variance, pos_variance)
            )

        def vary_time(base: int) -> Tuple[int, int]:
            delta = random.randint(-time_variance, time_variance)
            return (max(0, base + delta - 2000), base + delta + 2000)

        # Role-specific defense waypoints
        if role == Role.ANCHOR:
            # Anchors hold sites
            if strategy.primary_site == 'A' or (not strategy.primary_site and random.random() < 0.5):
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.3, 0.3)),
                    time_window_ms=vary_time(5000)
                ))
            else:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.7, 0.3)),
                    time_window_ms=vary_time(5000)
                ))

        elif role == Role.LURK:
            # Defense lurk pushes for info
            waypoints.append(StrategyWaypoint(
                position=vary_pos((0.5, 0.5)),
                time_window_ms=vary_time(8000)
            ))
            waypoints.append(StrategyWaypoint(
                position=vary_pos((0.45, 0.6)),
                time_window_ms=vary_time(20000),
                is_optional=True
            ))

        elif role == Role.OP:
            # OP player holds long angles
            if random.random() < 0.5:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.4, 0.35)),
                    time_window_ms=vary_time(6000)
                ))
            else:
                waypoints.append(StrategyWaypoint(
                    position=vary_pos((0.6, 0.35)),
                    time_window_ms=vary_time(6000)
                ))

        elif role in [Role.FLEX, Role.SUPPORT, Role.ENTRY]:
            # Flex rotates between sites
            waypoints.append(StrategyWaypoint(
                position=vary_pos((0.5, 0.35)),
                time_window_ms=vary_time(5000)
            ))

        return waypoints

    def check_rotation_trigger(
        self,
        round_state: 'RoundState',
        time_ms: int,
        info_gathered: Dict[str, any]
    ) -> Optional[str]:
        """Check if conditions trigger a rotation.

        Args:
            round_state: Current round state
            time_ms: Current time
            info_gathered: Information about enemy positions

        Returns:
            New target site if rotation triggered, None otherwise
        """
        if not self.current_strategy:
            return None

        # Check for number-based rotation (e.g., 3+ enemies spotted at site)
        enemy_at_a = info_gathered.get('enemies_at_a', 0)
        enemy_at_b = info_gathered.get('enemies_at_b', 0)

        if enemy_at_a >= 3:
            return 'A'
        if enemy_at_b >= 3:
            return 'B'

        # Check for spike-based rotation
        if round_state.spike_planted:
            return round_state.spike_site

        # Check for timeout rotation (attack)
        if self.current_strategy.side == 'attack' and time_ms > 70000:
            # Need to commit somewhere
            return self.current_strategy.primary_site or random.choice(['A', 'B'])

        return None

    def get_player_target_position(
        self,
        player_id: str,
        time_ms: int,
        current_position: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Get the target position for a player based on their role assignment.

        Args:
            player_id: Player to get target for
            time_ms: Current round time
            current_position: Player's current position

        Returns:
            Target position or None if no specific target
        """
        assignment = self.role_assignments.get(player_id)
        if not assignment or not assignment.waypoints:
            return None

        # Find current waypoint based on time
        for waypoint in assignment.waypoints:
            min_time, max_time = waypoint.time_window_ms
            if min_time <= time_ms <= max_time:
                return waypoint.position

            # If we're past this waypoint's window, check distance
            if time_ms > max_time:
                dx = current_position[0] - waypoint.position[0]
                dy = current_position[1] - waypoint.position[1]
                dist = math.sqrt(dx * dx + dy * dy)

                # Still trying to reach this waypoint
                if dist > 0.05:
                    return waypoint.position

        # Return last waypoint if past all windows
        return assignment.waypoints[-1].position if assignment.waypoints else None
