# Simulation Engine Data Integration Guide

**Last Updated**: January 2026
**Simulation Accuracy**: 86% (12/14 scenarios)

This guide explains how to integrate the extracted VCT match data into the simulation engine for more realistic player behaviors.

---

## Architecture Overview

See `ARCHITECTURE.md` in the project root for full codebase documentation.
See `COACHING_SIMULATOR_ROADMAP.md` for the path to coaching use cases.

## Data Files Available

The following data files are generated from 33 VCT matches:

| File | Description |
|------|-------------|
| `simulation_profiles.json` | Player behavior profiles (aggression, accuracy, roles) |
| `movement_patterns.json` | Zone statistics and transition probabilities |
| `opponent_profiles.json` | Team and player profiles for all opponents |
| `behavioral_patterns.json` | Role-specific and phase-specific behaviors |

## Using the Data Loader

```python
from app.services.data_loader import get_data_loader

# Get the singleton instance (loads data once)
data = get_data_loader()

# Get player profile by name or ID
profile = data.get_player_profile("OXY")
# Returns: PlayerProfile with stats like kd_ratio, headshot_rate, aggression

# Get team profile
team = data.get_team_profile("Cloud9")
# Returns: TeamProfile with players, avg_kd, team_aggression

# Get role behaviors
behavior = data.get_role_behavior("duelist")
# Returns: RoleBehavior with aggression_level, entry_probability, etc.

# Get combat parameters
combat = data.get_combat_params("duelist")
# Returns: CombatParams with engagement_distance_mean, headshot_base_rate

# Get player tendencies for PlayerTendencies dataclass
aggression, clutch, trade = data.get_player_tendencies("OXY")
```

## Integration Points

### 1. Player Creation (simulation_engine.py:528)

Replace random tendencies with data-driven values:

```python
from app.services.data_loader import get_data_loader

# In SimulationEngine.initialize_round():
data = get_data_loader()

# When creating a player:
player_name = player.get('name', '')
aggression, clutch, trade = data.get_player_tendencies(player_name)
profile = data.get_player_profile(player_name)

self.players[player['id']] = SimulatedPlayer(
    player_id=player['id'],
    team_id=session.attack_team_id,
    side='attack',
    x=pos[0],
    y=pos[1],
    agent=agent_name,
    has_spike=(i == 0),
    weapon=loadout.weapon if loadout else WeaponDatabase.WEAPONS['classic'],
    armor=loadout.armor if loadout else WeaponDatabase.ARMOR['none'],
    shield=loadout.armor.shield_value if loadout else 0,
    loadout_value=loadout.total_value if loadout else 0,
    # Use extracted data instead of random
    tendencies=PlayerTendencies(
        base_aggression=aggression,
        clutch_factor=clutch,
        trade_awareness=trade,
    ),
    # Set headshot rate from data
    headshot_rate=profile.headshot_rate if profile else 0.25,
    ultimate_cost=AGENT_ULTIMATE_COSTS.get(agent_name, 7),
)
```

### 2. Headshot Rate (simulation_engine.py:1419)

Use player-specific headshot rates:

```python
# In duel resolution:
player = self.players[player_id]
hs_rate = data.get_headshot_rate(player.player_id, player.agent)
is_headshot = random.random() < hs_rate
```

### 3. Movement/Positioning Decisions

Use zone transition probabilities for AI pathfinding:

```python
data = get_data_loader()

def choose_next_zone(self, current_zone: str, map_name: str) -> str:
    """Choose next zone based on transition probabilities."""
    transitions = data.get_zone_transitions(map_name)

    # Filter to transitions from current zone
    from_current = [t for t in transitions if t['from_zone'] == current_zone]

    if not from_current:
        return current_zone

    # Weighted random selection
    total_prob = sum(t['probability'] for t in from_current)
    r = random.random() * total_prob

    cumulative = 0
    for t in from_current:
        cumulative += t['probability']
        if r <= cumulative:
            return t['to_zone']

    return from_current[0]['to_zone']
```

### 4. Role-Based Behaviors

Use role behaviors for AI decisions:

```python
data = get_data_loader()

def should_entry(self, player: SimulatedPlayer) -> bool:
    """Decide if player should entry based on role."""
    role = self._get_player_role(player.agent)
    behavior = data.get_role_behavior(role)

    return random.random() < behavior.entry_probability

def get_aggression_level(self, player: SimulatedPlayer) -> float:
    """Get aggression level for player."""
    # First try player-specific
    profile = data.get_player_profile(player.player_id)
    if profile:
        return profile.aggression

    # Fall back to role-based
    role = self._get_player_role(player.agent)
    behavior = data.get_role_behavior(role)
    return behavior.aggression_level
```

### 5. Phase-Based Behaviors

Adjust behaviors based on round phase:

```python
data = get_data_loader()

def get_utility_usage_rate(self, phase: str) -> float:
    """Get utility usage rate for current phase."""
    phase_behavior = data.get_phase_behavior(phase)
    return phase_behavior.get('utility_usage_rate', 0.3)

def get_movement_speed(self, phase: str) -> float:
    """Get movement speed modifier for current phase."""
    phase_behavior = data.get_phase_behavior(phase)
    return phase_behavior.get('movement_speed', 0.7)
```

### 6. Economy Decisions

Use economy thresholds for buy decisions:

```python
data = get_data_loader()
economy = data.get_economy_behavior()

def determine_buy_type(self, credits: int) -> str:
    """Determine buy type based on extracted patterns."""
    if credits >= economy['full_buy_threshold']:
        return 'full_buy'
    elif credits >= economy['force_buy_threshold']:
        return 'force_buy'
    else:
        return 'eco'
```

## Data Statistics

- **Total Players Profiled**: 85
- **Teams**: 12
- **Position Samples**: 84,063
- **Maps Analyzed**: 11
- **Kills Analyzed**: 12,029

### C9 Player Profiles Available

| Player | Role | K/D | HS% | Aggression |
|--------|------|-----|-----|------------|
| OXY | Duelist | 1.11 | 26.6% | 34.3% |
| xeppaa | Initiator | 0.96 | 24.1% | 28.8% |
| v1c | Controller | 1.08 | 29.8% | 31.6% |
| neT | Sentinel | 0.90 | 31.0% | 28.3% |
| mitch | Initiator | 0.87 | 20.5% | 23.5% |

### Role Combat Averages

| Role | Engagement Distance | Headshot Rate | First Kill Rate |
|------|---------------------|---------------|-----------------|
| Duelist | 1777 ± 210 | 23.3% | 15.0% |
| Initiator | 1783 ± 178 | 19.0% | 7.4% |
| Controller | 1814 ± 200 | 19.6% | 7.1% |
| Sentinel | 1793 ± 210 | 13.3% | 8.9% |

## Example: Full Integration

```python
from app.services.data_loader import get_data_loader
from app.services.simulation_engine import SimulationEngine

class EnhancedSimulationEngine(SimulationEngine):
    """Simulation engine with data-driven behaviors."""

    def __init__(self, db):
        super().__init__(db)
        self.data = get_data_loader()

    async def initialize_round(self, session, round_type='normal'):
        # Call parent initialization
        await super().initialize_round(session, round_type)

        # Enhance players with extracted data
        for player_id, player in self.players.items():
            self._apply_player_data(player)

    def _apply_player_data(self, player: SimulatedPlayer):
        """Apply extracted data to player."""
        profile = self.data.get_player_profile(player.player_id)

        if profile:
            # Update tendencies
            player.tendencies.base_aggression = profile.aggression
            player.tendencies.clutch_factor = profile.clutch_potential
            player.headshot_rate = profile.headshot_rate
```

---

## Emergent Systems (New - Jan 2026)

The simulation now uses emergent systems instead of hardcoded penalties. These systems derive advantage from actual game mechanics.

### 1. Crossfire Advantage

Located in `simulation_engine.py:_get_crossfire_advantage()`

**How it works**:
- Counts teammates who have LOS on the same enemy
- +15% advantage per supporting teammate (capped at 2)
- Disadvantage when facing multiple enemies with LOS on you

**When it triggers**:
- Every engagement in `_resolve_engagement()`
- Applied to reaction time and accuracy

**Example**:
```python
# Player has 2 teammates with LOS on enemy
crossfire_advantage = 1.30  # 30% faster reaction, better accuracy

# Player facing 2 enemies who both have LOS
crossfire_advantage = 0.77  # Must split attention, slower reaction
```

### 2. Information Advantage

Located in `simulation_engine.py:_get_information_advantage()`

**How it works**:
- Checks if any teammate heard this enemy recently (via sound system)
- Uses `information_system.py` for knowledge tracking
- More players = more ears = better map coverage

**When it triggers**:
- Every engagement
- Applied to reaction time

**Example**:
```python
# Teammate heard enemy running 2 seconds ago
info_advantage = 1.20  # 20% faster reaction (expected enemy)

# No prior info on enemy location
info_advantage = 1.00  # Neutral
```

### 3. Molly Repositioning

Located in `simulation_engine.py:_apply_ability_effects()`

**How it works**:
- `ability_system.py` sets `should_reposition=True` when player is in molly
- Simulation reads this flag and sets `player.reposition_target`
- Movement system prioritizes reposition target over normal movement

**Player state added**:
```python
@dataclass
class SimulatedPlayer:
    # ... existing fields ...
    reposition_target: Optional[Tuple[float, float]] = None  # Flee to here
```

**Movement priority** (highest to lowest):
1. `reposition_target` - Fleeing danger (molly)
2. `sound_reaction_target` - Investigating sound
3. `site_execute_target` - Site execution
4. `strategy_target` - Role-based positioning

### Integration Example

```python
from app.services.simulation_engine import SimulationEngine

# The emergent systems are automatically used
# No special integration needed - they're built into _resolve_engagement()

# To check crossfire status for debugging:
attacker_support = engine._count_teammates_with_los_on_enemy(
    attacker, defender, time_ms
)
defender_support = engine._count_teammates_with_los_on_enemy(
    defender, attacker, time_ms
)
print(f"Attacker has {attacker_support} teammates supporting")
print(f"Defender has {defender_support} teammates supporting")

# To check information status:
info_adv = engine._get_information_advantage(attacker, defender, time_ms)
print(f"Attacker info advantage: {info_adv:.2f}x")
```

### Why Emergent is Better

| Scenario | Hardcoded | Emergent |
|----------|-----------|----------|
| 4v5 clustered | 60% penalty | Depends on actual positions |
| 4v5 spread | 60% penalty | Less penalty (no crossfire) |
| Retake | Same everywhere | Based on LOS geometry |
| Sound intel | Not modeled | Detection affects reaction |

---

## Upcoming: Coaching Features

See `COACHING_SIMULATOR_ROADMAP.md` for the full plan. Key additions:

### Player Synergy (Phase 1)
```python
synergy = data.get_player_synergy("OXY", "xeppaa", "haven")
# Returns: 0.72 (high synergy - they coordinate well)
```

### Position Preferences (Phase 2)
```python
positions = data.get_preferred_positions("OXY", "haven", "defense", "early")
# Returns: [(0.45, 0.32, 0.6), (0.52, 0.28, 0.3), ...]
```

### Team Tactics (Phase 3)
```python
tactics = data.get_team_tactics("Cloud9", "haven", "attack")
# Returns: {"default": {...}, "fast_a": {...}, "slow_b": {...}}
```
