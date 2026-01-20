# Coaching Simulator Roadmap

**Goal**: Enable coaches to ask counterfactual questions about tactical scenarios.

---

## Target Use Cases

### Type 1: Player Coordination Questions
> "What if OXY and xeppaa went together instead of separated at 15 seconds?"

**Requirements**:
- Player-specific movement patterns (not just role-based)
- Timing control (specify when players should be where)
- Synergy modeling (do these players work well together?)

### Type 2: Side Swap Questions
> "What if C9 played attacker on yesterday's match instead of defender?"

**Requirements**:
- Team tactical profiles (how does C9 attack vs defend?)
- Map-specific team tendencies
- Historical match data for comparison

### Type 3: Cross-Region Questions
> "What if C9 played against Asian teams? Is 7th US standing still valid?"

**Requirements**:
- Regional playstyle profiles
- Inter-region calibration
- Ranking/strength estimation

---

## Current State vs Target State

### What We Have

| Capability | Status | Data Source |
|------------|--------|-------------|
| Player combat profiles | ✅ 85 players | `opponent_profiles.json` |
| Player position data | ✅ 592,893 samples | `position_trajectories.json` |
| Player trade timing | ✅ Per-player | `trade_patterns.json` |
| Team rosters with roles | ✅ | `opponent_profiles.json` |
| Zone transition stats | ✅ 84,063 samples | `movement_patterns.json` |
| Basic simulation | ✅ 86% accuracy | `simulation_engine.py` |

### What's Missing

| Capability | Priority | Effort | Impact |
|------------|----------|--------|--------|
| Player synergy model | HIGH | Medium | Enables "OXY + xeppaa together" |
| Player position preferences | HIGH | Medium | Per-player positioning |
| Team tactical profiles | HIGH | High | Enables side-swap questions |
| Counterfactual API | HIGH | Medium | Coach interface |
| Regional style profiles | MEDIUM | High | Enables cross-region |
| Chain trade system | MEDIUM | Medium | Fixes man advantage gap |
| Utility system | LOW | High | Fixes retake gap |

---

## Implementation Phases

### Phase 1: Player Synergy Model (2-3 weeks)

**Goal**: Model how pairs of players perform together.

**Data Available**:
- `position_trajectories.json` has per-player XY every ~6 seconds
- Can calculate: distance between players, time spent together, zone overlap

**Implementation**:
```python
# New service: player_synergy.py

class PlayerSynergyModel:
    def calculate_synergy(self, player1: str, player2: str, map_name: str) -> float:
        """Calculate how well two players coordinate based on historical data."""
        # Extract from position_trajectories.json:
        # 1. How often they're within trade distance
        # 2. How often they push together
        # 3. Win rate when together vs separated

    def get_optimal_pairing(self, team: str, map_name: str) -> List[Tuple[str, str]]:
        """Find best player pairs for a map based on historical success."""
```

**Database Addition**:
```sql
CREATE TABLE player_synergies (
    player1_id UUID,
    player2_id UUID,
    map_name VARCHAR(50),
    synergy_score FLOAT,  -- 0-1 scale
    avg_distance FLOAT,
    time_together_pct FLOAT,
    win_rate_together FLOAT,
    sample_count INT
);
```

**Integration**:
- When initializing round, check if players have synergy data
- Apply synergy bonus to trade timing when paired players are nearby
- Adjust positioning to reflect historical preferences

---

### Phase 2: Player Position Preferences (2-3 weeks)

**Goal**: Each player should have preferred positions on each map.

**Data Available**:
- `position_trajectories.json` has 592,893 samples with player names
- Can extract: which zones each player frequents, at what round time

**Implementation**:
```python
# New service: player_positioning.py

class PlayerPositioningModel:
    def get_preferred_positions(
        self,
        player: str,
        map_name: str,
        side: str,
        phase: str  # "early", "mid", "late"
    ) -> List[Tuple[float, float, float]]:  # (x, y, probability)
        """Return player's preferred positions for this context."""

    def should_player_be_here(
        self,
        player: str,
        position: Tuple[float, float],
        context: SimulationContext
    ) -> float:
        """Return probability player would naturally be at this position."""
```

**Database Addition**:
```sql
CREATE TABLE player_position_preferences (
    player_id UUID,
    map_name VARCHAR(50),
    side VARCHAR(10),
    phase VARCHAR(20),
    zone_name VARCHAR(50),
    frequency FLOAT,
    avg_time_in_zone FLOAT,
    success_rate FLOAT
);
```

**Integration**:
- Replace generic zone targets with player-specific preferences
- When simulating "what if player X went to zone Y", calculate how unusual this is
- Show coach: "This would be unusual for OXY, he normally holds A main"

---

### Phase 3: Team Tactical Profiles (3-4 weeks)

**Goal**: Capture team-level tactical tendencies.

**What This Means**:
- How fast does C9 typically execute?
- Do they prefer A or B site on this map?
- How aggressive are they in mid control?
- What's their default setup vs their counter-strat?

**Data Extraction**:
```python
# Extract from position_trajectories.json

class TeamTacticalExtractor:
    def extract_execute_timing(self, team: str, map_name: str) -> Dict:
        """When does team typically execute? Early, mid, late?"""

    def extract_site_preference(self, team: str, map_name: str) -> Dict:
        """A vs B site split?"""

    def extract_aggression_profile(self, team: str, map_name: str) -> Dict:
        """How often do they contest mid early? Push for info?"""
```

**Database Addition**:
```sql
CREATE TABLE team_tactics (
    team_id UUID,
    map_name VARCHAR(50),
    side VARCHAR(10),
    tactic_name VARCHAR(50),  -- "default", "fast_a", "slow_b", etc.
    frequency FLOAT,
    avg_execute_time FLOAT,
    site_preference JSONB,
    player_positions JSONB,  -- Where each player usually is
    win_rate FLOAT
);
```

**Integration**:
- When simulating C9 attacking, use their actual tactical tendencies
- Allow coach to override: "Use C9's slow execute instead of their default"
- Compare: "C9's default vs Sentinels' default on this map"

---

### Phase 4: Counterfactual API (2 weeks)

**Goal**: Clean API for coaches to ask questions.

**API Design**:
```python
# POST /api/v1/whatif/player-pairing
{
    "base_match_id": "uuid",  # Reference match
    "round_number": 15,
    "modification": {
        "type": "player_pairing",
        "player1": "OXY",
        "player2": "xeppaa",
        "timing": 15000,  # ms into round
        "zone": "A_main"
    },
    "iterations": 100
}

# Response
{
    "original_outcome": {"win_rate": 0.45, "avg_time": 65000},
    "modified_outcome": {"win_rate": 0.58, "avg_time": 58000},
    "delta": "+13% win rate, -7s avg time",
    "confidence": 0.85,
    "key_factors": [
        "OXY and xeppaa have 0.72 synergy score",
        "Trade potential increased 2.3x",
        "Enemy attention split between two angles"
    ]
}
```

```python
# POST /api/v1/whatif/side-swap
{
    "match_id": "uuid",
    "swap_sides": true,  # C9 attacks instead of defends
    "iterations": 100
}

# Response
{
    "original": {"c9_rounds": 8, "opponent_rounds": 13},
    "swapped": {"c9_rounds": 10, "opponent_rounds": 11},
    "analysis": "C9's attack side win rate on Haven (62%) exceeds their defense (45%)"
}
```

```python
# POST /api/v1/whatif/cross-region
{
    "team": "Cloud9",
    "opponent_region": "APAC",
    "map": "Haven",
    "iterations": 500
}

# Response
{
    "expected_win_rate": 0.52,
    "confidence_interval": [0.48, 0.56],
    "key_differences": [
        "APAC teams average 15% faster execute timing",
        "C9 mid control rate (68%) exceeds APAC average (52%)",
        "Predicted advantage: C9 mid control, disadvantage: C9 retake speed"
    ]
}
```

---

### Phase 5: Regional Style Profiles (3-4 weeks)

**Goal**: Characterize NA vs APAC vs EMEA playstyles.

**What We Need**:
- Extract patterns from each region's VCT matches
- Identify systematic differences (execute timing, aggression, etc.)
- Create "synthetic team" that plays like average APAC team

**Data Challenge**:
- Current data is from 33 matches (mostly NA focused?)
- Need to expand data collection to cover all regions

**Implementation**:
```python
# New service: regional_profiles.py

class RegionalProfileModel:
    def get_regional_defaults(self, region: str) -> Dict:
        """Get average stats for a region."""
        return {
            "avg_execute_time": 45000,  # ms
            "aggression_level": 0.65,
            "mid_control_rate": 0.52,
            "retake_win_rate": 0.38,
            # ...
        }

    def simulate_vs_region(
        self,
        team: str,
        opponent_region: str,
        map_name: str
    ) -> SimulationResult:
        """Simulate team vs synthetic regional opponent."""
```

---

## Priority Order

### Must Have (Enables Core Use Cases)

1. **Player Synergy Model** - Enables "OXY + xeppaa together" questions
2. **Player Position Preferences** - Makes simulation player-specific
3. **Counterfactual API** - Coach interface
4. **Team Tactical Profiles** - Enables side-swap questions

### Nice to Have (Improves Accuracy)

5. **Chain Trade System** - Fixes man advantage scenarios
6. **Regional Profiles** - Enables cross-region questions
7. **Utility System** - Fixes retake scenarios

---

## Data Collection Needs

### High Priority
- More VCT matches for regional diversity
- Facing direction data (would enable readiness modeling)
- Ability usage timing per position

### Medium Priority
- Team-vs-team historical data
- Player performance vs specific opponents
- Clutch situation outcomes by player

---

## Success Metrics

### Phase 1 Success
- Can answer: "What's OXY + xeppaa synergy score?"
- Synergy affects simulation outcomes

### Phase 2 Success
- Can answer: "Where does OXY usually play on Haven defense?"
- Simulation uses player-specific positions

### Phase 3 Success
- Can answer: "What's C9's default vs Sentinels' default?"
- Side-swap simulation matches intuition

### Phase 4 Success
- Coach can submit questions via API
- Results include confidence intervals and explanations

### Phase 5 Success
- Can compare teams across regions
- Regional synthetic opponents are believable

---

## Estimated Timeline

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Player Synergy | 2-3 weeks | None |
| Phase 2: Position Preferences | 2-3 weeks | None (parallel with P1) |
| Phase 3: Team Tactics | 3-4 weeks | Phases 1-2 |
| Phase 4: Counterfactual API | 2 weeks | Phase 3 |
| Phase 5: Regional Profiles | 3-4 weeks | More data collection |

**Total**: ~12-16 weeks for full coaching simulator

**MVP (Phases 1-2 + basic API)**: ~6 weeks

---

## Immediate Next Steps

1. **Extract synergy data** from `position_trajectories.json`
   - Script to calculate pairwise player distances over time
   - Identify "together" vs "separated" moments

2. **Extract position preferences** from `position_trajectories.json`
   - Per-player zone frequency by map/side/phase
   - Heat maps of player positions

3. **Create database tables** for new models
   - `player_synergies`
   - `player_position_preferences`
   - `team_tactics`

4. **Build extraction scripts** in `scripts/` folder
   - `extract_player_synergy.py`
   - `extract_position_preferences.py`
   - `extract_team_tactics.py`
