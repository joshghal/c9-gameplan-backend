# VCT Data Analysis Findings

## Data Sources Available

From 33 VCT matches with 1739 rounds:

| Data File | Content | Key Metrics |
|-----------|---------|-------------|
| `position_trajectories.json` (50MB) | Full player positions per round | 592,893 position samples, ~34 positions/player/round |
| `trade_patterns.json` (125KB) | Trade timing and distances | 3,036 trades, 1.72s avg trade time |
| `movement_patterns.json` (79KB) | Zone statistics by phase | 84,063 samples across 11 maps |
| `behavioral_patterns.json` (19KB) | Role behaviors, ability timing | Headshot rates, aggression levels |
| `hold_angles.json` (76KB) | Hold angle data by zone | Mean angles, avg distances |
| `economy_patterns.json` (12KB) | Buy patterns, loadout values | Win rates by buy type |

---

## Key Finding: Post-Plant Behavior Analysis

### Attacker Clustering (Post-Plant)

**VCT Reality:**
- Average distance between attackers: **2,541 units**
- Distribution:
  - Tight (<1000 units): 16.8%
  - Medium (1000-2000): 26.1%
  - **Spread (>2000): 57.1%**

**Our Simulation:**
- Spawns attackers within ±0.08 of site center (~80-160 units)
- **INCORRECT**: Real attackers spread out significantly post-plant

### Attacker Movement (Post-Plant)

**VCT Reality:**
- Average total movement: 50,399 units
- **99.5% are mobile** (>1500 units movement)
- Average velocity: 1,737 units/second
- Movement states:
  - Stationary: 7.8%
  - Walking: 32.1%
  - **Running: 60.1%**

**Our Simulation:**
- Assumes attackers are static/holding angles
- **INCORRECT**: Real attackers actively reposition and hunt

### Defender Coordination (Retake)

**VCT Reality:**
- Average arrival spread: **12.9 seconds**
- Coordination levels:
  - Coordinated (≤3s): 25.9%
  - Spread (3-10s): 24.0%
  - **Very spread (>10s): 50.1%**

**Our Simulation:**
- Defenders push one-by-one based on spawn distance
- **CORRECT**: This matches VCT reality!

---

## Retake Scenario Analysis

### VCT Target Re-interpretation

From `scenario_analysis_v6.py`:
```python
VCT_REFERENCE = {
    Scenario.RETAKE: {'attack_win_rate': 0.65}
}
```

This means:
- **Attackers win 65%** of retake scenarios
- **Defenders win 35%** of retake scenarios

### Our Simulation Results

- Defenders win: **26%** (attackers win 74%)
- Gap: 9 percentage points from VCT target

### Analysis

The 9% gap is smaller than previously thought. Given:
1. Attackers spread out (avg 2,541 units) - not clustered
2. Attackers mobile (99.5%) - not static holding
3. Defenders arrive spread out (50% >10s gap)

**Conclusion**: Our 26% defender win rate may be close to realistic. The VCT 35% might include utility usage, staged pushes, and other factors not modeled.

---

## Current Simulation Architecture

### View Sight / Line of Sight (LOS)

From `pathfinding.py`:

```python
def has_line_of_sight(self, pos1, pos2) -> bool:
    """Check if two positions have line of sight (no obstacles)"""
    # Uses Bresenham's line algorithm
    # Checks against wall grid (1 = wall, 0 = open)
```

The LOS system:
1. Uses 2D grid-based pathfinding
2. Checks obstacle collision along line between two points
3. Does NOT model:
   - Vertical height differences
   - Smoke/wall obstructions
   - Facing direction / field of view

### Player Behavior System

From `scenario_analysis_v6.py`:

```python
def _get_target(self, player, engine, map_data, config):
    # Returns position to move toward
    # Based on side and spike status
```

Movement behaviors:
- Attackers post-plant: Move to site center (static holding) - **NOT REALISTIC**
- Defenders retaking: Move to spike site - **REALISTIC**
- Combat triggers when players in range + LOS

### Combat Resolution

From `combat_system_v2.py`:

The combat system models:
1. TTK (Time to Kill) based on weapon damage
2. Headshot rates by agent role (from VCT data)
3. Position advantage (15% base, higher post-plant)
4. Trade windows (from VCT 1.72s avg)

---

## Magic Numbers and Their Basis

### Trade System Parameters

| Parameter | Value | VCT Basis |
|-----------|-------|-----------|
| `TRADE_WINDOW_MS` | 1700 | From `trade_patterns.json`: avg 1.72s |
| `MAX_TRADE_WINDOW_MS` | 3000 | From VCT distribution (3-5s bucket) |
| `TRADE_PROB_UNDER_1S` | 0.70 | VCT: 40% of trades <1s, aggressive |
| `TRADE_PROB_1_TO_2S` | 0.45 | VCT: 26% of trades 1-2s |
| `TRADE_PROB_2_TO_3S` | 0.30 | VCT: 14% of trades 2-3s |
| `MAX_TRADE_DISTANCE` | 0.15 | Estimated ~1500 units normalized |

### Position Parameters

| Parameter | Value | VCT Basis |
|-----------|-------|-----------|
| `POSITION_ADVANTAGE_PCT` | 0.15 | Estimated, no direct VCT data |
| `HOLDING_ADVANTAGE_PCT` | 0.30 | Estimated, no direct VCT data |
| `ATTACKER_SPAWN_SPREAD` | ±0.08 | **WRONG** - VCT shows 2500+ units spread |

### Headshot Rates (from behavioral_patterns.json)

| Role | HS Rate | Source |
|------|---------|--------|
| Duelist | 23.3% | VCT combat data |
| Controller | 19.6% | VCT combat data |
| Initiator | 19.0% | VCT combat data |
| Sentinel | 13.3% | VCT combat data |

---

## Recommendations

### High Priority Fixes

1. **Attacker Post-Plant Positioning**
   - Current: Clustered within ±0.08 of site center
   - Should be: Spread 2000+ units apart
   - VCT data shows attackers take off-angles, flanks

2. **Attacker Movement Behavior**
   - Current: Static holding
   - Should be: Active repositioning (60% running, 32% walking)
   - Model information gathering / hunting

### Medium Priority Fixes

3. **Position Advantage Calibration**
   - Current: 15% base, 30% post-plant (estimated)
   - Need: VCT data on peek advantage (not currently extracted)

4. **Utility System (Not Modeled)**
   - Flashes/smokes enable entries
   - Critical for realistic retakes
   - Would explain VCT 35% vs our 26%

### Low Priority (Refinements)

5. **Readiness State Modeling**
   - VCT data shows -0.013 correlation between distance and trade time
   - Readiness (pre-aimed) matters more than distance
   - Current probabilistic approach (19/44/37 split) approximates this

6. **Staged Entry Points**
   - VCT data shows defenders arrive very spread out (12.9s avg)
   - Coordinated entry would require explicit staging mechanics

---

## What The Simulation Gets RIGHT

1. **Defender arrival timing** - Matches VCT reality (spread out)
2. **Trade timing mechanics** - Based on VCT 1.72s average
3. **Headshot rates by role** - Direct from VCT data
4. **Economy win rates** - Close to VCT patterns
5. **Trade distance analysis** - Distance doesn't predict timing (VCT validated)

## What The Simulation Gets WRONG

1. **Attacker clustering** - Way too tight (80 units vs 2500 units)
2. **Attacker movement** - Static when should be 99.5% mobile
3. **Position advantage values** - Estimated, not VCT-derived
4. **No utility modeling** - Critical gap for retake scenarios

---

## Second-by-Second Granularity

**Can we track per player per second?** YES!

The `position_trajectories.json` contains:
- Clock values (100s countdown to 0)
- X, Y coordinates per player per tick
- Alive status
- Team and side

Example trajectory:
```json
{
  "player": "Jakee",
  "positions": [
    {"clock": 100, "x": 10788.0, "y": -4.9, "alive": true},
    {"clock": 94, "x": 10747.1, "y": 134.4, "alive": true},
    {"clock": 84, "x": 10700.4, "y": 174.2, "alive": true},
    ...
  ]
}
```

**What we CAN know:**
- Player position every ~6 seconds (clock granularity)
- Movement direction and velocity
- When players die (alive flag)
- Relative positions between teammates/enemies

**What we CANNOT know:**
- Exact facing direction / view angle
- What each player sees (would need POV data)
- Ability usage timing per position
- Audio information state

---

## Conclusion

The VCT data analysis reveals that our simulation's "retake problem" (26% vs 65% target) is not as broken as thought:

1. The 65% is **attacker** win rate, meaning **defenders win 35%**
2. Our 26% defender win rate is only 9% off
3. VCT data confirms attackers spread out and actively hunt post-plant
4. VCT data confirms defenders DO arrive spread out (12.9s average)

The main improvements needed are:
1. Spread out attacker post-plant positions (2500+ units)
2. Model attacker movement/repositioning (99.5% mobile)
3. Consider adding utility system for staged entries

The coordination system experiment failed because crossfire mechanics help the clustered side - but VCT shows attackers aren't actually clustered. Fixing the spawn positions would change the coordination dynamics entirely.
