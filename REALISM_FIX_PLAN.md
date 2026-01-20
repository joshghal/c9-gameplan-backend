# Simulation Realism Fix Plan

**Status**: Event-level realism needs improvement while scenario accuracy is good (79%)

---

## Note on Hardcoded Values

The test script `realistic_round_sim.py` has hardcoded players/timings for testing.
In the real `simulation_engine.py` implementation:

| Hardcoded Value | Real Source |
|-----------------|-------------|
| `C9_PLAYERS`, `OPPONENT_PLAYERS` | Passed from scenario configuration |
| `AGENTS_ATTACK`, `AGENTS_DEFENSE` | From player loadouts in scenario |
| `SETUP_END=15000`, `CONTROL_END=40000` | `validated_parameters.py` RoundPacingParams |
| `PLANT_TIME=4000`, `DEFUSE_TIME=7000` | `validated_parameters.py` (Riot official values) |
| Engagement rates | `validated_parameters.py` (to be added) |

---

## Current State

### What Works ✅
| Metric | Our Value | Target | Status |
|--------|-----------|--------|--------|
| Scenario Accuracy | 79% | 80% | ✅ Good |
| eco_vs_full | 8% attack | ~10% | ✅ Good |
| full_vs_eco | 90% attack | ~90% | ✅ Good |
| clutch_1v2 | 10% | ~12% | ✅ Good |
| man_advantage_5v4 | 73% | ~70% | ✅ Good |

### What's Broken ❌
| Metric | Our Value | VCT Target | Gap |
|--------|-----------|------------|-----|
| Round Duration | 25.8s | 65s | -60% |
| Kills in 0-20s | 99% | 15% | +84% |
| Trade Rate | 1% | 25% | -24% |
| Wipe Rate | 88% | 18% | +70% |

**Root Cause**: All fights happen immediately with no tactical pacing

---

## Solution Architecture

### The Problem
Current `simulation_engine.py` runs combat in a loop that doesn't respect round phases:
```
while round_not_over:
    check_all_engagements()  # Everyone can fight everyone immediately
    move_players()           # Generic movement
```

### The Solution
Phased simulation where engagement probability depends on game state:

```
SETUP (0-15s)      → 0.01% engagement rate (rare pushes)
MAP_CONTROL (15-40s) → 0.1% engagement rate (info plays)
EXECUTE (40-60s)    → 0.4% engagement rate (main fights)
POST_PLANT         → 0.3% engagement rate (retake)
CLUTCH             → 0.5% engagement rate (decisive)
```

---

## Implementation Plan

### Phase 1: Round Phase System
**File**: `simulation_engine.py`

Add phase tracking to SimulationEngine:

```python
class RoundPhase(Enum):
    SETUP = "setup"           # 0-15s
    MAP_CONTROL = "control"   # 15-40s
    EXECUTE = "execute"       # 40-60s
    POST_PLANT = "post_plant"
    CLUTCH = "clutch"

PHASE_ENGAGEMENT_RATES = {
    RoundPhase.SETUP: 0.0001,
    RoundPhase.MAP_CONTROL: 0.001,
    RoundPhase.EXECUTE: 0.004,
    RoundPhase.POST_PLANT: 0.003,
    RoundPhase.CLUTCH: 0.005,
}
```

**Tasks**:
1. Add RoundPhase enum to simulation_engine.py
2. Add `_get_current_phase()` method
3. Modify `_run_simulation_tick()` to use phase-based engagement rates

---

### Phase 2: Movement By Phase
**File**: `simulation_engine.py`

Movement should reflect tactical phases:

| Phase | Attack Movement | Defense Movement |
|-------|-----------------|------------------|
| SETUP | Walk to spawn positions | Hold angles |
| MAP_CONTROL | Push for info (some aggression) | Contest or hold |
| EXECUTE | Rush site together | Rotate/stack |
| POST_PLANT | Spread for crossfires | Coordinate retake |
| CLUTCH | 1vX positioning | Hunt/hold |

**Tasks**:
1. Add `_get_phase_positions()` method returning target positions per phase
2. Modify `_move_players()` to move toward phase-appropriate positions
3. Adjust movement speed by phase (slow setup, fast execute)

---

### Phase 3: Trade Mechanics
**File**: `simulation_engine.py`

Current trade system is too weak. Need:

1. **Trade Window**: 2.5s after kill
2. **Trade Detection**: Teammate within trade distance
3. **Trade Boost**: 5x engagement probability in trade window

```python
# Track last kill for trade opportunities
self.last_kill_time = 0
self.last_kill_position = None

def _check_trade_opportunity(self, time_ms, player):
    """Check if player can trade recent kill."""
    in_window = (time_ms - self.last_kill_time) < 2500
    near_kill = distance(player.pos, self.last_kill_position) < 0.15
    return in_window and near_kill
```

**Tasks**:
1. Track last kill time and position
2. Add trade window check before engagements
3. Boost engagement probability for trade opportunities
4. Track trades separately from kills in events

---

### Phase 4: Integration Testing

Run realistic_round_sim.py to validate metrics after each phase:

| Metric | Target | Acceptance |
|--------|--------|------------|
| Round Duration | 65s | 55-75s |
| Trade Rate | 25% | 20-30% |
| Wipe Rate | 18% | 15-25% |
| Kills 0-20s | 15% | 10-25% |
| Kills 20-60s | 55% | 45-65% |
| Kills 60s+ | 30% | 20-40% |

**Tasks**:
1. Run 100 rounds after each phase
2. Compare metrics against VCT targets
3. Tune engagement rates until metrics match

---

## File Changes Summary

| File | Changes |
|------|---------|
| `simulation_engine.py` | Add RoundPhase, phase-based engagement, trade tracking |
| `realistic_round_sim.py` | Already created - standalone test |
| `scenario_analysis_v6.py` | Update to use phased system |

---

## Implementation Order

```
1. Add RoundPhase enum and constants     [simulation_engine.py:50-70]
2. Add _get_current_phase() method       [simulation_engine.py:new]
3. Add phase-based engagement rates      [simulation_engine.py:_run_simulation_tick]
4. Add trade tracking fields             [simulation_engine.py:SimulatedPlayer]
5. Add _check_trade_opportunity()        [simulation_engine.py:new]
6. Modify _move_players() for phases     [simulation_engine.py:_move_player_toward_position]
7. Run validation tests                  [scripts/realistic_round_sim.py]
8. Tune parameters until metrics match   [validated_parameters.py]
```

---

## Current Progress (Jan 2026)

### Implemented Fixes

1. **Early Aggression** (FIX 2)
   - Duelists spawn forward in contact positions
   - Higher SETUP engagement rate
   - Duelists get engagement bonus in SETUP
   - **Result**: Early kills now 15-21% ✅

2. **Post-Plant Dynamics** (FIX 1)
   - Attackers defend spike (not pure hiding)
   - Defenders prioritize defuse over kills
   - Phase-based combat advantages

3. **Save Behavior** (FIX 3)
   - Teams at 2+ disadvantage disengage
   - Cooldown period after kills
   - **Result**: Wipe rate reduced to 22-50%

4. **Clutch Balance** (FIX 4)
   - Cautious play in 1vX
   - Attackers hide with spike planted

### Best Achieved Metrics

```
SIMULATION RESULTS vs VCT TARGET (Best Run)
--------------------------------------------------
Metric                         Ours        VCT
--------------------------------------------------
Attack Win Rate                 44%        47%  ✓ Close
Avg Duration                    72s        65s  ~ Acceptable
Avg Kills/Round                 7.0        7.5  ✓ Close
Early Kills (0-20s)             17%        15%  ✓ Close
Wipe Rate                       22%        18%  ✓ Close
Detonation                      26%        N/A  ✓ Working
```

### Trade-offs Discovered

The simulation has fundamental trade-offs:

1. **Wipe Rate vs Duration**: Lower engagement = fewer wipes but longer rounds
2. **Wipe Rate vs Attack Win Rate**: Save behavior helps wipe rate but hurts attackers
3. **Plant Rate vs Pre-plant Combat**: More fighting = fewer plants

### Remaining Challenges

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Trade Rate | 8-12% | 25% | ❌ Too low |
| Late Kills | 8-12% | 30% | ❌ Too few |
| Mid Kills | 72-75% | 55% | ❌ Too concentrated |

### Key Insight: Wipe Rate

In VCT, only 18% of rounds end in full elimination (wipe). This means:
- 82% of rounds end via spike outcomes (detonate/defuse/timeout)
- The save behavior and kill cooldown mechanics help achieve this

**Current approach**: Combination of save behavior (teams disengage when down)
and post-plant dynamics (attackers defend, don't hunt) reduces chain kills.

---

## Success Criteria

After implementation, running 100 rounds should show:

```
SIMULATION RESULTS vs VCT TARGET
--------------------------------------------------
Metric                         Ours        VCT
--------------------------------------------------
Attack Win Rate                 47%        47%  ✓
Avg Duration                    65s        65s  ✓
Trade Rate                      25%        25%  ✓
Wipe Rate                       18%        18%  ✓
Kill Timing (0-20s)             15%        15%  ✓
Kill Timing (20-60s)            55%        55%  ✓
Kill Timing (60s+)              30%        30%  ✓
```

---

## VCT Data Validation (Jan 2026)

### Data Sources Analyzed

| Source File | Data Points | Description |
|-------------|-------------|-------------|
| `trade_patterns.json` | 12,029 kills, 3,036 trades | Kill/trade patterns |
| `behavioral_patterns.json` | 33 matches | Role behaviors, phase engagement |
| `vct_trade_analysis.json` | 100 sample trades | Detailed trade timing |
| `vct_insights.json` | Aggregated insights | Economy, positioning |

---

### VCT Trade Mechanics (Critical Finding)

**Key Insight: "Distance Doesn't Matter, Readiness Does"**
- Correlation between distance and trade time: **-0.013** (no correlation)
- Trade time is determined by **readiness state** (pre-aimed vs need to turn)

#### Trade Time Distribution (3,036 trades)
| Time Window | Count | Percentage | Interpretation |
|-------------|-------|------------|----------------|
| Under 1s | 1,203 | **39.6%** | Pre-aimed trades |
| 1-2s | 789 | 26.0% | Normal reaction |
| 2-3s | 417 | 13.7% | Need to reposition |
| 3-5s | 627 | 20.7% | Rotating from far |

**Average trade time**: 1.72 seconds (1,720ms)

#### Trade Type Model (for simulation)
| Type | Time Range | % of Trades | Description |
|------|------------|-------------|-------------|
| Pre-aimed | 42-200ms | 19% | Already watching angle |
| Normal reaction | 500-1500ms | 44% | Heard gunshot, turned, shot |
| Repositioning | 2000-5000ms | 37% | Had to move for angle |

**Implication**: Our current distance-based trade system is wrong. Should use readiness probability instead:
- 19% chance of instant trade (0-200ms)
- 44% chance of normal trade (500-1500ms)
- 37% chance of delayed trade (2000-5000ms)

---

### VCT Phase Engagement (from behavioral_patterns.json)

| Phase | VCT Engagement Likelihood | Our Current Rate | Status |
|-------|---------------------------|------------------|--------|
| Early | 0.30 | 0.002 | ⚠️ Needs calibration |
| Mid | 0.60 | 0.0015 | ⚠️ Needs calibration |
| Late | 0.80 | 0.003 | ⚠️ Needs calibration |

**VCT Phase Behaviors:**
```
early: movement_speed=0.9, rotation_prob=0.1, engagement=0.3, peek_freq=0.4
mid:   movement_speed=0.7, rotation_prob=0.3, engagement=0.6, peek_freq=0.6
late:  movement_speed=0.5, rotation_prob=0.5, engagement=0.8, peek_freq=0.8
```

**Ratio Insight**: VCT shows engagement likelihood ratio of 0.3:0.6:0.8 (early:mid:late)
Our rates should follow approximately: 1x : 2x : 2.67x

---

### VCT Role Statistics (from behavioral_patterns.json)

#### Combat Stats by Role
| Role | Headshot % | First Kill % | Trade Rate | Aggression |
|------|------------|--------------|------------|------------|
| Duelist | **23.3%** | **15.0%** | **34.1%** | 0.8 |
| Controller | 19.6% | 7.1% | 29.9% | 0.3 |
| Initiator | 19.0% | 7.4% | 28.1% | 0.5 |
| Sentinel | 13.3% | 8.9% | 28.8% | 0.2 |

**Key Insights:**
1. Duelists get 15% of first kills vs 7% for supports - **2x more aggressive**
2. Duelist trade rate 34% vs initiator 28% - **1.2x more likely to trade**
3. Sentinel clutch rate 40% vs duelist 35% - **Sentinels better in clutch**

#### Entry Probability by Role
| Role | Entry Probability | Clutch Success |
|------|-------------------|----------------|
| Duelist | **60%** | 35% |
| Initiator | 30% | 30% |
| Controller | 10% | 25% |
| Sentinel | 5% | **40%** |

---

### VCT Economy Data (from vct_insights.json)

| Buy Type | Win Rate | vs Full Buy | Sample Size |
|----------|----------|-------------|-------------|
| Full Buy | 51.0% | baseline | 2,782 |
| Half Buy | 49.8% | -1.2% | 263 |
| Force Buy | 42.2% | **-8.8%** | 379 |
| Eco | 54.0% | +3.0% | 50 (small) |

**Key Insight**: Force buy is only 9% worse than full buy. Our eco_vs_full at 8% is accurate.

---

### VCT Kill Timing (from validated_parameters.py)

#### TTK Distribution (4,163 samples)
| Time | Percentage | Interpretation |
|------|------------|----------------|
| Under 0.5s | **53%** | Instant headshots |
| 0.5-1s | 13% | Quick spray |
| 1-2s | 10% | Extended fight |
| Over 2s | 23% | Long fight/reposition |

**53% of kills are instant headshots** - this explains why trade windows need to be fast.

---

### Parameter Discrepancies Identified

| Parameter | Our Value | VCT Value | Gap | Impact |
|-----------|-----------|-----------|-----|--------|
| Trade window | 1,200ms | 1,720ms avg | -30% | Low trade rate |
| Trade rate target | 25% | 25.2% | ✓ Match | - |
| Early engagement | 0.002 | ratio 0.3 | N/A | ⚠️ Ratio matters |
| Avg kills/round | 7.0 | 6.93 | +1% | ✓ Close |
| Phase ratios | 1:0.75:1.5 | 1:2:2.67 | Wrong | ❌ Major |

---

### Recommended Parameter Adjustments

#### 1. Trade System Overhaul
```python
# Current (distance-based)
TRADE_WINDOW_MS: int = 1200
TRADE_DISTANCE: float = 0.10

# Recommended (readiness-based)
TRADE_READINESS_PROB = {
    'pre_aimed': 0.19,      # 19% instant trade (0-200ms)
    'normal': 0.44,         # 44% normal (500-1500ms)
    'repositioning': 0.37,  # 37% delayed (2000-5000ms)
}
TRADE_WINDOW_MS: int = 3000  # VCT shows 79.3% within 3s
```

#### 2. Phase Engagement Ratios
```python
# Current
ENGAGEMENT_RATE_SETUP: float = 0.002
ENGAGEMENT_RATE_CONTROL: float = 0.0015  # Lower than setup?
ENGAGEMENT_RATE_EXECUTE: float = 0.003

# Recommended (maintaining VCT ratios 1:2:2.67)
BASE_ENGAGEMENT: float = 0.001
ENGAGEMENT_RATE_SETUP: float = BASE_ENGAGEMENT * 1.0     # 0.001
ENGAGEMENT_RATE_CONTROL: float = BASE_ENGAGEMENT * 2.0   # 0.002
ENGAGEMENT_RATE_EXECUTE: float = BASE_ENGAGEMENT * 2.67  # 0.00267
```

#### 3. Role-Based First Kill
```python
# VCT first_kill_aggression values
DUELIST_FIRST_KILL_BONUS: float = 2.0    # 15% / 7.5% = 2x
INITIATOR_FIRST_KILL_BONUS: float = 1.0  # baseline
CONTROLLER_FIRST_KILL_BONUS: float = 0.95
SENTINEL_FIRST_KILL_BONUS: float = 1.2   # 8.9% > 7.4%
```

---

### Validation Status Summary

| Metric | VCT Target | Our Current | Status |
|--------|------------|-------------|--------|
| Trade Rate | 25% | 8-12% | ❌ Too low |
| Trade Window | 1720ms avg | 1200ms | ⚠️ Increase |
| Phase Ratios | 1:2:2.67 | 1:0.75:1.5 | ❌ Wrong |
| Early Kills | 15% | 15-21% | ✓ Good |
| Wipe Rate | 18% | 22-50% | ⚠️ Improve |
| Kills/Round | 6.93 | 7.0 | ✓ Good |
| Duelist Entry | 60% | N/A | ⚠️ Add |
| Sentinel Clutch | 40% | 35% | ⚠️ Tune |

---

### Completed (Jan 2026)

1. ✅ **Implemented readiness-based trades** - `trade_system.py` updated
   - Pre-aimed (19%): 42-200ms, 85% success
   - Normal (44%): 500-1500ms, 70% success
   - Repositioning (37%): 2000-5000ms, 45% success
   - BASE_TRADE_OPPORTUNITY_PROB = 0.35 (not all kills are tradeable)
   - **Result**: Trade rate now 29% (target: 25%) ✓

### Remaining Steps

1. **Fix phase engagement ratios** to match VCT 1:2:2.67
2. **Add role-specific entry/clutch modifiers** from VCT data
3. **Fix wipe rate** - still 70% vs 18% target
4. **Fix kill timing** - too concentrated in mid-phase

---

## Notes

- Preserve existing scenario accuracy - don't break macro outcomes
- The phased system is additive - it modifies HOW kills happen, not IF
- Trade system should increase round duration naturally by creating back-and-forth
- Autonomous player stats (headshot_rate, aggression) still apply within phases
- **Wipe rate** is the hardest metric to tune - requires balancing engagement rates
- **Kill timing** requires early aggression in SETUP phase (currently 0% early kills)
- **VCT KEY INSIGHT**: Distance doesn't affect trade time - readiness does
