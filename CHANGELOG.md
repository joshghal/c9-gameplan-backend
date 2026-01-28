# Changelog

All notable changes to the C9 Tactical Vision simulation engine.

---

## [2026-01-21] - VCT Combat Detection & Visualization Overhaul

### Added

#### Schema (`app/schemas/simulations.py`)
- `facing_angle: Optional[float]` - Player facing direction in radians (0=right, π/2=down)
- `has_spike: bool` - Whether player is carrying spike

#### Visualization (`scripts/visualize.py`)
- **Vision cones**: 90° FOV cones showing player facing direction
  - Proper alpha compositing for transparency (can see players behind)
  - Red tint for C9, blue tint for opponents
- **Player names**: First 6 characters displayed next to each dot
- **Spike diamond marker**: Yellow/orange diamond at exact spike plant location
- **Off-map detection**: Magenta border for players with invalid positions
- **Updated legend**: Two-row layout with all visual indicators

### Changed

#### Combat Detection (`app/services/simulation_engine.py`)
- **VCT-Calibrated Engagement Thresholds** (based on 1,837 VCT kills):
  - Point-blank (<0.05 / 500 units): **FORCED** engagement - players MUST fight
  - Close range (0.05-0.10 / 500-1000 units): 50% per tick
  - Medium range (0.10-0.20 / 1000-2000 units): 15% per tick
  - Long range (>0.20): Phase-based probability (existing logic)
- **Previous**: 1-3% probabilistic per tick at all ranges (caused "walk past each other" bug)

#### Player Initialization (`app/services/simulation_engine.py`)
- **Initial facing angle**: Now calculated toward opening position
  - Attackers: Default -π/2 (up toward sites) if no opening position
  - Defenders: Default π/2 (down toward attacker spawn) if no opening position
- **Previous**: All players faced right (0 radians)

#### Spike Site Detection (`app/services/simulation_engine.py`)
- Now finds **nearest** site to planter position
- **Previous**: Defaulted to 'A' site when no site within radius matched

#### Retake Pathing (`app/services/simulation_engine.py`)
- Stronger override: Clears both `hold_position` and `is_holding_angle`
- Defenders on-site now face toward site center
- **Previous**: Only cleared `is_holding_angle`, could still have stale hold target

#### Vision System (`app/services/simulation_engine.py`)
- `_update_information_state()` now uses `player.facing_angle`
- **Previous**: Hardcoded facing (0 for attack, π for defense)

#### Defender Position Validation (`app/services/simulation_engine.py`)
- Added bounds check: positions must be within 0.08-0.92 range
- Invalid VCT positions fall back to site-based distribution
- 3-site maps use 2-2-1 distribution across A, B, C sites

#### Visualization Sizing (`scripts/visualize.py`)
- Player dots: C9 radius 7→4px, Opponents 6→3px
- C9 ring width: 2→1px
- Legend height: 50→70px (two rows)

### Fixed

#### Haven Map Bug
- **Issue**: All defenders had VCT opening positions at x=0.050 (off-map left edge)
- **Symptom**: 0.6 kills/round, 100% spike detonation wins (no contest)
- **Fix**: Position validation rejects invalid coords, uses site-based fallback
- **Result**: 5.2-6.6 kills/round, proper combat

### Stress Test Results

120 rounds across 12 maps:

| Map | Side | Win Rate | Avg Kills | Notes |
|-----|------|----------|-----------|-------|
| Ascent | Attack | 80% | 5.9 | |
| Ascent | Defense | 40% | 6.0 | |
| Bind | Attack | 90% | 2.7 | |
| Split | Defense | 70% | 7.0 | |
| Haven | Attack | 100% | 5.2 | Fixed from 0.6 kills |
| Lotus | Attack | 80% | 5.6 | |
| Icebox | Defense | 60% | 5.9 | |
| Breeze | Attack | 80% | 5.1 | |
| Fracture | Defense | 0% | 6.8 | Needs investigation |
| Pearl | Attack | 100% | 4.5 | |
| Sunset | Defense | 50% | 5.9 | |
| Abyss | Attack | 60% | 6.8 | |
| Corrode | Defense | 10% | 7.0 | Needs investigation |

**Summary:**
- Overall C9 Win Rate: 57%
- Attack Win Rate: 84%
- Defense Win Rate: 46%
- Average Kills/Round: **5.7** (target: ~7.5)

---

## Files Modified

| File | Lines Changed | Summary |
|------|---------------|---------|
| `app/schemas/simulations.py` | +3 | Added facing_angle, has_spike |
| `app/services/simulation_engine.py` | ~100 | Combat detection, facing, validation |
| `scripts/visualize.py` | ~80 | Vision cones, transparency, legend |
| `VCT_ANALYSIS_FINDINGS.md` | +35 | Engagement distance analysis |
| `ARCHITECTURE.md` | +70 | Recent updates section |

---

## How to Verify

```bash
# Run simulation
python scripts/simulate.py --map haven --c9-side attack --rounds 5 --opponent sentinels

# Visualize
python scripts/visualize.py output/sim_*.json -o output/viz_test.png

# Expected:
# - Haven: 5+ kills/round (was 0.6)
# - Vision cones visible on players
# - Spike diamond marker at plant location
# - Players face toward objectives at spawn
```
