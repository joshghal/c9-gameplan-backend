# C9 Tactical Vision - Architecture Overview

**Last Updated**: January 30, 2026
**Simulation Accuracy**: 86% (12/14 scenarios)
**Average Kills/Round**: 5.7 (VCT target: ~7.5)

---

## Project Goal

A VALORANT tactical simulation platform for coaching analysis. Enables counterfactual questions like:
- "What if OXY and xeppaa pushed together at 15 seconds?"
- "What if C9 played attacker side in yesterday's match?"
- "How would C9 perform against Asian teams?"

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | FastAPI 0.109 (async) | REST API |
| Database | PostgreSQL + asyncpg | Persistent storage |
| Cache | Redis 5.0 | Session cache |
| Data Source | GRID esports API | VCT match data |
| Processing | Pandas, Numpy | Data analysis |

---

## Directory Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings (DB, Redis, GRID API)
│   ├── database.py             # SQLAlchemy async setup
│   │
│   ├── api/routes/             # REST endpoints
│   │   ├── simulations.py      # Simulation CRUD + execution
│   │   ├── strategy.py         # Strategy planner + VCT replay
│   │   ├── matches.py          # Match/Round queries
│   │   ├── teams.py            # Team/Player management
│   │   ├── patterns.py         # Movement pattern queries
│   │   └── maps.py             # Map configs and zones
│   │
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── teams.py            # Team, Player
│   │   ├── matches.py          # Match, Round
│   │   ├── simulations.py      # SimulationSession
│   │   ├── patterns.py         # MovementPattern, PlayerTendency
│   │   ├── maps.py             # MapConfig, MapZone
│   │   └── positions.py        # RawPosition, GridEvent
│   │
│   ├── schemas/                # Pydantic request/response
│   │
│   ├── data/                   # VCT-extracted JSON (10 files)
│   │   ├── position_trajectories.json (50MB)
│   │   ├── vct_match_metadata.json
│   │   ├── trade_patterns.json
│   │   ├── behavioral_patterns.json
│   │   ├── hold_angles.json
│   │   ├── movement_patterns.json
│   │   ├── economy_patterns.json
│   │   ├── opponent_profiles.json
│   │   └── simulation_profiles.json
│   │
│   └── services/               # Core simulation (16,729 lines)
│       ├── [ACTIVE] simulation_engine.py (3,363 lines)
│       ├── [ACTIVE] combat_model.py (880 lines)
│       ├── [ACTIVE] weapon_system.py (610 lines)
│       ├── [ACTIVE] economy_engine.py (533 lines)
│       ├── [ACTIVE] round_state.py (482 lines)
│       ├── [ACTIVE] behavior_adaptation.py (477 lines)
│       ├── [ACTIVE] strategy_coordinator.py (709 lines)
│       ├── [ACTIVE] ability_system.py (1,143 lines)
│       ├── [ACTIVE] pathfinding.py (698 lines)
│       ├── [ACTIVE] pattern_matcher.py (360 lines)
│       ├── [ACTIVE] data_loader.py (601 lines)
│       ├── [ACTIVE] information_system.py (524 lines)
│       ├── [ACTIVE] ai_decision_system.py (1,187 lines)
│       ├── [ACTIVE] neural_ai_system.py (860 lines)
│       ├── [ACTIVE] validated_parameters.py (809 lines)
│       ├── [ACTIVE] vct_round_service.py        # VCT round data, ghost paths, coord normalization
│       ├── [ACTIVE] strategy_executor.py        # Pro-anchored strategy simulation
│       ├── [DATA TOOL] grid_parser.py (524 lines)
│       ├── [DATA TOOL] grid_data_extractor.py (775 lines)
│       ├── [DATA TOOL] player_profiles.py (433 lines)
│       └── [EXPERIMENTAL] trade_system.py (391 lines)
│
├── scripts/                    # Analysis and testing
│   ├── scenario_analysis_v6.py # Current best (86% accuracy)
│   └── ...
│
└── tests/                      # Unit and integration tests
```

---

## Core Simulation Flow

```
API Request (POST /simulations/{id}/start)
    │
    ▼
SimulationEngine.initialize_round()
    ├── WeaponDatabase → Load weapon stats
    ├── EconomyEngine → Generate loadouts from credits
    ├── StrategyCoordinator → Assign roles (duelist, initiator, etc.)
    └── AbilitySystem → Initialize agent abilities
    │
    ▼
SimulationEngine.advance() [tick loop at 128ms]
    ├── Movement
    │   ├── Pathfinding (A* with LOS)
    │   ├── Pattern Matcher (VCT patterns)
    │   └── Behavior Adaptation (aggression modifiers)
    │
    ├── Information
    │   ├── Sound Detection (running/walking radius)
    │   ├── Vision (LOS checks)
    │   └── Knowledge Decay (info gets stale)
    │
    ├── Combat (when players in range + LOS)
    │   ├── CombatModel → TTK calculation
    │   ├── Crossfire Advantage → Teammate LOS support
    │   ├── Information Advantage → Prior knowledge bonus
    │   └── Trade System → Post-kill trade window
    │
    └── Decisions
        ├── AIDecisionSystem → Rule-based decisions
        └── NeuralAISystem → NN-based (64 features → 10 actions)
    │
    ▼
Return SimulationState (players, events, phase)
```

---

## Active Services (What Each Does)

### Core Engine
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `simulation_engine.py` | 3,363 | Main orchestrator - round init, tick loop, events |
| `round_state.py` | 482 | Round phases, win probability, economy tracking |

### Combat
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `combat_model.py` | 880 | Gunfight mechanics - reaction time, accuracy, TTK |
| `weapon_system.py` | 610 | Weapon stats, damage falloff, armor calculation |

### Economy
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `economy_engine.py` | 533 | Credit tracking, buy decisions, loadout generation |

### Movement & Positioning
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `pathfinding.py` | 698 | A* pathfinding, LOS (Bresenham line algorithm) |
| `pattern_matcher.py` | 360 | Match movement to VCT patterns |

### Player Behavior
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `strategy_coordinator.py` | 709 | Role assignments (5 roles), team tactics |
| `behavior_adaptation.py` | 477 | Aggression modifiers, clutch factors |
| `ai_decision_system.py` | 1,187 | Rule-based decisions (hold/advance/execute/etc) |
| `neural_ai_system.py` | 860 | Neural network decisions (64→10 actions) |

### Abilities
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `ability_system.py` | 1,143 | 19 agents, ability mechanics, ultimate economy |

### Information
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `information_system.py` | 524 | Player knowledge - who saw what, info decay |

### Data
| Service | Lines | Responsibility |
|---------|-------|----------------|
| `data_loader.py` | 601 | Singleton loader for VCT JSON data |
| `validated_parameters.py` | 809 | VCT-derived parameters with justification |

---

## Data Files (VCT Extracted)

All from **33 VCT matches with 1,739 rounds**:

| File | Size | Content |
|------|------|---------|
| `position_trajectories.json` | 50MB | 592,893 position samples (~34 per player/round) |
| `trade_patterns.json` | 125KB | 3,036 trades, 1.72s avg, time distribution |
| `behavioral_patterns.json` | 19KB | Role HS rates (Duelist 23.3%, Sentinel 13.3%) |
| `hold_angles.json` | 76KB | Engagement angles/distances by zone |
| `movement_patterns.json` | 79KB | 84,063 zone transition samples |
| `economy_patterns.json` | 12KB | Buy thresholds, win rates by buy type |
| `opponent_profiles.json` | var | 85 player profiles with stats |
| `simulation_profiles.json` | var | Behavioral profiles for AI |

---

## Dead/Unused Code

| File | Lines | Status | Why |
|------|-------|--------|-----|
| `trade_system.py` | 391 | **EXPERIMENTAL** | Used only in scenario analysis scripts |

`coordination_system.py` and `combat_system_v2.py` were deleted (dead code cleanup).

---

## Emergent vs Hardcoded Systems

### Philosophy
Every outcome should EMERGE from underlying mechanics, not be hardcoded.

### Current Emergent Systems

1. **Economy** - TTK differences produce correct eco/full win rates
2. **Crossfire** - LOS checks determine teammate support advantage
3. **Information** - Sound system provides detection coverage
4. **Molly Repositioning** - Ability effects trigger movement

### Still Hardcoded (Needs Work)

1. **Position Advantage** - 15% base estimated, no VCT data
2. **Trade Distance** - 15% of map threshold is tuned
3. **Clutch Factor** - Profile-based but arbitrary

---

## API Endpoints

### Strategy (`/api/v1/strategy`)
- `GET /rounds` - Get random VCT round setup (map_name, side)
- `GET /maps` - List available maps
- `POST /execute` - Execute user strategy against pro-anchored AI
- `POST /replay` - Replay real VCT round with interpolated trajectories

### Simulations (`/api/v1/simulations`)
- `POST /` - Create session
- `POST /{id}/start` - Initialize round
- `POST /{id}/tick` - Advance N ticks
- `POST /{id}/snapshot` - Save state
- `POST /{id}/what-if` - Run alternative from snapshot
- `GET /{id}/analysis` - Get recommendations

### Matches (`/api/v1/matches`)
- `GET /` - List matches
- `GET /{id}/rounds` - Get rounds

### Teams (`/api/v1/teams`)
- `GET /` - List teams
- `GET /{id}/players` - Get roster

### Patterns (`/api/v1/patterns`)
- `POST /query` - Query best patterns
- `GET /tendencies/{player}` - Player tendencies
- `GET /strategies/{team}` - Team strategies

### Maps (`/api/v1/maps`)
- `GET /` - List map configs
- `GET /{name}/zones` - Get zones

---

## Database Schema

| Table | Key Fields |
|-------|------------|
| `teams` | id, name, region |
| `players` | id, team_id, name, role |
| `matches` | team1_id, team2_id, map, tournament |
| `rounds` | match_id, round_number, winner, spike_planted |
| `simulation_sessions` | teams, map, phase, snapshots (JSONB) |
| `movement_patterns` | team, map, side, waypoints (JSONB) |
| `map_configs` | map_name, walls_grid (JSONB) |

---

## What Works Well

| System | Accuracy | Why |
|--------|----------|-----|
| Economy scenarios | 9/10 | TTK differences are physics-based |
| Trade timing | 8/10 | Based on VCT 1.72s average |
| Role behaviors | 8/10 | Direct from VCT headshot/aggression data |
| Defender arrival | 8/10 | Matches VCT spread (12.9s avg) |

## What Needs Work

| System | Gap | Root Cause |
|--------|-----|------------|
| Man advantage 4v5 | 51% vs 35% | Missing chain trades |
| Man advantage 3v5 | 42% vs 15% | Missing utility stacking |
| Retake | 26% vs 35% | Attacker positioning wrong |
| Post-plant | 83% vs 65% | Attackers too static |

---

## Configuration

Key settings in `app/config.py`:

| Setting | Value | Purpose |
|---------|-------|---------|
| `simulation_tick_rate` | 128ms | Tick duration |
| `max_simulation_time` | 120000ms | 2 min round limit |
| `pattern_similarity_threshold` | 0.85 | Pattern matching tolerance |

---

## Recent Updates (January 2026)

### VCT-Calibrated Combat Detection

Fixed "players walk past each other" bug using VCT engagement distance data (1,837 kills analyzed):

| Distance | Normalized | Engagement |
|----------|------------|------------|
| Point-blank (<500u) | <0.05 | **FORCED** - always fight |
| Close (500-1000u) | 0.05-0.10 | 50% per tick |
| Medium (1000-2000u) | 0.10-0.20 | 15% per tick |
| Long (2000+u) | >0.20 | Phase-based probability |

### Player Facing Direction

- **Schema**: Added `facing_angle` (radians) and `has_spike` to `PlayerPosition`
- **Initialization**: Players face toward their opening position at spawn
  - Attackers: face toward sites (not right)
  - Defenders: face toward attacker spawn
- **Updates**: Facing tracked during movement and holding

### Spike Site Detection

- Fixed: Now finds **nearest** site to planter position
- Previously: Defaulted to 'A' site when no match within radius

### Retake Pathing

- Stronger override: Clears `hold_position` and `is_holding_angle`
- Defenders on-site face toward site center
- Vision system now uses `player.facing_angle` consistently

### Defender Position Validation

- Added bounds check (0.08-0.92) for VCT opening positions
- Invalid positions (e.g., Haven x=0.05 bug) fall back to site-based distribution
- 3-site maps: 2-2-1 distribution across A, B, C

### Visualization (`scripts/visualize.py`)

| Feature | Change |
|---------|--------|
| **Dot size** | C9: 7→4px, Opponents: 6→3px |
| **Vision cones** | 90° FOV, proper alpha compositing |
| **Player names** | First 6 chars displayed |
| **Spike marker** | Yellow diamond at exact plant location |
| **Off-map detection** | Magenta border for invalid positions |
| **Legend** | Two-row layout with all indicators |

---

## Stress Test Results (Jan 2026)

120 rounds across 12 maps:

| Metric | Value |
|--------|-------|
| Overall C9 Win Rate | 57% |
| Attack Win Rate | 84% |
| Defense Win Rate | 46% |
| Average Kills/Round | 5.7 |

---

## Next Steps

See `COACHING_SIMULATOR_ROADMAP.md` for the path to coaching use case.
