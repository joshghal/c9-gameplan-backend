# C9 Gameplan — Backend

FastAPI backend powering the VALORANT tactical simulation engine. 128ms tick-rate simulation, VCT-calibrated combat model, A* pathfinding, economy engine, AI coaching via SSE streaming, and professional match data from 33 VCT matches.

**Live:** https://c9-gameplan-backend-902522310828.us-central1.run.app

## Tech Stack

- **FastAPI** — async REST API (Python 3.11)
- **NumPy / Pandas / Pillow / SciPy** — data processing, map masks, statistical modeling
- **Anthropic SDK** — AI coaching (narration, What-If chat, scouting reports)
- **SSE (sse-starlette)** — real-time streaming for AI responses
- **Pydantic v2** — request/response validation
- **GRID Esports API + Henrik API** — VCT professional match telemetry

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/simulations/` | Create simulation |
| POST | `/api/v1/simulations/{id}/tick` | Advance simulation by N ticks |
| POST | `/api/v1/simulations/{id}/monte-carlo` | Run N Monte Carlo iterations |
| GET | `/api/v1/strategy/rounds` | List VCT rounds for tactical planner |
| GET | `/api/v1/strategy/rounds/list` | List rounds by map |
| POST | `/api/v1/strategy/execute` | Execute tactical phase |
| GET | `/api/v1/matches/maps` | List available maps |
| GET | `/api/v1/matches/rounds/{map}` | Get rounds for a map |
| GET | `/api/v1/matches/round/{id}` | Get round replay data |
| POST | `/api/v1/coaching/narration/stream` | AI narration (SSE) |
| POST | `/api/v1/coaching/chat/stream` | What-If chat (SSE) |
| GET | `/health` | Health check |

## Getting Started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload    # http://localhost:8000
```

### Environment Variables

```bash
# .env
ASI1_API_KEY=sk_...              # LLM provider API key
FRONTEND_URL=http://localhost:3000  # CORS origin
```

## Project Structure

```
app/
├── main.py                      # FastAPI app, CORS, routes
├── config.py                    # Settings (env vars)
├── api/routes/
│   ├── simulations.py           # Simulation CRUD + tick + monte-carlo
│   ├── strategy.py              # Tactical planner endpoints
│   ├── matches.py               # VCT match archive
│   ├── coaching.py              # AI narration + chat (SSE)
│   ├── maps.py                  # Map data
│   └── teams.py                 # Team data
├── services/
│   ├── simulation_engine.py     # Core simulation loop (3,363 lines)
│   ├── combat_model.py          # Kill probability, LOS, weapons
│   ├── pathfinding.py           # A* with pattern-weighted costs
│   ├── ai_decision_system.py    # AI behavior logic
│   ├── ability_system.py        # 43 abilities from GRID data
│   ├── economy_engine.py        # VCT-calibrated buy decisions
│   ├── strategy_coordinator.py  # Named strategies, rotations
│   ├── behavior_adaptation.py   # Situational modifiers
│   ├── c9_realism.py            # C9 player-specific patterns
│   ├── vct_round_service.py     # VCT round data + ghost paths
│   ├── data_loader.py           # VCT data loading
│   └── llm/                     # AI integration
│       ├── client.py            # LLM client wrapper
│       ├── context_builder.py   # Simulation state → LLM context
│       ├── prompts.py           # Coaching persona
│       └── streaming.py         # SSE implementation
└── data/                        # VCT-derived data files
    ├── position_trajectories.json   # 51MB — 592,893 position samples
    ├── simulation_profiles.json     # 85 pro player profiles
    ├── c9_movement_models.json      # 5.6MB — C9 player patterns
    ├── behavioral_patterns.json     # Role-specific stats
    ├── vct_match_metadata.json      # Tournament/date/teams
    └── movement_patterns.json       # Zone transitions, 11 maps
```

## Data

- **592,893** position samples from **33 VCT matches**
- **12,029** kills analyzed across **85 players** on **11 maps**
- **43** unique abilities tracked from **2,294** GRID events
- **86%** accuracy against pro behavior benchmarks

## Deployment

Deployed on **Google Cloud Run** (gen2, 4 vCPU, 4Gi memory).

```bash
# Build and deploy
gcloud builds submit --tag us-central1-docker.pkg.dev/c9-gameplan/c9-gameplan/backend:latest
gcloud run deploy c9-gameplan-backend \
  --image us-central1-docker.pkg.dev/c9-gameplan/c9-gameplan/backend:latest \
  --region us-central1 \
  --allow-unauthenticated
```

Environment variables on Cloud Run: `ASI1_API_KEY`, `FRONTEND_URL`.

## Built With

Cloud9 Esports + JetBrains IDE + Junie AI — Sky's the Limit Hackathon 2026
