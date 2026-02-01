"""Coaching API routes for AI-powered tactical analysis."""

from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...services.coaching import (
    CoachService,
    ScoutingReportGenerator,
    MistakeAnalyzer,
    C9Predictor,
)
from ...services.coaching.coach_service import get_coach_service
from ...services.coaching.scouting_report import get_scouting_generator
from ...services.coaching.mistake_analyzer import get_mistake_analyzer
from ...services.coaching.c9_predictor import get_c9_predictor

router = APIRouter()


# Request/Response Models
class ChatRequest(BaseModel):
    """Request for coaching chat."""
    message: str
    session_id: Optional[str] = None
    map_context: Optional[str] = None
    team_context: Optional[str] = None
    use_tools: bool = True
    simulation_context: Optional[dict] = None  # Snapshots, events, final state for grounded responses


class ChatResponse(BaseModel):
    """Response from coaching chat."""
    session_id: str
    response: str


class ScoutingReportRequest(BaseModel):
    """Request for scouting report."""
    team_name: str
    map_name: Optional[str] = None
    force_refresh: bool = False


class C9PredictRequest(BaseModel):
    """Request for C9 action prediction."""
    map_name: str
    side: str
    phase: str = "opening"
    round_type: str = "full_buy"
    game_state: Optional[dict] = None
    opponent_team: Optional[str] = None


class MistakeAnalysisRequest(BaseModel):
    """Request for mistake analysis."""
    situation: str


class PositionAnalysisRequest(BaseModel):
    """Request for position analysis."""
    map_name: str
    positions: list[dict]
    side: str
    phase: str


# Endpoints
@router.post("/chat", response_model=ChatResponse)
async def coaching_chat(request: ChatRequest):
    """Send a message to the coaching AI (non-streaming)."""
    coach = get_coach_service()

    session_id = request.session_id or str(uuid4())

    # Initialize session with context if new
    session = coach.get_session(session_id)
    if not session:
        coach.create_session(
            session_id=session_id,
            map_name=request.map_context,
            team_name=request.team_context,
        )

    response = await coach.chat(
        session_id=session_id,
        message=request.message,
        use_tools=request.use_tools,
    )

    return ChatResponse(
        session_id=session_id,
        response=response,
    )


@router.post("/chat/stream")
async def coaching_chat_stream(request: ChatRequest):
    """Stream a response from the coaching AI using SSE."""
    coach = get_coach_service()

    session_id = request.session_id or str(uuid4())

    session = coach.get_session(session_id)
    if not session:
        coach.create_session(
            session_id=session_id,
            map_name=request.map_context,
            team_name=request.team_context,
        )

    # Store simulation context on session for grounded responses
    if request.simulation_context:
        session = coach.get_session(session_id)
        if session:
            session.simulation_context = request.simulation_context

    async def generate():
        async for event in coach.stream_chat(
            session_id=session_id,
            message=request.message,
            use_tools=request.use_tools,
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


@router.post("/scouting-report")
async def generate_scouting_report(request: ScoutingReportRequest):
    """Generate an AI scouting report for a team."""
    generator = get_scouting_generator()

    try:
        report = await generator.generate_report(
            team_name=request.team_name,
            map_name=request.map_name,
            force_refresh=request.force_refresh,
        )
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scouting-report/{team_name}")
async def get_scouting_report(
    team_name: str,
    map_name: Optional[str] = Query(None),
):
    """Get a cached scouting report or generate new one."""
    generator = get_scouting_generator()

    # Try cache first
    cached = generator.get_cached_report(team_name, map_name)
    if cached:
        return cached.to_dict()

    # Generate new
    report = await generator.generate_report(team_name, map_name)
    return report.to_dict()


@router.post("/c9-predict")
async def predict_c9_action(request: C9PredictRequest):
    """Predict what Cloud9 would do in a given situation."""
    predictor = get_c9_predictor()

    if request.phase == "opening":
        prediction = await predictor.predict_opening(
            map_name=request.map_name,
            side=request.side,
            round_type=request.round_type,
            opponent_team=request.opponent_team,
        )
    else:
        prediction = await predictor.predict_action(
            map_name=request.map_name,
            side=request.side,
            phase=request.phase,
            game_state=request.game_state or {},
            opponent_info={"team": request.opponent_team} if request.opponent_team else None,
        )

    return prediction.to_dict()


@router.post("/analyze-mistake")
async def analyze_mistake(request: MistakeAnalysisRequest):
    """Analyze a described situation for mistakes."""
    analyzer = get_mistake_analyzer()

    analysis = await analyzer.quick_analysis(request.situation)
    return analysis.to_dict()


@router.post("/analyze-position")
async def analyze_position(request: PositionAnalysisRequest):
    """Analyze a specific game position."""
    coach = get_coach_service()

    analysis = await coach.analyze_position(
        map_name=request.map_name,
        positions=request.positions,
        side=request.side,
        phase=request.phase,
    )

    return {"analysis": analysis}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a coaching session's history."""
    coach = get_coach_service()
    success = coach.clear_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "cleared"}


@router.post("/scouting-report/quick-summary")
async def quick_team_summary(
    team_name: str = Query(...),
    map_name: str = Query(...),
):
    """Get a quick 2-3 sentence summary for a team on a map."""
    generator = get_scouting_generator()
    summary = await generator.generate_quick_summary(team_name, map_name)
    return {"summary": summary}



class LiveNarrationRequest(BaseModel):
    """Request for live per-snapshot narration (Mode A)."""
    session_id: str
    snapshot: dict  # Single enriched snapshot
    previous_narrations: list[str] = []  # Last 5 narrations for context
    narration_type: str = "key_moment"  # "key_moment" or "transition"
    map_name: str = "ascent"
    attack_team: str = "cloud9"
    defense_team: str = "sentinels"
    player_roster: list[dict] = []  # [{id, name, agent, side, team}]


@router.post("/narrate-snapshot")
async def narrate_single_snapshot(request: LiveNarrationRequest):
    """Live Mode A: Narrate a single snapshot during simulation.

    Called when auto-snapshot fires (kill, plant, time interval).
    Simulation pauses until this returns. Returns structured narration step.
    """
    import json
    from ...services.llm import get_anthropic_client, get_coaching_prompt

    client = get_anthropic_client()

    # Build player ID → name lookup
    name_map: dict[str, str] = {}
    for p in request.player_roster:
        name_map[p.get("id", "")] = p.get("name", p.get("id", ""))

    def resolve_name(pid: str) -> str:
        return name_map.get(pid, pid)

    snap = request.snapshot
    label = snap.get("label", "")
    time_ms = snap.get("time_ms", 0)
    players = snap.get("players", [])
    atk_alive = [p for p in players if p.get("side") == "attack" and p.get("is_alive")]
    def_alive = [p for p in players if p.get("side") == "defense" and p.get("is_alive")]

    # Build per-player details
    atk_details = "\n".join(
        f"  - {resolve_name(p.get('player_id',''))} ({p.get('agent','?')}) at ({p.get('x',0):.2f},{p.get('y',0):.2f}) HP:{p.get('health',0)}"
        for p in atk_alive
    )
    def_details = "\n".join(
        f"  - {resolve_name(p.get('player_id',''))} ({p.get('agent','?')}) at ({p.get('x',0):.2f},{p.get('y',0):.2f}) HP:{p.get('health',0)}"
        for p in def_alive
    )

    # Fog of war
    pk = snap.get("player_knowledge", {})
    fog_lines = []
    for pid, info in (pk or {}).items():
        known = info.get("known_enemies", [])
        if known:
            fog_lines.append(f"  {resolve_name(pid)} sees {len(known)} enemies: {', '.join(resolve_name(e.get('enemy_id','?')) for e in known)}")

    # Decisions
    decisions = snap.get("decisions", {})
    dec_lines = []
    for pid, d in (decisions or {}).items():
        dec_lines.append(f"  {resolve_name(pid)}: {d.get('action','?')} (utility={d.get('utility_score',0):.2f}, reason: {d.get('reason','')})")

    # Round state
    rs = snap.get("round_state", {})
    man_adv = rs.get("man_advantage", 0) if rs else 0

    prev_context = ""
    if request.previous_narrations:
        prev_context = "\nPrevious narrations (maintain continuity):\n" + "\n".join(
            f"  [{i+1}] {n}" for i, n in enumerate(request.previous_narrations[-5:])
        )

    # Roster reference for prompt
    roster_ref = ""
    if request.player_roster:
        roster_ref = "\nPlayer Roster:\n" + "\n".join(
            f"  - {p.get('name', p.get('id'))} ({p.get('agent', '?')}, {p.get('side', '?')})"
            for p in request.player_roster
        ) + "\n"

    context = f"""You are narrating a LIVE VALORANT simulation on {request.map_name}.
Attack: {request.attack_team} | Defense: {request.defense_team}
{roster_ref}
IMPORTANT: ALWAYS use real player names — NEVER internal IDs like 'c9_1' or 'g2_2'.

Current moment: [{time_ms}ms] {label}
Man advantage: {'ATK +' + str(man_adv) if man_adv > 0 else 'DEF +' + str(abs(man_adv)) if man_adv < 0 else 'Even'}
Spike: {'Planted at ' + str(snap.get('spike_site','?')) if snap.get('spike_planted') else 'Not planted'}

Attack players ({len(atk_alive)} alive):
{atk_details or '  (none alive)'}

Defense players ({len(def_alive)} alive):
{def_details or '  (none alive)'}

Fog of war (what players know):
{chr(10).join(fog_lines) if fog_lines else '  No confirmed enemy positions'}

AI Decisions this tick:
{chr(10).join(dec_lines) if dec_lines else '  No decisions recorded'}
{prev_context}

Respond with a single JSON object:
{{
  "focus": {{
    "team": "attack" or "defense",
    "players": [{{ "player_id": "...", "x": 0.3, "y": 0.2, "action": "what this player is doing and why" }}]
  }},
  "enemy_state": {{
    "players": [{{ "player_id": "...", "x": 0.5, "y": 0.4, "action": "what the enemy is doing" }}]
  }},
  "camera_target": {{ "x": 0.3, "y": 0.2, "zoom": 1.5 }},
  "narration": "2-4 sentences of tactical commentary explaining what happened and why",
  "prediction": "1 sentence prediction of what happens next with rough probability"
}}

Focus on WHY players made decisions (use the fog-of-war and utility data). Be specific about players and positions."""

    system = get_coaching_prompt("coach", context)
    messages = [{"role": "user", "content": "Narrate this moment."}]

    try:
        response = await client.chat(
            messages=messages,
            system=system,
            max_tokens=1024,
            temperature=0.7,
        )
        text = response.content[0].text if hasattr(response, 'content') and response.content else "{}"
        # Try to parse as JSON
        try:
            # Strip markdown code fences if present
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {
                "narration": text,
                "camera_target": {"x": 0.5, "y": 0.5, "zoom": 1.0},
                "focus": {"team": "attack", "players": []},
                "enemy_state": {"players": []},
                "prediction": "",
            }
        return result
    except Exception as e:
        return {
            "narration": f"Narration unavailable: {str(e)}",
            "camera_target": {"x": 0.5, "y": 0.5, "zoom": 1.0},
            "focus": {"team": "attack", "players": []},
            "enemy_state": {"players": []},
            "prediction": "",
        }


class NarrationRequest(BaseModel):
    """Request for AI narration walkthrough (Mode B: post-sim replay)."""
    session_id: str
    snapshots: list[dict]  # Auto-snapshots from completed sim
    final_state: dict  # Final sim state
    map_name: str = "ascent"
    attack_team: str = "cloud9"
    defense_team: str = "g2"
    player_roster: list[dict] = []  # [{id, name, agent, side, team}]


@router.post("/narration/stream")
async def narration_walkthrough_stream(request: NarrationRequest):
    """Stream AI narration walkthrough for a completed simulation via SSE.

    The AI narrates each key moment (snapshot) with tactical analysis,
    returning structured events that drive camera movements and text.
    """
    import json
    from ...services.llm import get_anthropic_client, get_coaching_prompt

    client = get_anthropic_client()

    # Build player ID → name lookup from roster
    name_map: dict[str, str] = {}
    for p in request.player_roster:
        name_map[p.get("id", "")] = p.get("name", p.get("id", ""))

    def resolve_name(pid: str) -> str:
        return name_map.get(pid, pid)

    # Build enriched context from snapshots
    moments = []
    for i, snap in enumerate(request.snapshots):
        label = snap.get("label", "")
        time_ms = snap.get("time_ms", 0)
        player_count = snap.get("player_count", {})
        line = f"Moment {i}: [{time_ms}ms] {label} — ATK {player_count.get('attack', '?')}v{player_count.get('defense', '?')} DEF"

        if snap.get("spike_planted"):
            line += f" | Spike at {snap.get('spike_site', '?')}"

        # Include round state context
        rs = snap.get("round_state")
        if rs:
            adv = rs.get("man_advantage", 0)
            if adv != 0:
                line += f" | {'ATK' if adv > 0 else 'DEF'} has {abs(adv)}-man advantage"
            if rs.get("trade_window_active"):
                line += " | Trade window open"

        # Include fog-of-war summary
        pk = snap.get("player_knowledge")
        if pk:
            total_known = sum(len(p.get("known_enemies", [])) for p in pk.values())
            line += f" | {total_known} enemy positions known across all players"

        # Include AI decision context
        decisions = snap.get("decisions")
        if decisions:
            actions = [f"{resolve_name(pid)}: {d.get('action', '?')} (conf={d.get('confidence', 0):.0%})" for pid, d in list(decisions.items())[:3]]
            line += f" | Decisions: {', '.join(actions)}"

        # Include player positions for camera focus
        players = snap.get("players", [])
        alive_positions = [(p.get("x", 0.5), p.get("y", 0.5), p.get("player_id", ""), p.get("side", ""), p.get("agent", "?")) for p in players if p.get("is_alive")]
        if alive_positions:
            line += f"\n  Alive players: {', '.join(f'{resolve_name(pid)}({side},{agent}) at ({x:.2f},{y:.2f})' for x,y,pid,side,agent in alive_positions)}"

        moments.append(line)

    # Build roster summary for prompt
    roster_lines = ""
    if request.player_roster:
        roster_lines = "\nPlayer Roster:\n" + "\n".join(
            f"  - {p.get('name', p.get('id'))} ({p.get('agent', '?')}, {p.get('side', '?')}, {p.get('team', '?')})"
            for p in request.player_roster
        ) + "\n"

    context = f"""You are an expert VALORANT analyst narrating a completed simulation round on {request.map_name}.
Attack: {request.attack_team} | Defense: {request.defense_team}
{roster_lines}
IMPORTANT: ALWAYS use real player names (e.g. 'Jakee', 'leaf', 'Cryocells') — NEVER use internal IDs like 'c9_1' or 'g2_2'.

Key moments (auto-snapshots at kills, spike plant, and periodic intervals):
{chr(10).join(moments)}

Final result: {json.dumps(request.final_state, default=str)[:2000]}

For EACH moment, respond with a JSON object on its own line:
{{
  "moment_index": 0,
  "focus_x": 0.5,
  "focus_y": 0.5,
  "zoom": 1.2,
  "narration": "Tactical analysis...",
  "highlight_players": ["player_id1"],
  "focus": {{
    "team": "attack",
    "players": [{{ "player_id": "...", "x": 0.3, "y": 0.2, "action": "what and why" }}]
  }},
  "enemy_state": {{
    "players": [{{ "player_id": "...", "x": 0.5, "y": 0.4, "action": "what enemy is doing" }}]
  }},
  "prediction": "1-sentence prediction with probability"
}}

Rules:
- focus_x/focus_y: normalized 0-1 coordinates of the action center. Use provided player positions.
- zoom: 1.0 for overview, 1.5-2.0 for close-ups on duels/plants
- highlight_players: player_ids the camera should highlight
- focus.players: per-player breakdown of what key players are doing AND WHY (reference fog-of-war, decisions, man-advantage)
- enemy_state.players: what the opposing team is doing at this moment
- narration: 2-4 sentences of expert tactical analysis. Explain WHY things happened using the decision/knowledge data.
- prediction: what happens next with rough % probability
- Cover every moment provided, in order."""

    async def generate():
        try:
            system = get_coaching_prompt("coach", context)
            messages = [{"role": "user", "content": "Narrate this round moment by moment. One JSON per line."}]

            full_text = ""

            async for chunk in client.stream_chat(
                messages=messages,
                system=system,
                max_tokens=4096,
                temperature=0.7,
            ):
                full_text += chunk

            # Parse complete response for JSON objects using brace balancing
            brace_depth = 0
            in_string = False
            escape_next = False
            json_start = -1
            i = 0
            while i < len(full_text):
                ch = full_text[i]
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    i += 1
                    continue
                if ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        if brace_depth == 0:
                            json_start = i
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                        if brace_depth == 0 and json_start >= 0:
                            json_str = full_text[json_start:i + 1]
                            try:
                                data = json.loads(json_str)
                                if "moment_index" in data:
                                    yield f"data: {json.dumps({'type': 'moment', 'data': data})}\n\n"
                            except json.JSONDecodeError:
                                pass
                            json_start = -1
                i += 1

            yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class WhatIfExplainRequest(BaseModel):
    """Request for AI what-if explanation."""
    original: dict
    what_if: dict
    modifications: dict
    map_name: str = "ascent"


@router.post("/what-if/explain")
async def explain_what_if(request: WhatIfExplainRequest):
    """Get AI explanation of what-if comparison results."""
    import json
    from ...services.llm import get_anthropic_client, get_coaching_prompt

    client = get_anthropic_client()

    context = f"""Analyze this what-if comparison on {request.map_name}:

ORIGINAL OUTCOME:
- Winner: {request.original.get("winner")}
- Attack alive: {request.original.get("attack_alive")}, Defense alive: {request.original.get("defense_alive")}
- Spike planted: {request.original.get("spike_planted")}

WHAT-IF OUTCOME (with modifications):
- Winner: {request.what_if.get("winner")}
- Attack alive: {request.what_if.get("attack_alive")}, Defense alive: {request.what_if.get("defense_alive")}
- Spike planted: {request.what_if.get("spike_planted")}

MODIFICATIONS APPLIED:
{json.dumps(request.modifications, indent=2)}

Provide a concise tactical analysis explaining:
1. Why the outcome changed (or didn't)
2. Key tactical implications
3. What this tells the coach about this scenario"""

    system = get_coaching_prompt("coach")
    messages = [{"role": "user", "content": context}]

    response = await client.chat(
        messages=messages,
        system=system,
        max_tokens=1024,
        temperature=0.7,
    )

    # Extract text from response
    if hasattr(response, "content") and response.content:
        text = response.content[0].text if hasattr(response.content[0], "text") else str(response.content[0])
    else:
        text = "Unable to generate explanation."

    return {"explanation": text}
