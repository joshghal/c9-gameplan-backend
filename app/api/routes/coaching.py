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
