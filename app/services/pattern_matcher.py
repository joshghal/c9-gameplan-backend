"""Pattern matching service for movement analysis.

Matches current game state against learned movement patterns
and predicts likely next positions.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from ..models import MovementPattern, PlayerTendency


@dataclass
class Waypoint:
    """A single waypoint in a movement pattern."""
    timestamp_ms: int
    x: float
    y: float
    variance_x: float = 0.02
    variance_y: float = 0.02


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    pattern_id: int
    pattern_name: str
    confidence: float
    current_waypoint_index: int
    next_position: Tuple[float, float]
    remaining_waypoints: List[Waypoint]


@dataclass
class PositionPrediction:
    """Predicted position for a player."""
    player_id: str
    position: Tuple[float, float]
    confidence: float
    source: str  # 'pattern', 'tendency', 'interpolation'
    pattern_match: Optional[PatternMatch] = None


class PatternMatcher:
    """Matches current positions against learned movement patterns.

    Uses a combination of:
    1. Team-specific patterns (highest priority)
    2. Global patterns for the map/side
    3. Player tendencies
    4. Statistical interpolation
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._pattern_cache: Dict[str, List[MovementPattern]] = {}
        self._tendency_cache: Dict[str, PlayerTendency] = {}

    async def get_patterns(
        self,
        team_id: Optional[str],
        map_name: str,
        side: str,
        phase: str
    ) -> List[MovementPattern]:
        """Get relevant patterns for the given context."""
        cache_key = f"{team_id}:{map_name}:{side}:{phase}"

        if cache_key not in self._pattern_cache:
            query = select(MovementPattern).where(
                and_(
                    MovementPattern.is_active == True,
                    MovementPattern.map_name == map_name,
                    MovementPattern.side == side,
                    MovementPattern.phase == phase,
                    or_(
                        MovementPattern.team_id == team_id,
                        MovementPattern.team_id.is_(None)
                    )
                )
            ).order_by(
                MovementPattern.team_id.is_(None),  # Team-specific first
                MovementPattern.frequency.desc()
            ).limit(20)

            result = await self.db.execute(query)
            self._pattern_cache[cache_key] = list(result.scalars().all())

        return self._pattern_cache[cache_key]

    async def get_player_tendency(
        self,
        player_id: str,
        map_name: str,
        side: str
    ) -> Optional[PlayerTendency]:
        """Get tendency data for a player."""
        cache_key = f"{player_id}:{map_name}:{side}"

        if cache_key not in self._tendency_cache:
            # Try map+side specific first
            query = select(PlayerTendency).where(
                and_(
                    PlayerTendency.player_id == player_id,
                    PlayerTendency.map_name == map_name,
                    PlayerTendency.side == side
                )
            )
            result = await self.db.execute(query)
            tendency = result.scalar_one_or_none()

            # Fall back to map-only
            if not tendency:
                query = select(PlayerTendency).where(
                    and_(
                        PlayerTendency.player_id == player_id,
                        PlayerTendency.map_name == map_name,
                        PlayerTendency.side.is_(None)
                    )
                )
                result = await self.db.execute(query)
                tendency = result.scalar_one_or_none()

            # Fall back to global
            if not tendency:
                query = select(PlayerTendency).where(
                    and_(
                        PlayerTendency.player_id == player_id,
                        PlayerTendency.map_name.is_(None)
                    )
                )
                result = await self.db.execute(query)
                tendency = result.scalar_one_or_none()

            self._tendency_cache[cache_key] = tendency

        return self._tendency_cache[cache_key]

    def match_pattern(
        self,
        current_position: Tuple[float, float],
        current_time_ms: int,
        pattern: MovementPattern
    ) -> Optional[PatternMatch]:
        """Match current position against a pattern.

        Returns PatternMatch if position is close to expected waypoint.
        """
        waypoints = pattern.waypoints
        if not waypoints:
            return None

        # Find the closest waypoint by time
        best_idx = 0
        best_time_diff = float('inf')

        for i, wp in enumerate(waypoints):
            time_diff = abs(wp.get('timestamp_ms', 0) - current_time_ms)
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_idx = i

        expected_wp = waypoints[best_idx]
        expected_x = expected_wp.get('x', 0)
        expected_y = expected_wp.get('y', 0)
        variance_x = expected_wp.get('variance_x', 0.05)
        variance_y = expected_wp.get('variance_y', 0.05)

        # Calculate distance from expected position
        dx = current_position[0] - expected_x
        dy = current_position[1] - expected_y

        # Normalized distance (accounting for variance)
        norm_dist = math.sqrt((dx / variance_x) ** 2 + (dy / variance_y) ** 2)

        # Confidence based on distance (Gaussian-like)
        confidence = math.exp(-0.5 * norm_dist ** 2)

        # Need at least 0.3 confidence to consider it a match
        if confidence < 0.3:
            return None

        # Get next waypoint
        next_idx = min(best_idx + 1, len(waypoints) - 1)
        next_wp = waypoints[next_idx]

        # Add some random variance to next position
        next_x = next_wp.get('x', 0) + random.gauss(0, next_wp.get('variance_x', 0.02))
        next_y = next_wp.get('y', 0) + random.gauss(0, next_wp.get('variance_y', 0.02))

        # Remaining waypoints
        remaining = [
            Waypoint(
                timestamp_ms=wp.get('timestamp_ms', 0),
                x=wp.get('x', 0),
                y=wp.get('y', 0),
                variance_x=wp.get('variance_x', 0.02),
                variance_y=wp.get('variance_y', 0.02)
            )
            for wp in waypoints[next_idx:]
        ]

        return PatternMatch(
            pattern_id=pattern.id,
            pattern_name=pattern.pattern_name,
            confidence=confidence * pattern.frequency,  # Weight by pattern frequency
            current_waypoint_index=best_idx,
            next_position=(next_x, next_y),
            remaining_waypoints=remaining
        )

    async def predict_position(
        self,
        player_id: str,
        team_id: str,
        current_position: Tuple[float, float],
        current_time_ms: int,
        map_name: str,
        side: str,
        phase: str
    ) -> PositionPrediction:
        """Predict next position for a player based on patterns and tendencies."""

        # 1. Try pattern matching first
        patterns = await self.get_patterns(team_id, map_name, side, phase)

        best_match: Optional[PatternMatch] = None
        for pattern in patterns:
            match = self.match_pattern(current_position, current_time_ms, pattern)
            if match and (best_match is None or match.confidence > best_match.confidence):
                best_match = match

        if best_match and best_match.confidence > 0.5:
            return PositionPrediction(
                player_id=player_id,
                position=best_match.next_position,
                confidence=best_match.confidence,
                source='pattern',
                pattern_match=best_match
            )

        # 2. Fall back to player tendency
        tendency = await self.get_player_tendency(player_id, map_name, side)
        if tendency and tendency.common_positions:
            # Find closest common position
            common_positions = tendency.common_positions
            if common_positions:
                # Weight by frequency and distance
                best_pos = None
                best_score = 0

                for pos in common_positions:
                    px, py = pos.get('x', 0), pos.get('y', 0)
                    freq = pos.get('frequency', 0.1)
                    dist = math.sqrt(
                        (current_position[0] - px) ** 2 +
                        (current_position[1] - py) ** 2
                    )
                    # Prefer positions that are close but not the same
                    if 0.02 < dist < 0.2:
                        score = freq / (dist + 0.1)
                        if score > best_score:
                            best_score = score
                            best_pos = (px, py)

                if best_pos:
                    # Move towards common position
                    dx = best_pos[0] - current_position[0]
                    dy = best_pos[1] - current_position[1]
                    move_factor = 0.1  # Move 10% towards the position
                    next_pos = (
                        current_position[0] + dx * move_factor,
                        current_position[1] + dy * move_factor
                    )
                    return PositionPrediction(
                        player_id=player_id,
                        position=next_pos,
                        confidence=0.4,
                        source='tendency'
                    )

        # 3. Fall back to statistical interpolation
        # Simple random walk with drift towards center
        drift_x = (0.5 - current_position[0]) * 0.01
        drift_y = (0.5 - current_position[1]) * 0.01
        noise_x = random.gauss(0, 0.02)
        noise_y = random.gauss(0, 0.02)

        next_pos = (
            max(0, min(1, current_position[0] + drift_x + noise_x)),
            max(0, min(1, current_position[1] + drift_y + noise_y))
        )

        return PositionPrediction(
            player_id=player_id,
            position=next_pos,
            confidence=0.2,
            source='interpolation'
        )

    async def predict_team_positions(
        self,
        team_players: List[Dict[str, Any]],
        team_id: str,
        current_time_ms: int,
        map_name: str,
        side: str,
        phase: str
    ) -> List[PositionPrediction]:
        """Predict positions for all players on a team."""
        predictions = []

        for player in team_players:
            if not player.get('is_alive', True):
                continue

            prediction = await self.predict_position(
                player_id=player['player_id'],
                team_id=team_id,
                current_position=(player['x'], player['y']),
                current_time_ms=current_time_ms,
                map_name=map_name,
                side=side,
                phase=phase
            )
            predictions.append(prediction)

        return predictions

    def select_pattern_by_frequency(
        self,
        patterns: List[MovementPattern]
    ) -> Optional[MovementPattern]:
        """Select a pattern randomly weighted by frequency."""
        if not patterns:
            return None

        total_freq = sum(p.frequency for p in patterns)
        if total_freq == 0:
            return random.choice(patterns)

        # Weighted random selection
        r = random.random() * total_freq
        cumulative = 0
        for pattern in patterns:
            cumulative += pattern.frequency
            if r <= cumulative:
                return pattern

        return patterns[-1]

    def clear_cache(self):
        """Clear pattern and tendency caches."""
        self._pattern_cache.clear()
        self._tendency_cache.clear()
