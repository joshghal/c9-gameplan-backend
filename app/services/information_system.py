"""
Information System - Fog of War and Player Knowledge

This system models what each player KNOWS about the game state,
rather than giving all players perfect information.

Key Concepts:
1. Players only know about enemies they've seen or heard
2. Information decays over time (positions become stale)
3. AI decisions are based on knowledge, not global state
4. This creates EMERGENT behavior rather than tuned probabilities

Sources of Information:
- Direct vision (LOS + facing direction within FOV)
- Sound cues (footsteps, gunfire, abilities)
- Teammate callouts (shared info)
- Abilities (recon dart, camera, tripwire triggered)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class InfoSource(Enum):
    """How the player learned about an enemy."""
    VISION = "vision"           # Direct line of sight
    SOUND_FOOTSTEP = "footstep" # Heard running
    SOUND_GUNFIRE = "gunfire"   # Heard shooting
    SOUND_ABILITY = "ability"   # Heard ability use
    SOUND_SPIKE = "spike"       # Heard plant/defuse
    CALLOUT = "callout"         # Teammate shared info
    ABILITY_REVEAL = "reveal"   # Recon dart, camera, etc.
    DAMAGE = "damage"           # Got shot from direction


class InfoConfidence(Enum):
    """How confident the player is about the information."""
    EXACT = "exact"         # Currently seeing enemy
    HIGH = "high"           # Saw within 2 seconds
    MEDIUM = "medium"       # Saw 2-10 seconds ago
    LOW = "low"             # Saw 10-30 seconds ago
    STALE = "stale"         # Older than 30 seconds


@dataclass
class EnemyInfo:
    """Information a player has about a specific enemy."""
    enemy_id: str
    last_known_x: float
    last_known_y: float
    last_seen_ms: int
    source: InfoSource
    confidence: InfoConfidence = InfoConfidence.MEDIUM
    is_alive: bool = True  # Assume alive unless confirmed dead

    # Movement prediction
    last_known_direction: Optional[float] = None  # Radians
    was_running: bool = False

    def update_confidence(self, current_time_ms: int) -> InfoConfidence:
        """Update confidence based on time since last seen."""
        age_ms = current_time_ms - self.last_seen_ms

        if age_ms < 500:
            self.confidence = InfoConfidence.EXACT
        elif age_ms < 2000:
            self.confidence = InfoConfidence.HIGH
        elif age_ms < 10000:
            self.confidence = InfoConfidence.MEDIUM
        elif age_ms < 30000:
            self.confidence = InfoConfidence.LOW
        else:
            self.confidence = InfoConfidence.STALE

        return self.confidence

    def predict_current_position(self, current_time_ms: int) -> Tuple[float, float]:
        """Predict where enemy might be now based on last known info."""
        age_s = (current_time_ms - self.last_seen_ms) / 1000.0

        if age_s < 0.5 or self.last_known_direction is None:
            return (self.last_known_x, self.last_known_y)

        # Estimate movement speed (normalized units per second)
        speed = 0.05 if self.was_running else 0.025  # Running vs walking

        # Cap prediction to reasonable distance
        max_distance = min(age_s * speed, 0.3)  # Max 30% of map

        # Predict position along last known direction
        pred_x = self.last_known_x + math.cos(self.last_known_direction) * max_distance
        pred_y = self.last_known_y + math.sin(self.last_known_direction) * max_distance

        # Clamp to map bounds
        pred_x = max(0.0, min(1.0, pred_x))
        pred_y = max(0.0, min(1.0, pred_y))

        return (pred_x, pred_y)


@dataclass
class PlayerKnowledge:
    """Everything a player knows about the game state."""
    player_id: str
    team: str  # 'attack' or 'defense'

    # Known enemy information
    enemies: Dict[str, EnemyInfo] = field(default_factory=dict)

    # Confirmed kills (enemies we KNOW are dead)
    confirmed_dead: Set[str] = field(default_factory=set)

    # Spike knowledge
    knows_spike_planted: bool = False
    known_spike_site: Optional[str] = None
    spike_plant_time_ms: Optional[int] = None

    # Teammate positions (always known via comms)
    teammate_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Areas cleared (recently checked, no enemies)
    cleared_areas: Dict[str, int] = field(default_factory=dict)  # area -> cleared_time_ms

    def update_all_confidence(self, current_time_ms: int):
        """Update confidence levels for all known enemies."""
        for enemy_info in self.enemies.values():
            enemy_info.update_confidence(current_time_ms)

    def see_enemy(self, enemy_id: str, x: float, y: float, time_ms: int,
                  direction: Optional[float] = None, is_running: bool = False):
        """Update knowledge from directly seeing an enemy."""
        self.enemies[enemy_id] = EnemyInfo(
            enemy_id=enemy_id,
            last_known_x=x,
            last_known_y=y,
            last_seen_ms=time_ms,
            source=InfoSource.VISION,
            confidence=InfoConfidence.EXACT,
            last_known_direction=direction,
            was_running=is_running
        )

    def hear_enemy(self, x: float, y: float, time_ms: int,
                   source: InfoSource, enemy_id: Optional[str] = None):
        """Update knowledge from hearing a sound."""
        # Sound gives approximate position, not exact
        # Add some uncertainty
        uncertainty = 0.05  # 5% of map
        approx_x = x + (hash(f"{time_ms}_x") % 100 - 50) / 1000 * uncertainty
        approx_y = y + (hash(f"{time_ms}_y") % 100 - 50) / 1000 * uncertainty

        # If we don't know which enemy, create a generic entry
        if enemy_id is None:
            enemy_id = f"unknown_{int(x*100)}_{int(y*100)}"

        # Only update if this is newer info or we don't have info on this enemy
        if enemy_id not in self.enemies or \
           time_ms > self.enemies[enemy_id].last_seen_ms:
            self.enemies[enemy_id] = EnemyInfo(
                enemy_id=enemy_id,
                last_known_x=approx_x,
                last_known_y=approx_y,
                last_seen_ms=time_ms,
                source=source,
                confidence=InfoConfidence.MEDIUM,
                was_running=(source == InfoSource.SOUND_FOOTSTEP)
            )

    def receive_callout(self, enemy_id: str, x: float, y: float,
                        time_ms: int, original_time_ms: int):
        """Receive information from a teammate callout."""
        # Ignore callouts about confirmed dead enemies
        if enemy_id in self.confirmed_dead:
            return
        # Callouts have delay and uncertainty
        if enemy_id not in self.enemies or \
           original_time_ms > self.enemies[enemy_id].last_seen_ms:
            self.enemies[enemy_id] = EnemyInfo(
                enemy_id=enemy_id,
                last_known_x=x,
                last_known_y=y,
                last_seen_ms=original_time_ms,  # Use original sighting time
                source=InfoSource.CALLOUT,
                confidence=InfoConfidence.MEDIUM
            )

    def confirm_kill(self, enemy_id: str):
        """Confirm an enemy is dead."""
        self.confirmed_dead.add(enemy_id)
        if enemy_id in self.enemies:
            self.enemies[enemy_id].is_alive = False

    def clear_area(self, area_name: str, time_ms: int):
        """Mark an area as cleared (checked, no enemies)."""
        self.cleared_areas[area_name] = time_ms

    def is_area_cleared(self, area_name: str, time_ms: int,
                        clear_duration_ms: int = 15000) -> bool:
        """Check if an area was recently cleared."""
        if area_name not in self.cleared_areas:
            return False
        return (time_ms - self.cleared_areas[area_name]) < clear_duration_ms

    def get_known_enemy_count(self) -> int:
        """Get count of enemies known to be alive."""
        return len([e for e in self.enemies.values()
                   if e.is_alive and e.enemy_id not in self.confirmed_dead])

    def get_enemies_near(self, x: float, y: float, radius: float,
                         time_ms: int) -> List[EnemyInfo]:
        """Get known enemies near a position (using predicted positions)."""
        nearby = []
        for enemy in self.enemies.values():
            if not enemy.is_alive or enemy.enemy_id in self.confirmed_dead:
                continue

            pred_x, pred_y = enemy.predict_current_position(time_ms)
            dist = math.sqrt((pred_x - x)**2 + (pred_y - y)**2)

            if dist < radius:
                nearby.append(enemy)

        return nearby

    def get_site_threat_level(self, site_x: float, site_y: float,
                              site_radius: float, time_ms: int) -> float:
        """
        Calculate threat level at a site based on known enemy positions.

        Returns 0.0 (safe) to 1.0 (very dangerous)
        """
        enemies_near = self.get_enemies_near(site_x, site_y, site_radius * 3, time_ms)

        if not enemies_near:
            return 0.0

        threat = 0.0
        for enemy in enemies_near:
            pred_x, pred_y = enemy.predict_current_position(time_ms)
            dist = math.sqrt((pred_x - site_x)**2 + (pred_y - site_y)**2)

            # Closer = more threat
            distance_factor = max(0, 1 - dist / (site_radius * 3))

            # Fresher info = more threat (we're more sure they're there)
            confidence_factor = {
                InfoConfidence.EXACT: 1.0,
                InfoConfidence.HIGH: 0.9,
                InfoConfidence.MEDIUM: 0.6,
                InfoConfidence.LOW: 0.3,
                InfoConfidence.STALE: 0.1
            }.get(enemy.confidence, 0.5)

            threat += distance_factor * confidence_factor

        return min(1.0, threat)


class InformationManager:
    """
    Manages information flow between players.

    Handles:
    - Vision checks (can player A see player B?)
    - Sound propagation (who hears what?)
    - Callout system (sharing info with teammates)
    - Knowledge updates each tick
    """

    # Vision parameters
    VISION_RANGE = 0.50  # 50% of map max vision
    VISION_FOV = math.pi * 0.6  # ~108 degree field of view

    # Sound parameters
    SOUND_RANGE_RUNNING = 0.40    # Running footsteps
    SOUND_RANGE_WALKING = 0.0    # Walking is silent
    SOUND_RANGE_GUNFIRE = 0.60   # Gunfire
    SOUND_RANGE_ABILITY = 0.50   # Most abilities
    SOUND_RANGE_SPIKE = 0.80     # Spike plant/defuse

    # Callout parameters
    CALLOUT_DELAY_MS = 500  # Time to communicate info to team

    def __init__(self):
        self.player_knowledge: Dict[str, PlayerKnowledge] = {}
        self.pending_callouts: List[Dict] = []

    def initialize_player(self, player_id: str, team: str):
        """Initialize knowledge state for a player."""
        self.player_knowledge[player_id] = PlayerKnowledge(
            player_id=player_id,
            team=team
        )

    def reset(self):
        """Reset all knowledge for new round."""
        self.player_knowledge.clear()
        self.pending_callouts.clear()

    def get_knowledge(self, player_id: str) -> Optional[PlayerKnowledge]:
        """Get a player's current knowledge state."""
        return self.player_knowledge.get(player_id)

    def update_vision(self, observer_id: str, observer_pos: Tuple[float, float],
                      observer_facing: float, targets: List[Dict], time_ms: int,
                      smoke_positions: List[Tuple[float, float]] = None):
        """
        Update observer's knowledge based on what they can see.

        Args:
            observer_id: Player doing the looking
            observer_pos: (x, y) position
            observer_facing: Direction facing in radians
            targets: List of {id, x, y, team, is_running} for potential targets
            time_ms: Current time
            smoke_positions: List of active smoke positions that block vision
        """
        knowledge = self.player_knowledge.get(observer_id)
        if not knowledge:
            return

        smoke_positions = smoke_positions or []

        for target in targets:
            # Skip teammates
            if target.get('team') == knowledge.team:
                # But update teammate positions
                knowledge.teammate_positions[target['id']] = (target['x'], target['y'])
                continue

            # Check if target is visible
            if self._can_see(observer_pos, observer_facing,
                           (target['x'], target['y']), smoke_positions):

                # Calculate target's facing direction if available
                target_dir = target.get('facing', None)

                knowledge.see_enemy(
                    enemy_id=target['id'],
                    x=target['x'],
                    y=target['y'],
                    time_ms=time_ms,
                    direction=target_dir,
                    is_running=target.get('is_running', False)
                )

                # Queue callout to teammates
                self._queue_callout(observer_id, target['id'],
                                   target['x'], target['y'], time_ms)

    def _can_see(self, observer_pos: Tuple[float, float],
                 observer_facing: float,
                 target_pos: Tuple[float, float],
                 smoke_positions: List[Tuple[float, float]]) -> bool:
        """Check if observer can see target."""
        ox, oy = observer_pos
        tx, ty = target_pos

        # Distance check
        dist = math.sqrt((tx - ox)**2 + (ty - oy)**2)
        if dist > self.VISION_RANGE:
            return False

        # FOV check
        angle_to_target = math.atan2(ty - oy, tx - ox)
        angle_diff = abs(self._normalize_angle(angle_to_target - observer_facing))
        if angle_diff > self.VISION_FOV / 2:
            return False

        # Smoke check
        for sx, sy in smoke_positions:
            if self._line_intersects_circle(ox, oy, tx, ty, sx, sy, 0.05):
                return False

        return True

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _line_intersects_circle(self, x1: float, y1: float, x2: float, y2: float,
                                 cx: float, cy: float, r: float) -> bool:
        """Check if line segment intersects circle (smoke)."""
        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1

        # Vector from start to circle center
        fx = x1 - cx
        fy = y1 - cy

        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return False

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if intersection is within line segment
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def propagate_sound(self, source_pos: Tuple[float, float],
                       sound_type: InfoSource, source_team: str,
                       all_players: List[Dict], time_ms: int,
                       source_id: Optional[str] = None):
        """
        Propagate a sound to all players who can hear it.

        Args:
            source_pos: Where the sound originated
            sound_type: Type of sound (footstep, gunfire, etc.)
            source_team: Team that made the sound
            all_players: List of all players with positions
            time_ms: Current time
            source_id: ID of player who made sound (if known)
        """
        # Determine sound range based on type
        sound_range = {
            InfoSource.SOUND_FOOTSTEP: self.SOUND_RANGE_RUNNING,
            InfoSource.SOUND_GUNFIRE: self.SOUND_RANGE_GUNFIRE,
            InfoSource.SOUND_ABILITY: self.SOUND_RANGE_ABILITY,
            InfoSource.SOUND_SPIKE: self.SOUND_RANGE_SPIKE,
        }.get(sound_type, self.SOUND_RANGE_ABILITY)

        if sound_range == 0:
            return  # Silent action

        sx, sy = source_pos

        for player in all_players:
            # Don't hear your own team's sounds as enemy info
            if player.get('team') == source_team:
                continue

            if not player.get('is_alive', True):
                continue

            px, py = player['x'], player['y']
            dist = math.sqrt((px - sx)**2 + (py - sy)**2)

            if dist <= sound_range:
                knowledge = self.player_knowledge.get(player['id'])
                if knowledge:
                    knowledge.hear_enemy(sx, sy, time_ms, sound_type, source_id)

    def _queue_callout(self, caller_id: str, enemy_id: str,
                       x: float, y: float, time_ms: int):
        """Queue a callout to be delivered to teammates."""
        knowledge = self.player_knowledge.get(caller_id)
        if not knowledge:
            return

        self.pending_callouts.append({
            'caller_id': caller_id,
            'caller_team': knowledge.team,
            'enemy_id': enemy_id,
            'x': x,
            'y': y,
            'spotted_time': time_ms,
            'deliver_time': time_ms + self.CALLOUT_DELAY_MS
        })

    def process_callouts(self, time_ms: int):
        """Process pending callouts that are ready to be delivered."""
        delivered = []

        for callout in self.pending_callouts:
            if time_ms >= callout['deliver_time']:
                # Deliver to all teammates
                for pid, knowledge in self.player_knowledge.items():
                    if knowledge.team == callout['caller_team'] and \
                       pid != callout['caller_id']:
                        knowledge.receive_callout(
                            enemy_id=callout['enemy_id'],
                            x=callout['x'],
                            y=callout['y'],
                            time_ms=time_ms,
                            original_time_ms=callout['spotted_time']
                        )
                delivered.append(callout)

        # Remove delivered callouts
        for callout in delivered:
            self.pending_callouts.remove(callout)

    def notify_kill(self, killer_id: str, victim_id: str,
                    victim_team: str, kill_pos: Tuple[float, float],
                    time_ms: int):
        """Notify all players about a kill. Victim's team gets killer position callout."""
        # Everyone knows enemy is dead (kill feed)
        for pid, knowledge in self.player_knowledge.items():
            knowledge.confirm_kill(victim_id)

        # Death callout: victim's surviving teammates learn killer position
        for pid, knowledge in self.player_knowledge.items():
            if knowledge.team == victim_team and pid != victim_id:
                self.pending_callouts.append({
                    'caller_id': victim_id,
                    'caller_team': victim_team,
                    'enemy_id': killer_id,
                    'x': kill_pos[0],
                    'y': kill_pos[1],
                    'spotted_time': time_ms,
                    'deliver_time': time_ms + self.CALLOUT_DELAY_MS
                })

    def notify_spike_plant(self, site: str, plant_pos: Tuple[float, float],
                          time_ms: int):
        """Notify all players that spike was planted."""
        # Everyone hears the global spike plant sound
        for knowledge in self.player_knowledge.values():
            knowledge.knows_spike_planted = True
            knowledge.known_spike_site = site
            knowledge.spike_plant_time_ms = time_ms

    def update_tick(self, time_ms: int):
        """Update information state each tick."""
        # Update confidence decay
        for knowledge in self.player_knowledge.values():
            knowledge.update_all_confidence(time_ms)

        # Process pending callouts
        self.process_callouts(time_ms)
