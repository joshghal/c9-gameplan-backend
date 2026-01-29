"""Scouting report generator using AI analysis."""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from ..llm import get_anthropic_client, ContextBuilder, get_coaching_prompt


@dataclass
class ScoutingReport:
    """A generated scouting report."""
    team_name: str
    map_name: Optional[str]
    generated_at: datetime
    expires_at: datetime
    report_data: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "map_name": self.map_name,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "report": self.report_data,
        }


class ScoutingReportGenerator:
    """Generates AI-powered scouting reports."""

    def __init__(self):
        self.client = get_anthropic_client()
        self.context_builder = ContextBuilder()
        self._cache: dict[str, ScoutingReport] = {}

    def _cache_key(self, team_name: str, map_name: Optional[str]) -> str:
        return f"{team_name}:{map_name or 'all'}"

    async def generate_report(
        self,
        team_name: str,
        map_name: Optional[str] = None,
        force_refresh: bool = False,
        cache_hours: int = 24,
    ) -> ScoutingReport:
        """Generate a scouting report for a team.

        Args:
            team_name: Target team to scout
            map_name: Optional specific map focus
            force_refresh: Ignore cache and regenerate
            cache_hours: How long to cache the report

        Returns:
            ScoutingReport with analysis data
        """
        cache_key = self._cache_key(team_name, map_name)

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired:
                return cached

        # Build context from pattern data
        context = self.context_builder.build_scouting_context(
            team_name=team_name,
            map_name=map_name,
        )

        # Generate report
        prompt = self._build_report_prompt(team_name, map_name, context)
        system = get_coaching_prompt("scouting_report")

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=4096,
        )

        report_text = response.content[0].text

        # Parse structured sections
        report_data = self._parse_report(report_text)

        report = ScoutingReport(
            team_name=team_name,
            map_name=map_name,
            generated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=cache_hours),
            report_data=report_data,
        )

        # Cache
        self._cache[cache_key] = report

        return report

    def _build_report_prompt(
        self,
        team_name: str,
        map_name: Optional[str],
        context: str,
    ) -> str:
        """Build the prompt for report generation."""
        map_focus = f" on {map_name}" if map_name else ""

        return f"""Generate a comprehensive scouting report for {team_name}{map_focus}.

Available Data:
{context}

Structure your report with clear sections:
1. Executive Summary
2. Attack Tendencies
3. Defense Tendencies
4. Key Players
5. Economic Patterns
6. Recommended Counters

Use specific data points when available. Be analytical and actionable."""

    def _parse_report(self, report_text: str) -> dict:
        """Parse report text into structured sections."""
        sections = {
            "raw": report_text,
            "executive_summary": "",
            "attack_tendencies": "",
            "defense_tendencies": "",
            "key_players": "",
            "economic_patterns": "",
            "recommended_counters": "",
        }

        current_section = "raw"
        section_mapping = {
            "executive summary": "executive_summary",
            "attack tendencies": "attack_tendencies",
            "attack": "attack_tendencies",
            "defense tendencies": "defense_tendencies",
            "defense": "defense_tendencies",
            "key players": "key_players",
            "players": "key_players",
            "economic patterns": "economic_patterns",
            "economy": "economic_patterns",
            "recommended counters": "recommended_counters",
            "counters": "recommended_counters",
            "recommendations": "recommended_counters",
        }

        lines = report_text.split("\n")

        for line in lines:
            # Check for section headers
            lower_line = line.lower().strip()

            # Remove markdown formatting
            clean_line = lower_line.lstrip("#").lstrip("*").strip()

            for key, section_name in section_mapping.items():
                if key in clean_line and len(clean_line) < 40:
                    current_section = section_name
                    break
            else:
                if current_section != "raw":
                    sections[current_section] += line + "\n"

        # Clean up sections
        for key in sections:
            if key != "raw":
                sections[key] = sections[key].strip()

        return sections

    async def generate_quick_summary(
        self,
        team_name: str,
        map_name: str,
    ) -> str:
        """Generate a quick 2-3 sentence summary for a team on a map."""
        context = self.context_builder.build_coaching_context(
            query="quick summary",
            team_name=team_name,
            map_name=map_name,
        )

        prompt = f"""In 2-3 sentences, summarize {team_name}'s key tendencies on {map_name}.
Focus on their most exploitable patterns.

Data:
{context}"""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system="You are a concise VALORANT analyst. Be brief but insightful.",
            max_tokens=200,
        )

        return response.content[0].text

    def get_cached_report(
        self,
        team_name: str,
        map_name: Optional[str] = None,
    ) -> Optional[ScoutingReport]:
        """Get a cached report if available and not expired."""
        cache_key = self._cache_key(team_name, map_name)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired:
                return cached

        return None

    def clear_cache(self, team_name: Optional[str] = None):
        """Clear cached reports."""
        if team_name:
            keys_to_remove = [
                k for k in self._cache.keys()
                if k.startswith(team_name)
            ]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


# Singleton instance
_generator: Optional[ScoutingReportGenerator] = None


def get_scouting_generator() -> ScoutingReportGenerator:
    """Get singleton scouting report generator."""
    global _generator
    if _generator is None:
        _generator = ScoutingReportGenerator()
    return _generator
