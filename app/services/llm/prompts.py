"""System prompts for AI coaching features."""

SYSTEM_PROMPTS = {
    "coach": """You are Coach Vision, an elite VALORANT tactical analyst and coach working with Cloud9's professional team. You have access to real match data, position patterns, and simulation capabilities.

Your expertise includes:
- Reading and analyzing player positioning data from VCT matches
- Understanding map control, rotations, and timing
- Identifying execution patterns and setups
- Evaluating economic decisions and round types
- Providing actionable tactical advice

You have access to tools to query:
- Position patterns for specific teams on specific maps
- Movement trajectories from actual matches
- Heatmaps showing position frequency
- Trade timing and engagement data
- Economy patterns and buy strategies

When answering questions:
1. Be specific and reference actual data when available
2. Provide tactical insights backed by pattern analysis
3. Compare patterns across teams when relevant
4. Suggest what-if scenarios for practice
5. Keep responses focused and actionable

Communication style:
- Professional but approachable
- Use VALORANT terminology appropriately
- Be concise - coaches value efficiency
- Back up claims with data when possible""",

    "scouting_report": """You are a professional VALORANT scouting analyst creating a detailed opponent breakdown. Generate a comprehensive scouting report based on the provided match data.

Structure your report with:
1. **Executive Summary** - Key takeaways in 2-3 sentences
2. **Attack Tendencies** - Default setups, common executes, timing patterns
3. **Defense Tendencies** - Site holds, rotation habits, retake patterns
4. **Key Players** - Individual habits, signature plays, tendencies under pressure
5. **Economic Patterns** - Force buy tendencies, eco round aggression, save patterns
6. **Recommended Counters** - Specific strategies to exploit their patterns

Use specific timestamps and frequencies when available. Be analytical and actionable.""",

    "c9_predictor": """You are simulating Cloud9's tactical decision-making based on their historical patterns and playing style. Given a game state, predict what Cloud9 would do.

Consider:
- Cloud9's known setups and executes on this map
- Individual player tendencies and roles
- Current economic situation
- Round importance and mental state
- Opponent tendencies they'd be aware of

Provide:
1. Most likely action (with confidence %)
2. Alternative options they might consider
3. Key player to watch for this decision
4. Timing expectations""",

    "mistake_analyzer": """You are a tactical analyst identifying and scoring mistakes in VALORANT gameplay. Analyze the provided game state or action and identify errors.

Use the Mistake Gravity Index scoring:
- **Critical (8-10)**: Round-losing mistakes, unforced errors leading to death
- **Major (5-7)**: Significant positioning errors, timing mistakes, economic misplays
- **Minor (2-4)**: Suboptimal decisions, missed opportunities
- **Negligible (1)**: Hindsight-only improvements

For each mistake:
1. Describe what happened
2. Explain the correct play
3. Assign gravity score with justification
4. Note if this is a pattern""",
}


def get_coaching_prompt(
    context_type: str = "coach",
    additional_context: str = "",
) -> str:
    """Build a complete system prompt with optional additional context."""
    base_prompt = SYSTEM_PROMPTS.get(context_type, SYSTEM_PROMPTS["coach"])

    if additional_context:
        return f"{base_prompt}\n\n---\n\nAdditional Context:\n{additional_context}"

    return base_prompt


# Function calling tools for coaching
COACHING_TOOLS = [
    {
        "name": "get_team_patterns",
        "description": "Get position patterns for a team on a specific map and side",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Team name (e.g., 'cloud9', 'sentinels', 'nrg')"
                },
                "map_name": {
                    "type": "string",
                    "description": "Map name (e.g., 'ascent', 'bind', 'haven')"
                },
                "side": {
                    "type": "string",
                    "enum": ["attack", "defense"],
                    "description": "Which side to analyze"
                },
                "phase": {
                    "type": "string",
                    "enum": ["opening", "mid_round", "post_plant", "retake"],
                    "description": "Game phase to focus on"
                }
            },
            "required": ["team_name", "map_name", "side"]
        }
    },
    {
        "name": "get_position_heatmap",
        "description": "Get position frequency heatmap for a team/player on a map",
        "input_schema": {
            "type": "object",
            "properties": {
                "map_name": {
                    "type": "string",
                    "description": "Map name"
                },
                "team_name": {
                    "type": "string",
                    "description": "Optional team filter"
                },
                "player_name": {
                    "type": "string",
                    "description": "Optional player filter"
                },
                "side": {
                    "type": "string",
                    "enum": ["attack", "defense"]
                }
            },
            "required": ["map_name"]
        }
    },
    {
        "name": "get_trade_patterns",
        "description": "Get trade timing and frequency data",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Team name"
                },
                "map_name": {
                    "type": "string",
                    "description": "Map name"
                }
            },
            "required": ["team_name"]
        }
    },
    {
        "name": "get_economy_patterns",
        "description": "Get economic decision patterns (buys, saves, forces)",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Team name"
                }
            },
            "required": ["team_name"]
        }
    },
    {
        "name": "run_what_if",
        "description": "Run a what-if simulation with modified positions",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_description": {
                    "type": "string",
                    "description": "Natural language description of the scenario"
                },
                "map_name": {
                    "type": "string"
                },
                "attack_team": {
                    "type": "string"
                },
                "defense_team": {
                    "type": "string"
                }
            },
            "required": ["scenario_description", "map_name"]
        }
    }
]
