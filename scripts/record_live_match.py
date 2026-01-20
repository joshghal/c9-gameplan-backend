#!/usr/bin/env python3
"""
GRID Live Match Recorder

Records live match events from GRID's real-time feed.
Run this during a live match to capture detailed event data.

Usage:
    # Record a specific series
    python scripts/record_live_match.py --series-id 2629398

    # Find and list live matches
    python scripts/record_live_match.py --find-live

    # Record C9's next live match (auto-detect)
    python scripts/record_live_match.py --team cloud9

Note: This requires the match to be currently live or recently started.
The GRID live data feed is only available during active matches.

UPCOMING C9 MATCHES (VCT 26):
    - Jan 25, 2026: Cloud9 vs LEVIATÁN @ VCT Americas Kickoff

To record this match:
    1. Wait until the match starts
    2. Find the series ID from GRID or VLR.gg
    3. Run: python scripts/record_live_match.py --series-id <SERIES_ID>
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv()

GRID_API_KEY = os.getenv('GRID_API_KEY', '')
OUTPUT_DIR = Path(__file__).parent.parent.parent / "grid_data"

# GRID API endpoints
GRID_CENTRAL_API = "https://api-op.grid.gg/central-data/graphql"
GRID_LIVE_FEED_BASE = "https://api-op.grid.gg/live-data-feed"

# Team IDs (from GRID)
TEAM_IDS = {
    'cloud9': '79',
    'sentinels': '77',
    'nrg': '92',
    'g2': '95',
    'mibr': '94',
    'leviatan': '96',
}


async def record_live_feed(series_id: str, output_path: Path):
    """Record events from a live match feed."""

    # GRID WebSocket feed URL (example - actual URL may differ)
    ws_url = f"wss://live-feed.grid.gg/series/{series_id}"

    print(f"Connecting to live feed for series {series_id}...")
    print(f"Output: {output_path}")
    print("Press Ctrl+C to stop recording.\n")

    event_count = 0

    try:
        async with websockets.connect(
            ws_url,
            extra_headers={"x-api-key": GRID_API_KEY}
        ) as ws:
            print("Connected! Recording events...\n")

            with open(output_path, 'w') as f:
                async for message in ws:
                    try:
                        data = json.loads(message)
                        f.write(json.dumps(data) + '\n')
                        f.flush()

                        event_count += 1
                        event_type = data.get('type', 'unknown')

                        # Log important events
                        if 'kill' in event_type.lower():
                            actor = data.get('actor', {}).get('state', {}).get('name', '?')
                            target = data.get('target', {}).get('state', {}).get('name', '?')
                            print(f"[{event_count}] KILL: {actor} -> {target}")
                        elif 'round' in event_type.lower():
                            print(f"[{event_count}] {event_type}")
                        elif event_count % 100 == 0:
                            print(f"[{event_count}] Events recorded...")

                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON received")

    except websockets.exceptions.ConnectionClosed:
        print("\nConnection closed by server.")
    except Exception as e:
        print(f"\nConnection error: {e}")

    print(f"\nRecording complete. Total events: {event_count}")
    print(f"Saved to: {output_path}")


async def poll_live_feed(series_id: str, output_path: Path):
    """Poll for events using HTTP (fallback if WebSocket unavailable)."""

    print(f"Polling live feed for series {series_id}...")
    print(f"Output: {output_path}")
    print("Press Ctrl+C to stop.\n")

    event_count = 0
    last_sequence = 0

    async with httpx.AsyncClient() as client:
        with open(output_path, 'w') as f:
            while True:
                try:
                    # Try to get events since last sequence
                    url = f"https://api-op.grid.gg/live-data-feed/series/{series_id}/events"
                    params = {"afterSequence": last_sequence} if last_sequence > 0 else {}

                    response = await client.get(
                        url,
                        headers={"x-api-key": GRID_API_KEY},
                        params=params,
                        timeout=30
                    )

                    if response.status_code == 200:
                        events = response.json()
                        if isinstance(events, list) and events:
                            for event in events:
                                f.write(json.dumps(event) + '\n')
                                event_count += 1
                                seq = event.get('sequenceNumber', 0)
                                if seq > last_sequence:
                                    last_sequence = seq

                            print(f"[{event_count}] Events recorded (seq: {last_sequence})")
                            f.flush()
                        else:
                            print(".", end="", flush=True)

                    elif response.status_code == 404:
                        print("\nSeries not found or not live.")
                        break
                    else:
                        print(f"\nHTTP {response.status_code}")

                    await asyncio.sleep(1)  # Poll every second

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    await asyncio.sleep(5)

    print(f"\n\nRecording complete. Total events: {event_count}")


async def find_live_series(team_filter: Optional[str] = None) -> List[Dict]:
    """Find currently live Valorant series."""

    query = """
    query GetLiveSeries($titleId: Int!) {
        liveSeries(titleId: $titleId) {
            id
            startedAt
            teams {
                baseInfo {
                    id
                    name
                }
            }
            tournament {
                name
            }
        }
    }
    """

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                GRID_CENTRAL_API,
                headers={"x-api-key": GRID_API_KEY, "Content-Type": "application/json"},
                json={"query": query, "variables": {"titleId": 6}},  # 6 = Valorant
                timeout=30
            )

            if response.status_code != 200:
                print(f"Failed to fetch live series: HTTP {response.status_code}")
                return []

            data = response.json()

            if 'errors' in data:
                print(f"GraphQL error: {data['errors']}")
                return []

            live_series = data.get('data', {}).get('liveSeries', []) or []

            result = []
            for series in live_series:
                teams = [t['baseInfo']['name'] for t in series.get('teams', [])]
                team_ids = [t['baseInfo']['id'] for t in series.get('teams', [])]

                # Filter by team if specified
                if team_filter:
                    filter_id = TEAM_IDS.get(team_filter.lower(), '')
                    if filter_id not in team_ids and team_filter.lower() not in [t.lower() for t in teams]:
                        continue

                result.append({
                    'id': series['id'],
                    'teams': teams,
                    'tournament': series.get('tournament', {}).get('name', 'Unknown'),
                    'started_at': series.get('startedAt', ''),
                })

            return result

        except Exception as e:
            print(f"Error finding live series: {e}")
            return []


async def main():
    parser = argparse.ArgumentParser(description='Record live GRID match data')
    parser.add_argument('--series-id', type=str, help='Series ID to record')
    parser.add_argument('--team', type=str, help='Team name to filter (e.g., cloud9)')
    parser.add_argument('--find-live', action='store_true', help='Find currently live matches')
    parser.add_argument('--method', choices=['ws', 'poll'], default='poll', help='Recording method')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.find_live or args.team:
        print("Searching for live Valorant matches...")
        live_series = await find_live_series(args.team)

        if not live_series:
            print("\nNo live matches found.")
            if args.team:
                print(f"No live {args.team.upper()} matches at this time.")
            print("\nUpcoming C9 matches:")
            print("  - Jan 25, 2026: Cloud9 vs LEVIATÁN @ VCT Americas Kickoff")
            print("\nTo record when live, find the series ID and run:")
            print("  python scripts/record_live_match.py --series-id <SERIES_ID>")
            return

        print(f"\nFound {len(live_series)} live match(es):\n")
        for series in live_series:
            print(f"  Series ID: {series['id']}")
            print(f"  Match: {' vs '.join(series['teams'])}")
            print(f"  Tournament: {series['tournament']}")
            print(f"  Started: {series['started_at']}")
            print()

        if not args.series_id and len(live_series) == 1:
            # Auto-select if only one match
            args.series_id = live_series[0]['id']
            print(f"Auto-selecting series {args.series_id}...\n")

    if not args.series_id:
        print("No series ID specified. Use --series-id or --find-live")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f"events_{args.series_id}_live_{timestamp}.jsonl"

    if args.method == 'ws':
        await record_live_feed(args.series_id, output_path)
    else:
        await poll_live_feed(args.series_id, output_path)


if __name__ == "__main__":
    asyncio.run(main())
