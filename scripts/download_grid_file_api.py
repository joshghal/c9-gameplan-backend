#!/usr/bin/env python3
"""
GRID File Download API Client

Downloads VALORANT match event data from GRID's File Download API.
This API is available to hackathon participants with a valid API key.

Endpoints:
    List files:     GET https://api.grid.gg/file-download/list/{series_id}
    Download events: GET https://api.grid.gg/file-download/events/grid/series/{series_id}
    Download state:  GET https://api.grid.gg/file-download/end-state/grid/series/{series_id}

Usage:
    # Download specific series
    python scripts/download_grid_file_api.py --series 2629391

    # Download all C9 matches
    python scripts/download_grid_file_api.py --team cloud9

    # List available series (no download)
    python scripts/download_grid_file_api.py --team cloud9 --list

Rate Limits:
    - File download: 20 requests/minute
    - File listing: No limit
"""

import os
import sys
import asyncio
import argparse
import zipfile
from pathlib import Path
from datetime import datetime

import httpx

# Configuration
GRID_API_KEY = os.getenv('GRID_API_KEY', 'XsetiGFZvo03aZRaMRQbDf5eljk0jU6iVBHnNhTl')
OUTPUT_DIR = Path(__file__).parent.parent.parent / "grid_data"

# API endpoints
GRID_GRAPHQL = "https://api-op.grid.gg/central-data/graphql"
GRID_FILE_API = "https://api.grid.gg/file-download"

# VALORANT title ID
VALORANT_TITLE_ID = "6"

# Team IDs
TEAM_IDS = {
    'cloud9': '79',
    'sentinels': '77',
    'nrg': '92',
    'g2': '95',
    'mibr': '94',
    'leviatan': '96',
    'furia': '98',
    'kru': '99',
    'loud': '93',
    '100thieves': '91',
    'eg': '97',
}


async def get_team_series(client: httpx.AsyncClient, team_id: str, limit: int = 50) -> list:
    """Get all series for a team using GraphQL."""
    query = """
    query AllSeries($teamId: ID!, $titleId: ID!, $first: Int!) {
        allSeries(
            filter: { teamId: $teamId, titleId: $titleId }
            first: $first
            orderBy: StartTimeScheduled
        ) {
            edges {
                node {
                    id
                    startTimeScheduled
                    format { name }
                    tournament { name }
                    teams { baseInfo { id name } }
                }
            }
        }
    }
    """

    resp = await client.post(
        GRID_GRAPHQL,
        headers={"x-api-key": GRID_API_KEY, "Content-Type": "application/json"},
        json={
            "query": query,
            "variables": {"teamId": team_id, "titleId": VALORANT_TITLE_ID, "first": limit}
        },
        timeout=30
    )

    data = resp.json()
    if 'errors' in data:
        print(f"GraphQL error: {data['errors']}")
        return []

    edges = data.get('data', {}).get('allSeries', {}).get('edges', [])
    return [e['node'] for e in edges]


async def list_files(client: httpx.AsyncClient, series_id: str) -> dict:
    """List available files for a series."""
    resp = await client.get(
        f"{GRID_FILE_API}/list/{series_id}",
        headers={"x-api-key": GRID_API_KEY},
        timeout=30
    )

    if resp.status_code == 200:
        return resp.json()
    return {"files": [], "error": resp.status_code}


async def download_events(client: httpx.AsyncClient, series_id: str, output_dir: Path) -> bool:
    """Download events JSONL file for a series."""
    output_zip = output_dir / f"events_{series_id}_grid.jsonl.zip"
    output_jsonl = output_dir / f"events_{series_id}_grid.jsonl"

    # Skip if already downloaded
    if output_jsonl.exists() and output_jsonl.stat().st_size > 10000:
        return True

    try:
        resp = await client.get(
            f"{GRID_FILE_API}/events/grid/series/{series_id}",
            headers={"x-api-key": GRID_API_KEY},
            follow_redirects=True,
            timeout=120
        )

        if resp.status_code == 200 and len(resp.content) > 1000:
            # Save zip
            with open(output_zip, 'wb') as f:
                f.write(resp.content)

            # Extract
            try:
                with zipfile.ZipFile(output_zip, 'r') as z:
                    z.extractall(output_dir)
                return True
            except zipfile.BadZipFile:
                # Might be raw JSONL
                output_jsonl.write_bytes(resp.content)
                return True

        return False

    except Exception as e:
        print(f"  Error downloading {series_id}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description='Download GRID match data via File API')
    parser.add_argument('--series', type=str, help='Specific series ID to download')
    parser.add_argument('--team', type=str, help='Team name (cloud9, sentinels, etc.)')
    parser.add_argument('--limit', type=int, default=50, help='Max matches to fetch')
    parser.add_argument('--list', action='store_true', help='List only, no download')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        if args.series:
            # Download specific series
            print(f"Downloading series {args.series}...")
            success = await download_events(client, args.series, OUTPUT_DIR)
            if success:
                print(f"  ✓ Downloaded to {OUTPUT_DIR}")
            else:
                print(f"  ✗ Failed")

        elif args.team:
            team_lower = args.team.lower()
            team_id = TEAM_IDS.get(team_lower)

            if not team_id:
                print(f"Unknown team: {args.team}")
                print(f"Available: {list(TEAM_IDS.keys())}")
                return

            print(f"Finding {args.team.upper()} matches...")
            series_list = await get_team_series(client, team_id, args.limit)

            if not series_list:
                print("No matches found")
                return

            print(f"Found {len(series_list)} matches\n")

            if args.list:
                print(f"{'ID':<12} {'Date':<12} {'Opponent':<20} {'Tournament'}")
                print("-" * 80)
                for s in series_list:
                    teams = [t['baseInfo']['name'] for t in s.get('teams', [])]
                    opponent = [t for t in teams if team_lower not in t.lower()]
                    date = (s.get('startTimeScheduled') or '')[:10]
                    tournament = (s.get('tournament') or {}).get('name', '?')[:30]
                    print(f"{s['id']:<12} {date:<12} {opponent[0] if opponent else '?':<20} {tournament}")
            else:
                print("Downloading matches...\n")
                success_count = 0

                for i, s in enumerate(series_list):
                    series_id = s['id']
                    teams = [t['baseInfo']['name'] for t in s.get('teams', [])]

                    print(f"[{i+1}/{len(series_list)}] {series_id}: {' vs '.join(teams)}")

                    if await download_events(client, series_id, OUTPUT_DIR):
                        success_count += 1
                        jsonl_file = OUTPUT_DIR / f"events_{series_id}_grid.jsonl"
                        if jsonl_file.exists():
                            size_mb = jsonl_file.stat().st_size / 1024 / 1024
                            print(f"  ✓ {size_mb:.1f} MB")
                    else:
                        print(f"  ✗ Failed")

                    # Rate limit: 20 req/min = 3 sec between requests
                    await asyncio.sleep(3)

                print(f"\nDownloaded {success_count}/{len(series_list)} matches")

        else:
            print("Usage:")
            print("  --series ID     Download specific series")
            print("  --team NAME     Download all matches for team")
            print("  --list          List matches without downloading")
            print()
            print(f"Available teams: {', '.join(TEAM_IDS.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
