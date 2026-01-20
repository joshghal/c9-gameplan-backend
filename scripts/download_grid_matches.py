#!/usr/bin/env python3
"""
GRID Match Data Downloader

Downloads match event data from GRID for Cloud9 Valorant matches.

IMPORTANT: Historical event data (JSONL files) can only be downloaded from
the GRID Portal web interface (https://grid.gg/). The API only provides metadata.

This script helps by:
1. Finding all C9 match series IDs
2. Checking which ones are already downloaded
3. Providing the series IDs needed for manual portal download

Usage:
    python scripts/download_grid_matches.py --list          # List all C9 matches
    python scripts/download_grid_matches.py --check         # Check download status
    python scripts/download_grid_matches.py --series 2629398  # Try API download

Manual Download Steps (GRID Portal):
    1. Log into https://grid.gg/ with VCT partner credentials
    2. Navigate to VALORANT Data Portal
    3. Search for series ID (e.g., 2629398)
    4. Download the match events as JSONL
    5. Save to: grid_data/events_<series_id>_grid.jsonl
"""

import os
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

GRID_API_KEY = os.getenv('GRID_API_KEY', '')
GRID_API_URL = "https://api-op.grid.gg/central-data/graphql"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "grid_data"


async def find_c9_series() -> list:
    """Find all Cloud9 Valorant series IDs from GRID."""

    query = """
    query GetAllSeries {
        allSeries(filter: { titleId: 6 }, first: 50) {
            edges {
                node {
                    id
                    teams {
                        baseInfo { name }
                    }
                }
            }
        }
    }
    """

    async with httpx.AsyncClient() as client:
        response = await client.post(
            GRID_API_URL,
            headers={"x-api-key": GRID_API_KEY, "Content-Type": "application/json"},
            json={"query": query},
            timeout=30
        )

        data = response.json()
        if 'errors' in data:
            print(f"API Error: {data['errors'][0].get('message', 'Unknown')}")
            return []

        c9_matches = []
        series_list = data.get('data', {}).get('allSeries', {}).get('edges', [])

        for s in series_list:
            node = s.get('node', {})
            series_id = node.get('id')
            teams = [t.get('baseInfo', {}).get('name', '?') for t in node.get('teams', [])]

            # Check if C9 is playing
            is_c9 = any('cloud9' in str(t).lower() or 'c9' in str(t).lower() for t in teams)
            if is_c9:
                c9_matches.append({
                    'id': series_id,
                    'teams': ' vs '.join(teams),
                    'downloaded': check_downloaded(series_id)
                })

        return c9_matches


def check_downloaded(series_id: str) -> bool:
    """Check if a series is already downloaded."""
    path = OUTPUT_DIR / f"events_{series_id}_grid.jsonl"
    return path.exists() and path.stat().st_size > 10000


def list_downloaded():
    """List all downloaded GRID files."""
    files = list(OUTPUT_DIR.glob("events_*_grid.jsonl"))

    print("\nDownloaded GRID Files:")
    print("=" * 60)

    for f in sorted(files):
        size = f.stat().st_size
        series_id = f.stem.split('_')[1]

        # Count events
        event_count = 0
        try:
            with open(f, 'r') as fp:
                for line in fp:
                    event_count += 1
        except:
            pass

        print(f"  {f.name}")
        print(f"    Size: {size:,} bytes | Events: {event_count:,}")
        print()

    return files


async def try_api_download(series_id: str) -> bool:
    """Attempt to download from GRID API (usually fails for historical data)."""

    endpoints = [
        f"https://api-op.grid.gg/live-data-feed/series-state/valorant/{series_id}",
        f"https://api-op.grid.gg/live-data-feed/series/{series_id}/events",
    ]

    headers = {
        "x-api-key": GRID_API_KEY,
        "Accept": "application/json, application/x-ndjson",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for url in endpoints:
            try:
                response = await client.get(url, headers=headers)

                if response.status_code == 200 and len(response.content) > 1000:
                    # Check for errors
                    try:
                        data = json.loads(response.content)
                        if 'errors' in data:
                            continue
                    except:
                        pass

                    output_path = OUTPUT_DIR / f"events_{series_id}_grid.jsonl"
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded to {output_path}")
                    return True

            except Exception:
                continue

    return False


async def main():
    parser = argparse.ArgumentParser(description='Download GRID match data')
    parser.add_argument('--list', action='store_true', help='List all C9 matches')
    parser.add_argument('--check', action='store_true', help='Check download status')
    parser.add_argument('--series', type=str, help='Try to download specific series')

    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        list_downloaded()
        return

    if args.series:
        print(f"Attempting API download for series {args.series}...")
        if await try_api_download(args.series):
            print("Success!")
        else:
            print("Failed - historical data requires GRID Portal access")
            print(f"\nManual download: https://grid.gg/ -> Search for series {args.series}")
        return

    # Default: list C9 matches
    print("Finding Cloud9 matches from GRID...")
    c9_matches = await find_c9_series()

    if not c9_matches:
        print("No C9 matches found in recent data")
        return

    print(f"\n{'='*70}")
    print(f"{'Series ID':<12} {'Status':<12} {'Match'}")
    print(f"{'='*70}")

    for m in c9_matches:
        status = "✓ Downloaded" if m['downloaded'] else "✗ Missing"
        print(f"{m['id']:<12} {status:<12} {m['teams']}")

    missing = [m for m in c9_matches if not m['downloaded']]

    if missing:
        print(f"\n{'='*70}")
        print("TO DOWNLOAD (via GRID Portal):")
        print(f"{'='*70}")
        for m in missing:
            print(f"  Series {m['id']}: {m['teams']}")
        print("\nSteps:")
        print("  1. Log into https://grid.gg/ with VCT partner credentials")
        print("  2. Navigate to VALORANT Data Portal")
        print("  3. Search for series ID and download events as JSONL")
        print(f"  4. Save to: {OUTPUT_DIR}/events_<series_id>_grid.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
