#!/usr/bin/env python3
"""
VLR.gg Stats Scraper

Scrapes player statistics from VLR.gg to build comprehensive player profiles.
This gives us aggregate stats across many matches, which is statistically
more reliable than single-match event data.

Usage:
    python scripts/scrape_vlr_stats.py --team cloud9
    python scripts/scrape_vlr_stats.py --player xeppaa
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import time

import httpx
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent.parent.parent / "grid_data"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Known player VLR IDs
PLAYER_IDS = {
    # Cloud9
    'zellsis': 9,
    'xeppaa': 601,
    'jakee': 16176,
    'leaf': 7,
    'vanity': 349,
    # Sentinels
    'tenz': 9,
    'zekken': 7859,
    'sacy': 2308,
    'johnqt': 5765,
    'pancada': 2310,
    # G2
    'icy': 16203,
    'valyn': 4004,
    'trent': 2217,
    'jonahp': 1961,
    # NRG
    's0m': 4120,
    'fns': 54,
    'crashies': 602,
    'ardiis': 147,
    'victor': 1410,
}

TEAM_ROSTERS = {
    'cloud9': ['zellsis', 'xeppaa', 'jakee', 'leaf', 'vanity'],
    'sentinels': ['tenz', 'zekken', 'sacy', 'johnqt', 'pancada'],
    'g2': ['icy', 'valyn', 'leaf', 'trent', 'jonahp'],
    'nrg': ['s0m', 'fns', 'crashies', 'ardiis', 'victor'],
}


@dataclass
class VLRPlayerStats:
    """Player statistics from VLR.gg"""
    player_name: str
    team: str

    # Core stats
    rating: float = 0.0
    acs: float = 0.0  # Average Combat Score
    kd_ratio: float = 0.0
    kast: float = 0.0  # Kill/Assist/Survive/Trade %
    adr: float = 0.0  # Average Damage per Round
    kpr: float = 0.0  # Kills per Round
    apr: float = 0.0  # Assists per Round
    fkpr: float = 0.0  # First Kills per Round
    fdpr: float = 0.0  # First Deaths per Round
    headshot_pct: float = 0.0
    clutch_pct: float = 0.0

    # Sample size
    rounds_played: int = 0
    matches_played: int = 0

    # Metadata
    source: str = "vlr.gg"
    scraped_at: str = ""


def scrape_player_page(player_name: str, player_id: int) -> Optional[VLRPlayerStats]:
    """Scrape stats from a player's VLR.gg page."""

    url = f"https://www.vlr.gg/player/{player_id}/{player_name.lower()}"

    try:
        response = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)

        if response.status_code != 200:
            print(f"  Error: HTTP {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the player name to verify we got the right page
        title = soup.find('h1', class_='wf-title')
        if not title:
            print(f"  Error: Could not find player title")
            return None

        actual_name = title.get_text(strip=True)

        # Find team
        team_el = soup.find('div', class_='wf-module-item')
        team = team_el.get_text(strip=True) if team_el else "Unknown"

        stats = VLRPlayerStats(
            player_name=actual_name,
            team=team
        )

        # Find stats table
        stats_table = soup.find('table', class_='wf-table-inset')

        if stats_table:
            rows = stats_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)

                    try:
                        if 'rating' in label:
                            stats.rating = float(value)
                        elif 'acs' in label:
                            stats.acs = float(value)
                        elif 'k:d' in label or 'k/d' in label:
                            stats.kd_ratio = float(value)
                        elif 'kast' in label:
                            stats.kast = float(value.replace('%', ''))
                        elif 'adr' in label:
                            stats.adr = float(value)
                        elif 'kpr' in label:
                            stats.kpr = float(value)
                        elif 'apr' in label:
                            stats.apr = float(value)
                        elif 'fkpr' in label:
                            stats.fkpr = float(value)
                        elif 'fdpr' in label:
                            stats.fdpr = float(value)
                        elif 'hs' in label:
                            stats.headshot_pct = float(value.replace('%', ''))
                        elif 'clutch' in label:
                            stats.clutch_pct = float(value.replace('%', ''))
                    except (ValueError, AttributeError):
                        pass

        # Try to find stats from the career stats section
        career_stats = soup.find_all('div', class_='value')
        stat_labels = soup.find_all('div', class_='label')

        for label_el, value_el in zip(stat_labels, career_stats):
            label = label_el.get_text(strip=True).lower()
            value = value_el.get_text(strip=True)

            try:
                if 'rating' in label and stats.rating == 0:
                    stats.rating = float(value)
                elif 'hs%' in label or 'headshot' in label:
                    stats.headshot_pct = float(value.replace('%', ''))
                elif 'k/d' in label or 'kd' in label:
                    stats.kd_ratio = float(value)
            except (ValueError, AttributeError):
                pass

        return stats

    except Exception as e:
        print(f"  Error scraping {player_name}: {e}")
        return None


def scrape_stats_leaderboard(region: str = "na", min_rounds: int = 200) -> List[VLRPlayerStats]:
    """Scrape the VLR.gg stats leaderboard."""

    url = f"https://www.vlr.gg/stats/?event_group_id=all&event_id=all&region={region}&min_rounds={min_rounds}&timespan=90d"

    print(f"Fetching stats leaderboard: {url}")

    try:
        response = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)

        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find stats table
        table = soup.find('table', class_='wf-table')
        if not table:
            print("Could not find stats table")
            return []

        players = []
        rows = table.find('tbody').find_all('tr') if table.find('tbody') else []

        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 10:
                continue

            try:
                # Extract player info
                player_cell = cells[0]
                player_link = player_cell.find('a')
                if not player_link:
                    continue

                player_name = player_link.get_text(strip=True)

                # Team
                team_el = player_cell.find('div', class_='stats-player-team')
                team = team_el.get_text(strip=True) if team_el else "Unknown"

                # Stats columns (order: Agents, Rnd, Rating, ACS, K:D, KAST, ADR, KPR, APR, FKPR, FDPR, HS%, CL%)
                stats = VLRPlayerStats(
                    player_name=player_name,
                    team=team
                )

                # Parse stats from cells
                def safe_float(text):
                    try:
                        return float(text.replace('%', '').replace(',', ''))
                    except:
                        return 0.0

                if len(cells) > 2:
                    stats.rounds_played = int(safe_float(cells[2].get_text(strip=True)))
                if len(cells) > 3:
                    stats.rating = safe_float(cells[3].get_text(strip=True))
                if len(cells) > 4:
                    stats.acs = safe_float(cells[4].get_text(strip=True))
                if len(cells) > 5:
                    stats.kd_ratio = safe_float(cells[5].get_text(strip=True))
                if len(cells) > 6:
                    stats.kast = safe_float(cells[6].get_text(strip=True))
                if len(cells) > 7:
                    stats.adr = safe_float(cells[7].get_text(strip=True))
                if len(cells) > 8:
                    stats.kpr = safe_float(cells[8].get_text(strip=True))
                if len(cells) > 9:
                    stats.apr = safe_float(cells[9].get_text(strip=True))
                if len(cells) > 10:
                    stats.fkpr = safe_float(cells[10].get_text(strip=True))
                if len(cells) > 11:
                    stats.fdpr = safe_float(cells[11].get_text(strip=True))
                if len(cells) > 12:
                    stats.headshot_pct = safe_float(cells[12].get_text(strip=True))
                if len(cells) > 13:
                    stats.clutch_pct = safe_float(cells[13].get_text(strip=True))

                players.append(stats)

            except Exception as e:
                continue

        return players

    except Exception as e:
        print(f"Error scraping leaderboard: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Scrape VLR.gg player stats')
    parser.add_argument('--team', type=str, help='Team name to scrape')
    parser.add_argument('--player', type=str, help='Specific player to scrape')
    parser.add_argument('--leaderboard', action='store_true', help='Scrape stats leaderboard')
    parser.add_argument('--region', type=str, default='na', help='Region for leaderboard')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []

    if args.leaderboard:
        print("Scraping VLR.gg stats leaderboard...")
        all_stats = scrape_stats_leaderboard(region=args.region)

    elif args.team:
        team = args.team.lower()
        if team not in TEAM_ROSTERS:
            print(f"Unknown team: {team}")
            print(f"Available: {list(TEAM_ROSTERS.keys())}")
            return

        print(f"Scraping {team.upper()} roster stats...")
        for player in TEAM_ROSTERS[team]:
            if player in PLAYER_IDS:
                print(f"  Fetching {player}...")
                stats = scrape_player_page(player, PLAYER_IDS[player])
                if stats:
                    all_stats.append(stats)
                time.sleep(1)  # Be nice to VLR servers

    elif args.player:
        player = args.player.lower()
        if player in PLAYER_IDS:
            print(f"Scraping {player}...")
            stats = scrape_player_page(player, PLAYER_IDS[player])
            if stats:
                all_stats.append(stats)
        else:
            print(f"Unknown player: {player}")

    else:
        # Default: scrape leaderboard
        print("Scraping VLR.gg NA leaderboard (last 90 days)...")
        all_stats = scrape_stats_leaderboard(region='na', min_rounds=200)

    if all_stats:
        # Print results
        print(f"\n{'='*70}")
        print(f"{'Player':<15} {'Team':<12} {'Rating':>7} {'ACS':>6} {'K/D':>5} {'HS%':>5} {'FKPR':>5}")
        print(f"{'='*70}")

        for s in all_stats[:30]:
            print(f"{s.player_name:<15} {s.team:<12} {s.rating:>7.2f} {s.acs:>6.1f} {s.kd_ratio:>5.2f} {s.headshot_pct:>5.1f} {s.fkpr:>5.2f}")

        # Save to file
        output_path = OUTPUT_DIR / "vlr_player_stats.json"
        with open(output_path, 'w') as f:
            json.dump([asdict(s) for s in all_stats], f, indent=2)
        print(f"\nSaved {len(all_stats)} player stats to {output_path}")


if __name__ == "__main__":
    main()
