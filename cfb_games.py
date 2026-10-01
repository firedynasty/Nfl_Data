"""
cfb_games.py -- college football (FBS) schedules and scores from ESPN's
public scoreboard API, shaped EXACTLY like nflverse's games.csv so the
whole NFL pipeline (srs.py, backtest_srs.py, predict_cfb_week.py) reuses
unchanged: game_id, season, week, gameday, gametime, away_team, home_team,
away_score, home_score, location, spread_line.

Same idea as nfl_box_score_analysis.load_games(), different source:
nflverse doesn't cover college, but ESPN's scoreboard endpoint does
(no key, server-rendered JSON), back through 2021 and beyond.

  https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard
      ?dates=<season>&seasontype=<2 reg | 3 post>&week=<n>&groups=80

groups=80 = FBS. Games vs FCS opponents are dropped (a team appearing in
fewer than MIN_FBS_GAMES of a season's fetched games is treated as FCS) --
that's the standard SRS practice; rating FCS teams off 1-2 crossover games
pollutes everyone else's schedule correction.

Postseason (bowls/playoff) weeks are renumbered to follow the last regular
week, so "as of week W" ordering works across the whole season.

spread_line is home-positive (positive = home favored), the verified
nflverse convention, parsed from ESPN's odds details string ("GT -3.5").
ESPN only carries odds for recent/upcoming games, so historical rows have
no line -- fine, SRS needs only scores and the market line only matters
for the current week's sheet.

Results are cached to cfb_games_cache.csv: past seasons are fetched once,
the current season is re-fetched on every load (scores update).

USAGE
-----
  python cfb_games.py                 # fetch what's missing, print a summary
  python cfb_games.py --refresh       # re-fetch the current season too
"""

import json
import re
import subprocess
import sys
import time

import pandas as pd

ESPN = ("https://site.api.espn.com/apis/site/v2/sports/football/"
        "college-football/scoreboard")
GROUPS_FBS = 80
CACHE_PATH = "cfb_games_cache.csv"
FIRST_SEASON = 2021          # matches the NFL pipeline's backtest window
MIN_FBS_GAMES = 6            # below this in a season's feed -> treat as FCS
REG_WEEKS = range(1, 17)     # week 1 folds in "week 0"; 16 = conf championships
POST_WEEKS = range(1, 4)     # bowls/playoff; loop stops early on empty weeks


def fetch_week(season, seasontype, week):
    """One ESPN scoreboard page -> list of game dicts, or [] when the week
    doesn't exist (ESPN returns an empty events list past the end)."""
    url = (f"{ESPN}?dates={season}&seasontype={seasontype}&week={week}"
           f"&groups={GROUPS_FBS}&limit=300")
    out = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", url],
                         capture_output=True, timeout=30)
    try:
        return json.loads(out.stdout).get("events", [])
    except json.JSONDecodeError:
        print(f"warning: {season} st={seasontype} wk={week} not JSON, skipping",
              file=sys.stderr)
        return []


def parse_spread(details, home_abbr):
    """ESPN odds details ('GT -3.5', 'EVEN') -> home-positive spread_line."""
    if not details:
        return None
    if details.strip().upper().startswith("EVEN"):
        return 0.0
    m = re.match(r"(.+?)\s+(-?\d+\.?\d*)$", details.strip())
    if not m:
        return None
    fav, pts = m.group(1), float(m.group(2))
    return -pts if fav == home_abbr else pts


def parse_event(e, season, week):
    comp = e["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    home, away = sides.get("home"), sides.get("away")
    if not home or not away:
        return None
    final = comp["status"]["type"]["name"] == "STATUS_FINAL"

    def score(side):
        try:
            return float(side["score"]) if final else None
        except (KeyError, TypeError, ValueError):
            return None

    odds = (comp.get("odds") or [{}])[0]
    dt = pd.to_datetime(e["date"], utc=True)
    return {
        "game_id": f"cfb_{e['id']}",
        "season": season,
        "week": week,
        "gameday": dt.strftime("%Y-%m-%d"),
        "gametime": dt.strftime("%H:%M"),
        "away_team": away["team"]["abbreviation"],
        "home_team": home["team"]["abbreviation"],
        "away_score": score(away),
        "home_score": score(home),
        "location": "Neutral" if comp.get("neutralSite") else "Home",
        "spread_line": parse_spread(odds.get("details"), home["team"]["abbreviation"]),
    }


def fetch_season(season, sleep=0.2):
    """All regular + postseason weeks of one season. Postseason weeks are
    renumbered to continue after the last regular week. Games involving a
    sub-FBS opponent are dropped (see module docstring)."""
    rows = []
    last_reg = 0
    for wk in REG_WEEKS:
        events = fetch_week(season, 2, wk)
        if not events:
            break
        last_reg = wk
        rows += [r for e in events if (r := parse_event(e, season, wk))]
        time.sleep(sleep)
    for wk in POST_WEEKS:
        events = fetch_week(season, 3, wk)
        if not events:
            break
        rows += [r for e in events if (r := parse_event(e, season, last_reg + wk))]
        time.sleep(sleep)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    played = df.dropna(subset=["home_score", "away_score"])
    counts = pd.concat([played["home_team"], played["away_team"]]).value_counts()
    # Mid-season nobody has played MIN_FBS_GAMES yet, so scale the FBS bar
    # to the season so far: FCS crossover opponents play 1-3 FBS games a
    # year, FBS teams roughly two-thirds or more of whatever the leaders
    # have. Below 3 games played by anyone it's week 1-2 and the bar is 2.
    mx = int(counts.max()) if len(counts) else 0
    min_games = 2 if mx < 3 else min(MIN_FBS_GAMES, max(3, (mx * 2) // 3))
    fbs = set(counts[counts >= min_games].index)
    keep = df["home_team"].isin(fbs) & df["away_team"].isin(fbs)
    dropped = df[~keep]
    if len(dropped):
        fcs = sorted((set(dropped["home_team"]) | set(dropped["away_team"])) - fbs)
        print(f"  {season}: dropped {len(dropped)} games vs {len(fcs)} sub-FBS teams "
              f"(bar: {min_games}+ games; {', '.join(fcs[:8])}{'...' if len(fcs) > 8 else ''})",
              file=sys.stderr)
    return df[keep].reset_index(drop=True)


def current_cfb_season():
    """College season for 'today': the season year is the fall year, so
    Jan-Aug still belongs to the previous season (bowls land in January)."""
    now = pd.Timestamp.now()
    return now.year if now.month >= 8 else now.year - 1


def load_cfb_games(seasons=None, refresh=False):
    """The load_games() equivalent. Past seasons are fetched once and
    cached; the current season is re-fetched on every load (scores update).
    --refresh re-fetches everything."""
    cur = current_cfb_season()
    seasons = seasons or list(range(FIRST_SEASON, cur + 1))

    cached = pd.DataFrame()
    if not refresh:
        try:
            cached = pd.read_csv(CACHE_PATH)
        except FileNotFoundError:
            pass

    frames, have = [], set(cached["season"].unique()) if len(cached) else set()
    if len(cached):
        frames.append(cached)
    for s in seasons:
        if s in have and s < cur:
            continue                 # past season: cache is final
        if s in have:
            # current season (or --refresh): scores change, drop stale rows
            frames[0] = frames[0][frames[0]["season"] != s]
        print(f"fetching {s} FBS season from ESPN...", file=sys.stderr)
        df = fetch_season(s)
        if not df.empty:
            frames.append(df)

    games = (pd.concat(frames, ignore_index=True)
             .drop_duplicates("game_id", keep="last")
             .sort_values(["season", "week", "gameday"])
             .reset_index(drop=True))
    games.to_csv(CACHE_PATH, index=False)
    return games


def main():
    refresh = "--refresh" in sys.argv
    games = load_cfb_games(refresh=refresh)
    played = games.dropna(subset=["home_score", "away_score"])
    print(f"\ncfb games cache: {len(games)} rows "
          f"({len(played)} played), seasons {games['season'].min()}-"
          f"{games['season'].max()} -> {CACHE_PATH}")
    print(played.groupby("season").agg(games=("game_id", "count"),
                                       teams=("home_team", "nunique")).to_string())


if __name__ == "__main__":
    main()
