"""
Wind-under totals system -- weekly qualifier + historical validation.

THE SYSTEM (flagged in backtest_weather.py Phase 10 raw cuts, 2000-2025,
as the project's first dose-response + era-stable totals signal):

  Bet UNDER the closing total on OUTDOOR games with wind >= 11 mph.

  Backtest (7,017 games with closing totals; blind under):
    wind 0-6 mph   48.3% under (n=1,518)
    wind 6-11 mph  50.2% under (n=2,088)
    wind 11-16 mph 55.5% under (n=945)   <-- the rule
    wind 16+ mph   55.5% under (n=457)   <-- lifetime fine, but DECAYING
  Era split in the 11-16 band: 53.9 / 51.0 / 59.1 / 60.3% across
  2000-06 / 07-13 / 14-19 / 20-25 -- not decaying; the market lowers
  windy-game totals only ~half as much as scoring actually drops
  (fitted effect -0.23 pts/mph, market miss -1.4 to -1.6 pts recent).
  The 16+ band decayed to ~51-52% in the last two eras (thin n ~85/era)
  -- maybe noise, maybe books sharpening on high-profile wind games.
  So qualifiers are tagged by band (11-15 vs 16+) and the forward paper
  trail grades them separately; if 16+ keeps sliding, drop it.

  Mechanism: wind degrades passing efficiency AND field-goal range.
  Breakeven at -110 is 52.4%. Parameters were chosen after seeing the
  backtest buckets -- same standing as road_dog: a hypothesis being
  forward-tested, not a proven edge.

FORECAST vs ACTUAL WIND (read before betting): games.csv's `wind` is the
ACTUAL game-time wind, recorded post-game. For unplayed games this tool
pulls the Open-Meteo forecast (free, no key, global) at the stadium's
coordinates for the kickoff hour. Forecasts shift: run it close to
kickoff. The --results paper trail grades qualifiers reconstructed from
ACTUAL recorded wind (the historical record), so live picks made on a
forecast will occasionally differ from the graded set -- that's the
honest cost of not storing forecasts.

RETRACTABLE ROOFS: games.csv leaves `roof` empty for unplayed games at
retractable venues (ATL/DAL/HOU/IND/ARI). Those games FAIL the qualifier
with reason "roof TBD" -- a stadium that closes its roof in high wind is
not an outdoor game, and you won't know until ~90 min before kickoff.
Re-run once the roof decision is announced.

Data: nflverse games.csv (lines, totals, actual weather) + Open-Meteo
forecast API (upcoming games only). No keys, no scraping.

USAGE
-----
  python3 systems/wind_under/wind_under_system.py            # this week's qualifiers (forecast)
  python3 systems/wind_under/wind_under_system.py --week 1   # past week, graded (actual wind)
  python3 systems/wind_under/wind_under_system.py --results  # season paper trail, graded
  python3 systems/wind_under/wind_under_system.py --backtest # historical validation
  python3 systems/wind_under/wind_under_system.py --min-wind 12 --no-forecast
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
BREAKEVEN = 52.4  # cover % needed at standard -110 juice
ET = ZoneInfo("America/New_York")  # games.csv gametime is ET
OUTDOOR_ROOFS = ("outdoors", "open")

# Stadium display name (exactly as in games.csv) -> (lat, lon).
# Only venues hosting games from 2023 on are needed for FORECASTS --
# played games use the recorded actual wind, no coordinates required.
STADIUM_COORDS = {
    "Mercedes-Benz Stadium": (33.7554, -84.4008),
    "M&T Bank Stadium": (39.2780, -76.6227),
    "Gillette Stadium": (42.0909, -71.2643),
    "Highmark Stadium": (42.7738, -78.7870),
    "New Era Field": (42.7738, -78.7870),
    "Bank of America Stadium": (35.2258, -80.8530),
    "Soldier Field": (41.8623, -87.6167),
    "Paycor Stadium": (39.0954, -84.5160),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "Huntington Bank Field": (41.5061, -81.6995),
    "AT&T Stadium": (32.7473, -97.0945),
    "Empower Field at Mile High": (39.7439, -105.0201),
    "Ford Field": (42.3400, -83.0456),
    "Deutsche Bank Park": (50.0686, 8.6455),
    "Allianz Arena": (48.2188, 11.6248),
    "FC Bayern Munich Stadium": (48.2188, 11.6248),
    "Lambeau Field": (44.5013, -88.0622),
    "NRG Stadium": (29.6847, -95.4107),
    "Reliant Stadium": (29.6847, -95.4107),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "EverBank Stadium": (30.3239, -81.6373),
    "TIAA Bank Stadium": (30.3239, -81.6373),
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839),
    "Arrowhead Stadium": (39.0489, -94.4839),
    "SoFi Stadium": (33.9535, -118.3392),
    "Wembley Stadium": (51.5560, -0.2795),
    "Tottenham Hotspur Stadium": (51.6043, -0.0664),
    "Tottenham Stadium": (51.6043, -0.0664),
    "Melbourne Cricket Ground": (-37.8200, 144.9834),
    "Estadio Banorte": (19.3029, -99.1505),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "U.S. Bank Stadium": (44.9738, -93.2577),
    "Nissan Stadium": (36.1665, -86.7713),
    "Caesars Superdome": (29.9511, -90.0812),
    "Mercedes-Benz Superdome": (29.9511, -90.0812),
    "MetLife Stadium": (40.8128, -74.0742),
    "Stade de France": (48.9245, 2.3602),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "State Farm Stadium": (33.5276, -112.2626),
    "Acrisure Stadium": (40.4468, -80.0158),
    "Arena Corinthians": (-23.5453, -46.4743),
    "Lumen Field": (47.5952, -122.3316),
    "Levi's Stadium": (37.4030, -121.9700),
    "Raymond James Stadium": (27.9759, -82.5033),
    "Allegiant Stadium": (36.0909, -115.1833),
    "FedExField": (38.9076, -76.8645),
    "Northwest Stadium": (38.9076, -76.8645),
}

_fc_cache = {}  # stadium name -> (times, speeds) from Open-Meteo


def fetch_games():
    g = pd.read_csv(GAMES_URL, low_memory=False)
    return g[g["game_type"] == "REG"].copy()


def kickoff_utc(row):
    """gameday + gametime (ET) -> UTC datetime for the forecast lookup."""
    try:
        t = row["gametime"] if isinstance(row["gametime"], str) else "13:00"
        return datetime.strptime(f"{row['gameday']} {t}", "%Y-%m-%d %H:%M") \
            .replace(tzinfo=ET).astimezone(ZoneInfo("UTC"))
    except (ValueError, TypeError):
        return None


def forecast_wind(stadium, kickoff, offline=False):
    """Open-Meteo hourly 10m wind speed (mph) at the stadium for the hour
    nearest kickoff. None when unavailable (no coords / beyond the ~7-day
    forecast horizon / network down)."""
    if offline or kickoff is None or stadium not in STADIUM_COORDS:
        return None
    if stadium not in _fc_cache:
        lat, lon = STADIUM_COORDS[stadium]
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=UTC&forecast_days=10")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                h = json.load(r)["hourly"]
            _fc_cache[stadium] = (h["time"], h["wind_speed_10m"])
        except Exception as e:
            print(f"  note: forecast fetch failed for {stadium} "
                  f"({type(e).__name__})", file=sys.stderr)
            _fc_cache[stadium] = None
    cached = _fc_cache[stadium]
    if cached is None:
        return None
    times, speeds = cached
    stamps = [datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=ZoneInfo("UTC"))
              for t in times]
    best = min(range(len(stamps)), key=lambda i: abs((stamps[i] - kickoff).total_seconds()))
    if abs((stamps[best] - kickoff).total_seconds()) > 3 * 3600:
        return None  # kickoff outside the forecast horizon
    return speeds[best]


def profit_100(odds, won):
    """Flat 100 stake at American odds."""
    return (100 * 100 / -odds) if (won and odds < 0) else (odds if won else -100.0)


def band(wind):
    return "11-15" if wind < 16 else "16+"


def evaluate(g, season, week, args, use_forecast):
    """One row per game in the week. Played games grade on ACTUAL wind;
    unplayed games qualify on the FORECAST (when available)."""
    wk = g[(g["season"] == season) & (g["week"] == week)].copy()
    wk = wk.dropna(subset=["total_line"])
    rows = []
    for _, r in wk.iterrows():
        played = pd.notna(r["home_score"]) and pd.notna(r["away_score"])
        roof = r["roof"] if isinstance(r["roof"], str) else None
        wind, src = None, ""
        if played and pd.notna(r["wind"]):
            wind, src = float(r["wind"]), ""
        elif not played and use_forecast:
            w_ = forecast_wind(r["stadium"], kickoff_utc(r), args.no_forecast)
            if w_ is not None:
                wind, src = round(float(w_), 1), "fc"
        checks = {
            "outdoor": roof in OUTDOOR_ROOFS,
            f"wind>={args.min_wind:g}": wind is not None and wind >= args.min_wind,
        }
        row = {
            "gameday": r["gameday"], "weekday": r["weekday"],
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "roof": roof or "TBD(retract)", "wind": wind, "src": src,
            "total_line": r["total_line"], "under_odds": r.get("under_odds"),
            "checks": checks, "qualifies": all(checks.values()),
            "played": played, "score": None, "total_str": None, "bet": None,
        }
        if played:
            a_sc, h_sc = int(r["away_score"]), int(r["home_score"])
            actual = a_sc + h_sc
            row["score"] = f"{r['away_team']} {a_sc} - {r['home_team']} {h_sc}"
            word = ("over" if actual > r["total_line"]
                    else "under" if actual < r["total_line"] else "push")
            row["total_str"] = f"total {actual} ({word} {r['total_line']:g})"
            row["bet"] = ("PUSH" if actual == r["total_line"]
                          else "WON" if actual < r["total_line"] else "LOST")
        rows.append(row)
    return rows


def detect_current_week(g):
    season = int(g["season"].max())
    s = g[(g["season"] == season)].dropna(subset=["total_line"])
    unplayed = s[s["home_score"].isna()]
    if unplayed.empty:
        return season, int(s["week"].max())
    return season, int(unplayed["week"].min())


def fail_reason(row):
    roof = row["roof"]
    if roof == "TBD(retract)":
        return "roof TBD (retractable -- likely closes in wind; re-check)"
    if roof not in OUTDOOR_ROOFS:
        return f"{roof}"
    if row["wind"] is None:
        return "no wind data" if row["played"] else "no forecast (horizon/offline)"
    return f"wind {row['wind']:g} < {MIN_WIND_DISPLAY:g}"


MIN_WIND_DISPLAY = 11.0  # cosmetic, set from args in main()


def show_week(g, season, week, args):
    rows = evaluate(g, season, week, args, use_forecast=True)
    print(f"\n=== {season} week {week} -- wind-under totals system ===")
    print(f"rule: OUTDOOR game, wind >= {args.min_wind:g} mph -> bet UNDER "
          f"(breakeven {BREAKEVEN}% at -110)")
    if any(r["src"] == "fc" for r in rows):
        print("fc = Open-Meteo forecast at kickoff hour, pulled just now -- "
              "re-check near kickoff")
    print()
    if not rows:
        print("  no games with posted totals for this week yet")
        return
    quals = graded = wins = 0
    for row in sorted(rows, key=lambda x: x["gameday"]):
        wind_s = f"{row['src']}{row['wind']:g}mph" if row["wind"] is not None else "  --  "
        if row["played"]:
            mid = f"{row['score']}, {row['total_str']}"
        else:
            mid = f"O/U {row['total_line']:g}"
        if row["qualifies"]:
            quals += 1
            if row["bet"] in ("WON", "LOST"):
                graded += 1
                wins += row["bet"] == "WON"
            bet_s = f"  bet {row['bet']}" if row["bet"] else ""
            tag = f">>> UNDER {row['total_line']:g} ({band(row['wind'])} mph band) <<<{bet_s}"
        else:
            tag = f"fails: {fail_reason(row)}"
        print(f"  {row['gameday']} {row['weekday']:<9} {row['matchup']:<12} "
              f"{mid:<38} {row['roof']:<12} {wind_s:<10} {tag}")
    print(f"\n  QUALIFIERS: {quals}", end="")
    print(f" | this week: {wins}-{graded - wins}" if graded else "")
    if quals == 0:
        print("  (normal -- league-wide, only a few windy games most weeks)")
    season_record(g, season, week, args)


def season_record(g, season, through_week, args):
    """Forward-test paper trail, graded from ACTUAL recorded wind (see
    module docstring for why live forecast picks can differ)."""
    bets = []
    for w in range(1, through_week + 1):
        for row in evaluate(g, season, w, args, use_forecast=False):
            if row["qualifies"] and row["bet"] in ("WON", "LOST"):
                bets.append((w, row))
    if not bets:
        return
    wins = sum(b[1]["bet"] == "WON" for b in bets)
    n = len(bets)
    profit = sum(profit_100(pd.to_numeric(b[1]["under_odds"], errors="coerce")
                            if pd.notna(b[1]["under_odds"]) else -110.0,
                            b[1]["bet"] == "WON") for b in bets)
    print(f"\n  --- {season} forward test through week {through_week} "
          f"(graded on ACTUAL wind) ---")
    print(f"  record {wins}-{n - wins} ({100 * wins / n:.1f}% under) | "
          f"ROI {profit / n:+5.1f}% per bet")
    for w, row in bets:
        mark = "W " if row["bet"] == "WON" else "L "
        print(f"    {mark}wk{w:<2} {row['matchup']:<12} under {row['total_line']:g} "
              f"(wind {row['wind']:g}, {band(row['wind'])} band; final {row['score']})")


def show_results(g, season, week, args):
    if week is not None:
        weeks = [week]
    else:
        weeks = list(range(1, detect_current_week(g)[1] + 1))
    bets = []
    for w in weeks:
        for row in evaluate(g, season, w, args, use_forecast=False):
            if row["qualifies"]:
                bets.append((w, row))
    scope = f"week {week}" if week is not None else f"weeks 1-{weeks[-1]}"
    print(f"\n=== {season} {scope} -- wind-under results (ACTUAL wind) ===")
    if not bets:
        print("  no qualifying bets")
        return
    print(f"  {'wk':<3}{'date':<12}{'bet':<14}{'wind':>10}  {'total':<18}{'final':<16}result")
    units = w_ = l_ = p_ = 0
    for wk, row in bets:
        tot = (row["total_str"].replace("total ", "") if row["total_str"]
               else f"{row['total_line']:g}")
        odds = pd.to_numeric(row["under_odds"], errors="coerce")
        odds = odds if pd.notna(odds) else -110.0
        if row["bet"] == "WON":
            w_ += 1; u = profit_100(odds, True) / 100; units += u; res = f"WON  +{u:.2f}u"
        elif row["bet"] == "LOST":
            l_ += 1; units -= 1.0; res = "LOST -1.00u"
        elif row["bet"] == "PUSH":
            p_ += 1; res = "PUSH 0.00u"
        else:
            res = "pending"
        print(f"  {wk:<3}{row['gameday']:<12}{row['matchup']:<14}{row['wind']:g}mph"
              f"{'':<3}{tot:<18}{row['score'] or '':<16}{res}")
    decided = w_ + l_
    if decided:
        print(f"\n  record {w_}-{l_}" + (f"-{p_}" if p_ else "") +
              f" ({100 * w_ / decided:.1f}% under) | units {units:+.2f}u | "
              f"ROI {100 * units / decided:+.1f}% per bet" +
              (f" | {p_} push" if p_ else ""))


def backtest(g, args):
    s = g.dropna(subset=["home_score", "away_score", "total_line"]).copy()
    s = s[s["season"].between(args.start_year, 2025)]
    s["actual"] = s["home_score"] + s["away_score"]
    s = s[s["roof"].isin(OUTDOOR_ROOFS) & s["wind"].notna()]
    s = s[s["wind"] >= args.min_wind]
    if s.empty:
        sys.exit("no qualifying games in backtest window")
    decided = s[s["actual"] != s["total_line"]].copy()
    decided["won"] = decided["actual"] < decided["total_line"]
    odds = pd.to_numeric(decided["under_odds"], errors="coerce").fillna(-110.0)
    decided["profit"] = [profit_100(o, w) for o, w in zip(odds, decided["won"])]

    def stat_line(df, label):
        if len(df) < 10:
            return f"  {label:<22}: n={len(df)} (too few)"
        pct = 100 * df["won"].mean()
        roi = 100 * df["profit"].sum() / (100 * len(df))
        verdict = "ABOVE breakeven" if pct >= BREAKEVEN else "below breakeven"
        return (f"  {label:<22}: {pct:5.1f}% under | ROI {roi:+5.1f}% | "
                f"n={len(df)}  -- {verdict}")

    print(f"\n=== Wind-under backtest {args.start_year}-2025 ===")
    print(f"rule: outdoor game, ACTUAL wind >= {args.min_wind:g} mph -> blind under")
    print(f"qualifying games: {len(decided)} decided "
          f"({len(s) - len(decided)} pushes) | breakeven {BREAKEVEN}%\n")
    print(stat_line(decided, "pooled"))
    for lo, hi in [(args.start_year, 2006), (2007, 2013), (2014, 2019), (2020, 2025)]:
        if lo <= hi:
            print(stat_line(decided[decided["season"].between(lo, hi)], f"{lo}-{str(hi)[2:]}"))
    print()
    print(stat_line(decided[decided["wind"] < 16], f"band {args.min_wind:g}-15 mph"))
    print(stat_line(decided[decided["wind"] >= 16], "band 16+ mph"))
    print("\nReminder: the threshold was chosen after seeing these buckets.")
    print("This is the hypothesis being forward-tested -- not a proven edge.")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--min-wind", type=float, default=11.0,
                   help="mph threshold (backtest signal starts at 11)")
    p.add_argument("--no-forecast", action="store_true",
                   help="skip Open-Meteo calls (offline / grading only)")
    p.add_argument("--results", "--result", action="store_true",
                   help="results-only view: one row per bet, graded in units")
    args = p.parse_args()

    global MIN_WIND_DISPLAY
    MIN_WIND_DISPLAY = args.min_wind

    print("  downloading nflverse games.csv ...", file=sys.stderr)
    g = fetch_games()

    if args.backtest:
        backtest(g, args)
        return

    season, week = detect_current_week(g)
    season = args.season or season
    if args.results:
        show_results(g, season, args.week, args)
        return
    week = args.week or week
    if args.week is None and args.season is None:
        print(f"  auto-detected current week: {season} week {week}", file=sys.stderr)
    show_week(g, season, week, args)


if __name__ == "__main__":
    main()
