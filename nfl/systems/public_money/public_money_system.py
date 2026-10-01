"""
Public money system -- weekly bet% vs money% divergence, scraped from Yahoo.

THE IDEA: Action Network sells exactly this ("fade the public" data) --
ticket % vs money % per side. When money% leans harder one way than bet%
does, it means fewer, bigger bets are on that side: the classic "sharp
money" tell. This project already confirmed (see intermediate/qa_log.md,
"fade the public" entry) there is no free HISTORICAL source for this, so
unlike road_dog/wind_under it CANNOT be backtested here -- this tool only
gets the free weekly snapshot. Same honesty rule as predictions_log.csv:
no backtest claim, just a forward paper trail from here on.

DATA SOURCE: sports.yahoo.com's per-game page (?section=odds) publicly
renders a "Full Game Public Bet %" table (bets% and money% for spread /
total / moneyline) in plain server-rendered HTML -- confirmed with a bare
`requests.get`, no JS engine needed. This is still scraping a webpage, not
a documented API (Yahoo's ToS technically disallows automated access), so:
  - keep request volume light -- one page per game per run, ~16/week
  - the CSS class names are build-hashed and WILL change on Yahoo's next
    deploy. This parser matches on visible header text and DOM structure,
    not exact class names, to survive most such changes -- but if a week
    comes back with 0 games parsed, that's the first thing to check.
  - pre-kickoff only: Yahoo's own disclaimer is "data is from bets placed
    before the game began," so this is a snapshot, re-check near kickoff
    like wind_under's forecast.

LOGGING: every run appends its full (unfiltered) set of rows to
public_money_log.csv with a scraped_at timestamp -- no dedup, unlike
predictions_log.csv. This is deliberate: unlike a model prediction, a bet%/
money% reading isn't something re-logging could bias with hindsight, and the
value genuinely moves as kickoff approaches, so repeat runs across a week
are a feature (you can see the split move), not noise. Pass --no-log to
skip writing when you're just eyeballing.

USAGE
-----
  python3 systems/public_money/public_money_system.py            # current week
  python3 systems/public_money/public_money_system.py --week 3   # a specific week
  python3 systems/public_money/public_money_system.py --min-diff 15  # only big splits
  python3 systems/public_money/public_money_system.py --no-log   # don't append to the CSV
  python3 systems/public_money/public_money_system.py --results          # grade every logged week
  python3 systems/public_money/public_money_system.py --results --week 1 # grade one week

RESULTS: --results grades whichever side of each logged market had
money%% > bets%% (the "sharper" side) against the actual final score from
nflverse games.csv. Spread/Total are graded flat -110 (no per-side price in
Yahoo's consensus table); Money Line uses the actual price shown. This is
NOT the same as a backtest -- it can only grade weeks you've already logged
going forward, per the no-free-history limit noted above.
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
SCOREBOARD_URL = "https://sports.yahoo.com/nfl/scoreboard/"
GAME_URL = "https://sports.yahoo.com{path}?section=odds"
GAME_LINK_RE = re.compile(r"/nfl/[a-z0-9-]+-\d{11}/")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "public_money_log.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def fetch_games():
    g = pd.read_csv(GAMES_URL, low_memory=False)
    return g[g["game_type"] == "REG"].copy()


def detect_current_week(g):
    """Same logic as road_dog/wind_under: earliest week of the latest season
    with unplayed games -- used so the logged `week` column is the real
    nflverse week number, not left to Yahoo's own default."""
    season = int(g["season"].max())
    s = g[g["season"] == season]
    unplayed = s[s["home_score"].isna()]
    if unplayed.empty:
        return season, int(s["week"].max())
    return season, int(unplayed["week"].min())


def fetch_week_game_links(week=None):
    """Scrape the Yahoo NFL scoreboard for this week's per-game URL paths."""
    params = {"week": week} if week else {}
    r = requests.get(SCOREBOARD_URL, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return sorted(set(GAME_LINK_RE.findall(r.text)))


def matchup_from_title(soup):
    if soup.title and soup.title.string:
        return soup.title.string.split(":")[0].strip()
    return "?"


def parse_bet_pct_table(soup):
    """Find the 'Full Game Public Bet %' table and return one row per
    (market, side): market name, side label (team + line/odds), bets_pct,
    money_pct -- or [] if the table isn't on the page (e.g. line not
    posted yet, or the game has already kicked off and it was pulled)."""
    target = None
    for table in soup.find_all("table"):
        first_row_labels = [td.get_text(strip=True).upper()
                             for td in table.select("tbody tr:first-child td")]
        if first_row_labels and first_row_labels[0] == "% OF BETS":
            target = table
            break
    if target is None:
        return []

    markets = []  # [(market_name, [side_label, side_label]), ...]
    for th in target.select("thead th")[1:]:  # skip the blank corner cell
        outer_div = th.find("div")
        if outer_div is None:
            continue
        name_spans = outer_div.find_all("span", recursive=False)
        market_name = max((s.get_text(strip=True) for s in name_spans), key=len, default="?")
        inner_div = outer_div.find("div")
        side_labels = [s.get_text(strip=True) for s in inner_div.find_all("span")] if inner_div else []
        markets.append((market_name, side_labels))

    rows_by_label = {}
    for tr in target.select("tbody tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        label = cells[0].get_text(strip=True).upper()
        per_market_pcts = []
        for td in cells[1:]:
            pcts = [d.get_text(strip=True) for d in td.select("div[style*='flex-basis']")]
            per_market_pcts.append(pcts)
        rows_by_label[label] = per_market_pcts

    bets_row = rows_by_label.get("% OF BETS", [])
    money_row = rows_by_label.get("% OF MONEY", [])

    out = []
    for i, (market_name, side_labels) in enumerate(markets):
        bets = bets_row[i] if i < len(bets_row) else []
        money = money_row[i] if i < len(money_row) else []
        for j, side_label in enumerate(side_labels):
            if j >= len(bets) or j >= len(money):
                continue
            try:
                b, m = float(bets[j].rstrip("%")), float(money[j].rstrip("%"))
            except ValueError:
                continue
            out.append({"market": market_name, "side": side_label, "bets_pct": b, "money_pct": m})
    return out


def fetch_game(path):
    r = requests.get(GAME_URL.format(path=path), headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return matchup_from_title(soup), parse_bet_pct_table(soup)


def log_rows(rows, path=LOG_PATH):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"\nLogged {len(df)} rows to {os.path.abspath(path)}")


# --- grading (--results) -----------------------------------------------

YAHOO_TEAM_FIX = {"LAR": "LA"}  # nflverse uses "LA" for the Rams; Yahoo uses "LAR"

TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
SPREAD_RE = re.compile(r"^([A-Z]{2,3})\s+([+-]?\d+\.?\d*)$")
TOTAL_RE = re.compile(r"^([OU])\s+(\d+\.?\d*)$")
ML_RE = re.compile(r"^([A-Z]{2,3})\s+([+-]\d+)$")


def parse_side(market, side):
    """'ATL +2.5' -> ('ATL', 2.5); 'O 43.5' -> ('O', 43.5); 'CAR -150' -> ('CAR', -150)."""
    if market == "Total":
        m = TOTAL_RE.match(side)
        return (m.group(1), float(m.group(2))) if m else (None, None)
    m = SPREAD_RE.match(side) or ML_RE.match(side)
    if not m:
        return None, None
    return YAHOO_TEAM_FIX.get(m.group(1), m.group(1)), float(m.group(2))


def matchup_team_codes(matchup):
    """'Away Team @ Home Team' (pregame) or 'Away Team NN - Home Team NN' (final)
    -> (away_code, home_code), or (None, None) if a name doesn't map."""
    if "@" in matchup:
        left, right = matchup.split("@", 1)
    elif " - " in matchup:
        left, right = matchup.split(" - ", 1)
    else:
        return None, None
    clean = lambda name: re.sub(r"\s*\d+\s*$", "", name).strip()
    return TEAM_NAME_TO_CODE.get(clean(left)), TEAM_NAME_TO_CODE.get(clean(right))


SLUG_TO_CODE = {name.lower().replace(" ", "-"): code for name, code in TEAM_NAME_TO_CODE.items()}


def link_team_codes(path):
    """Derive the two team codes mentioned in a scoreboard link's slug, e.g.
    '/nfl/washington-commanders-dallas-cowboys-20260920006/' -> {'WAS','DAL'}.
    Used by --only-missing to tell which already-scraped game a link is,
    without needing to fetch it again first."""
    return frozenset(code for slug, code in SLUG_TO_CODE.items() if slug in path)


def logged_team_code_pairs(log, season, week):
    """{frozenset({away_code, home_code}), ...} for matchups already logged
    this week, so --only-missing can skip re-fetching them."""
    pairs = set()
    wk = log[(log["season"] == season) & (log["week"] == week)]
    for matchup in wk["matchup"].unique():
        away, home = matchup_team_codes(matchup)
        if away and home:
            pairs.add(frozenset({away, home}))
    return pairs


def build_results_lookup(g):
    """(season, week, team_code) -> (team_score, opp_score), games with a final only."""
    lookup = {}
    for _, r in g.dropna(subset=["home_score", "away_score"]).iterrows():
        season, week = int(r["season"]), int(r["week"])
        lookup[(season, week, r["home_team"])] = (r["home_score"], r["away_score"])
        lookup[(season, week, r["away_team"])] = (r["away_score"], r["home_score"])
    return lookup


def ml_profit_100(odds, won):
    return (100 * 100 / -odds) if (won and odds < 0) else (odds if won else -100.0)


def show_results(args):
    if not os.path.exists(LOG_PATH):
        sys.exit(f"no log yet at {os.path.abspath(LOG_PATH)} -- run without --results first")
    log = pd.read_csv(LOG_PATH)
    if args.season:
        log = log[log["season"] == args.season]
    if args.week:
        log = log[log["week"] == args.week]
    if log.empty:
        sys.exit("no logged rows match that season/week")
    # a game may have been scraped more than once in a week; grade the first snapshot
    log = log.sort_values("scraped_at").drop_duplicates(
        subset=["season", "week", "matchup", "market", "side"], keep="first")

    print("  downloading nflverse games.csv (for actual results) ...", file=sys.stderr)
    lookup = build_results_lookup(fetch_games())

    scope = f"week {args.week}" if args.week else "all logged weeks"
    print(f"\n=== Public-money results -- {scope} ===")
    print("Pick = the side where money%% > bets%% (the 'sharper' side). Spread/Total")
    print("graded flat -110 (no per-side price in Yahoo's table); Money Line uses the")
    print("actual price shown. NOT a backtest -- only weeks you've logged so far.\n")

    wins = losses = pushes = ungraded = 0
    units = 0.0
    html_rows = []
    for (season, week, matchup), game_grp in log.groupby(["season", "week", "matchup"]):
        away_code, home_code = matchup_team_codes(matchup)
        away_score_pair = lookup.get((season, week, away_code)) if away_code else None
        final_score = (f"{away_code} {away_score_pair[0]:g} - {home_code} {away_score_pair[1]:g}"
                       if away_score_pair else "not yet played")
        printed_matchup = False
        for market, mkt_grp in game_grp.groupby("market"):
            if len(mkt_grp) != 2:
                continue
            sharp = mkt_grp.loc[mkt_grp["diff"].idxmax()]
            if sharp["diff"] <= 0:
                continue  # exact 50/50 split, no sharper side
            team_or_ou, value = parse_side(market, sharp["side"])
            if team_or_ou is None:
                continue

            if market == "Total":
                score = lookup.get((season, week, away_code)) or lookup.get((season, week, home_code))
                if score is None:
                    ungraded += 1
                    html_rows.append({"Week": week, "Matchup": matchup, "Market": market,
                                      "Pick": f"{team_or_ou} {value:g}",
                                      "Bets%": sharp["bets_pct"], "Money%": sharp["money_pct"],
                                      "Diff": sharp["diff"],
                                      "Final": final_score, "Result": "pending"})
                    continue
                actual = score[0] + score[1]
                result = "PUSH" if actual == value else (
                    "WON" if (team_or_ou == "O") == (actual > value) else "LOST")
                pick_label = f"{team_or_ou} {value:g}"
            else:
                score = lookup.get((season, week, team_or_ou))
                if score is None:
                    ungraded += 1
                    html_rows.append({"Week": week, "Matchup": matchup, "Market": market,
                                      "Pick": sharp["side"],
                                      "Bets%": sharp["bets_pct"], "Money%": sharp["money_pct"],
                                      "Diff": sharp["diff"],
                                      "Final": final_score, "Result": "pending"})
                    continue
                team_score, opp_score = score
                if market == "Money Line":
                    result = "PUSH" if team_score == opp_score else (
                        "WON" if team_score > opp_score else "LOST")
                else:  # Spread
                    cover = (team_score - opp_score) + value
                    result = "PUSH" if cover == 0 else ("WON" if cover > 0 else "LOST")
                pick_label = f"{team_or_ou} {value:+g}"

            if not printed_matchup:
                print(f"  {matchup}")
                printed_matchup = True
            print(f"    {market:<11} pick {pick_label:<10} diff {sharp['diff']:+5.1f}  {result}")
            html_rows.append({"Week": week, "Matchup": matchup, "Market": market,
                              "Pick": pick_label,
                              "Bets%": sharp["bets_pct"], "Money%": sharp["money_pct"],
                              "Diff": sharp["diff"],
                              "Final": final_score, "Result": result})

            if result == "PUSH":
                pushes += 1
            elif result == "WON":
                wins += 1
                units += (ml_profit_100(value, True) / 100 if market == "Money Line" else 90.9 / 100)
            else:
                losses += 1
                units -= 1.0

    decided = wins + losses
    print(f"\n  record {wins}-{losses}" + (f"-{pushes}" if pushes else "")
          + (f" ({100 * wins / decided:.1f}% following sharper money)" if decided else "")
          + (f" | units {units:+.2f}u" if decided else "")
          + (f" | {ungraded} not yet played/ungraded" if ungraded else ""))

    if args.html:
        write_html_table(html_rows, os.path.expanduser(args.html), scope,
                         summary=(f"record {wins}-{losses}" + (f"-{pushes}" if pushes else "")
                                 + (f" ({100 * wins / decided:.1f}%)" if decided else "")
                                 + (f" | units {units:+.2f}u" if decided else "")))


def write_html_table(rows, path, scope, summary=""):
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Week", "Matchup", "Market"])
        df["Diff"] = df["Diff"].map(lambda d: f"{d:+.1f}")
        df["Bets%"] = df["Bets%"].map(lambda v: f"{v:.1f}%")
        df["Money%"] = df["Money%"].map(lambda v: f"{v:.1f}%")

    def row_color(result):
        return {"WON": "#e6f4ea", "LOST": "#fce8e6", "PUSH": "#fff8e1"}.get(result, "#f1f1f1")

    body_rows = "".join(
        f"<tr style='background:{row_color(r.Result)}'>"
        f"<td>{r.Week}</td><td>{r.Matchup}</td><td>{r.Market}</td><td>{r.Pick}</td>"
        f"<td>{r._5}</td><td>{r._6}</td><td>{r.Diff}</td><td>{r.Final}</td><td><b>{r.Result}</b></td></tr>"
        for r in df.itertuples()
    ) if not df.empty else "<tr><td colspan='9'>No graded rows yet</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Public Money Results -- {scope}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 20px; }}
  .summary {{ font-size: 15px; margin-bottom: 16px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #333; color: white; position: sticky; top: 0; }}
  tr:hover {{ filter: brightness(0.96); }}
</style></head>
<body>
  <h1>Public Money Results -- {scope}</h1>
  <div class="summary">Pick = side where money% &gt; bets% (the "sharper" side). {summary}</div>
  <table>
    <thead><tr><th>Week</th><th>Matchup</th><th>Market</th><th>Pick</th><th>Bets%</th><th>Money%</th><th>Diff</th><th>Final</th><th>Result</th></tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"\nWrote HTML table to {os.path.abspath(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, default=None, help="default: auto-detect (nflverse)")
    ap.add_argument("--season", type=int, default=None, help="--results only: default all seasons logged")
    ap.add_argument("--min-diff", type=float, default=0.0,
                    help="only print sides where |money%% - bets%%| >= this (logging is unfiltered)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between per-game requests")
    ap.add_argument("--no-log", action="store_true", help="don't append to public_money_log.csv")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip games already logged for this week (backfill gaps without re-scraping everything)")
    ap.add_argument("--results", action="store_true",
                    help="grade logged weeks instead of scraping a new one")
    ap.add_argument("--html", nargs="?", const="~/Downloads/public_money_results.html", default=None,
                    help="--results only: also write a browser-viewable table (default path if no value given)")
    args = ap.parse_args()

    if args.results:
        show_results(args)
        return

    print("  downloading nflverse games.csv (for season/week) ...", file=sys.stderr)
    season, week = detect_current_week(fetch_games())
    week = args.week or week

    print("  fetching Yahoo NFL scoreboard ...", file=sys.stderr)
    links = fetch_week_game_links(week)
    if not links:
        sys.exit("no games found -- Yahoo scoreboard page structure may have changed")

    if args.only_missing:
        if not os.path.exists(LOG_PATH):
            sys.exit("--only-missing needs an existing public_money_log.csv -- run without it first")
        already = logged_team_code_pairs(pd.read_csv(LOG_PATH), season, week)
        before = len(links)
        links = [p for p in links if link_team_codes(p) not in already]
        print(f"  --only-missing: {before - len(links)} of {before} games already logged, "
              f"fetching the remaining {len(links)}", file=sys.stderr)
        if not links:
            sys.exit("nothing missing -- every game this week is already logged")

    print(f"\n=== Public bet%/money%% splits -- {season} week {week} ({len(links)} games) ===")
    print("(source: sports.yahoo.com odds tab; pre-kickoff snapshot, not backtestable --")
    print(" see systems/public_money/public_money_system.py docstring)\n")

    scraped_at = pd.Timestamp.now().isoformat(timespec="seconds")
    log_buffer = []
    any_rows = False
    for i, path in enumerate(links):
        if i:
            time.sleep(args.delay)
        try:
            matchup, rows = fetch_game(path)
            if not rows:  # one retry -- most misses are a transient rate-limit blip
                time.sleep(max(args.delay, 2.0))
                matchup, rows = fetch_game(path)
        except requests.RequestException as e:
            print(f"  {path}: fetch failed ({type(e).__name__})", file=sys.stderr)
            continue
        if not rows:
            print(f"  {matchup}: no bet%% table after retry (line not posted yet, or already kicked off)")
            continue
        printed_header = False
        for row in rows:
            diff = round(row["money_pct"] - row["bets_pct"], 2)
            log_buffer.append({"season": season, "week": week, "matchup": matchup,
                               "market": row["market"], "side": row["side"],
                               "bets_pct": row["bets_pct"], "money_pct": row["money_pct"],
                               "diff": diff, "scraped_at": scraped_at})
            if abs(diff) < args.min_diff:
                continue
            if not printed_header:
                print(f"  {matchup}")
                printed_header = True
            any_rows = True
            flag = "  <-- sharper money" if abs(diff) >= 15 else ""
            print(f"    {row['market']:<11} {row['side']:<14} "
                  f"bets {row['bets_pct']:5.1f}%  money {row['money_pct']:5.1f}%  "
                  f"diff {diff:+5.1f}{flag}")
    if not any_rows:
        print("  (nothing cleared --min-diff)" if args.min_diff else "  no data parsed")

    if log_buffer and not args.no_log:
        log_rows(log_buffer)


if __name__ == "__main__":
    main()
