"""
Road-dog small-spread system -- weekly qualifier + historical validation.

THE SYSTEM (reverse-engineered from an Action Network trend, then refined
against 26 seasons of nflverse lines in backtest_baddogs.py):

  Bet a ROAD underdog getting +1 to +3.5 when
    - the dog won ~5 or fewer games last season (prior win% <= .294), and
    - the game total is under 45, and
    - the game is Sunday or Monday (short-week games excluded).

The logic: a team that went 3-14 "should" be a 6-7 point road underdog.
When the market hangs only +2.5, the line itself is leaking that the
experts rate them far better than last year's record -- the public is
betting last year's team, you're betting this year's team with points.

Backtest 2000-2025 (nflverse closing lines): ~58-62% cover depending on
thresholds, positive in every era, vs 52.4% breakeven at -110. BUT the
parameters were chosen after seeing the system's picks -- treat this as a
hypothesis being forward-tested, not a proven edge. It makes ~5-8 bets
per SEASON; weeks with zero qualifiers are normal.

For played games the week view also shows the final score, whether each
qualifier's bet WON/LOST/PUSHED (flat -110 grading), and the over/under
result vs the closing total, plus the season-to-date forward-test record
(the paper trail that decides whether the backtest was signal or noise).
The deep box-score stats (yards, turnovers, red zone) stay in the
streamlit app -- this tool tracks the bet, not the game film.

Data: nflverse games.csv (free, updated weekly; 100% spread/total
coverage back to 1999). No API key, no scraping.

USAGE
-----
  python3 systems/road_dog_system.py                 # this week's qualifiers
  python3 systems/road_dog_system.py --week 1        # past week, graded
  python3 systems/road_dog_system.py --results       # every bet + outcome, season
  python3 systems/road_dog_system.py --results --week 1
  python3 systems/road_dog_system.py --backtest      # historical validation
  python3 systems/road_dog_system.py --total-max 46  # loosen any filter
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from srs import blended_asof_ratings  # noqa: E402  (needs the sys.path fix above)

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
BREAKEVEN = 52.4  # cover % needed at standard -110 juice
WIN_PAYOUT = 90.9  # flat 100 stake at -110 returns 90.9 profit


def fetch_games():
    g = pd.read_csv(GAMES_URL, low_memory=False)
    return g[g["game_type"] == "REG"].copy()


def season_records(g, season):
    """team -> win pct that season (ties count half)."""
    s = g[(g["season"] == season)].dropna(subset=["home_score", "away_score"])
    rec = {}
    for _, r in s.iterrows():
        rec.setdefault(r["home_team"], [0.0, 0.0])
        rec.setdefault(r["away_team"], [0.0, 0.0])
        if r["home_score"] > r["away_score"]:
            rec[r["home_team"]][0] += 1; rec[r["away_team"]][1] += 1
        elif r["home_score"] < r["away_score"]:
            rec[r["away_team"]][0] += 1; rec[r["home_team"]][1] += 1
        else:
            rec[r["home_team"]][0] += 0.5; rec[r["away_team"]][0] += 0.5
            rec[r["home_team"]][1] += 0.5; rec[r["away_team"]][1] += 0.5
    return {t: w / (w + l) for t, (w, l) in rec.items() if (w + l) > 0}


def evaluate(g, season, week, args, prior=None, srs_ratings=None):
    """One row per game in the week with per-filter verdicts, plus final
    score / over-under result / bet result when the game has been played.

    `srs_ratings` (dog -> current-season, no-leakage SRS rating as of this
    week, from blended_asof_ratings()) is display-only -- it does not
    affect `qualifies`. It's there so prior-season win% (static all
    season) can be eyeballed next to how the team rates *right now*
    (updates weekly)."""
    if prior is None:
        prior = season_records(g, season - 1)
    wk = g[(g["season"] == season) & (g["week"] == week)].copy()
    wk = wk.dropna(subset=["spread_line", "total_line"])
    rows = []
    for _, r in wk.iterrows():
        if r["spread_line"] == 0:
            continue  # pick'em: no dog
        dog_home = r["spread_line"] < 0
        dog = r["home_team"] if dog_home else r["away_team"]
        pts = abs(r["spread_line"])
        p = prior.get(dog)
        srs = srs_ratings.get(dog) if srs_ratings else None
        checks = {
            "Sun/Mon": r["weekday"] in ("Sunday", "Monday"),
            "away=underdog": (not dog_home) or args.include_home,
            f"+{args.min_pts:g}-{args.max_pts:g}": args.min_pts <= pts <= args.max_pts,
            f"prior<={args.prior_max:g}": p is not None and round(p, 3) <= args.prior_max,
            f"total<{args.total_max:g}": r["total_line"] < args.total_max,
        }
        played = pd.notna(r["home_score"]) and pd.notna(r["away_score"])
        row = {
            "gameday": r["gameday"], "weekday": r["weekday"],
            "away": r["away_team"], "home": r["home_team"],
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "spread_line": r["spread_line"], "total_line": r["total_line"],
            "dog": dog, "dog_home": dog_home, "pts": pts, "prior": p, "srs": srs,
            "checks": checks, "qualifies": all(checks.values()),
            "played": played, "score": None, "total_str": None, "bet": None,
        }
        if played:
            a_sc, h_sc = int(r["away_score"]), int(r["home_score"])
            actual_total = a_sc + h_sc
            row["score"] = f"{r['away_team']} {a_sc} - {r['home_team']} {h_sc}"
            word = ("over" if actual_total > r["total_line"]
                    else "under" if actual_total < r["total_line"] else "push")
            row["total_str"] = f"total {actual_total} ({word} {r['total_line']:g})"
            beat = (h_sc - a_sc) - r["spread_line"]  # >0: home covers
            if beat == 0:
                row["bet"] = "PUSH"
            else:
                covered = (beat < 0) if not dog_home else (beat > 0)
                row["bet"] = "WON" if covered else "LOST"
        rows.append(row)
    return rows


def detect_current_week(g):
    """Earliest week of the latest season with unplayed games that have lines."""
    season = int(g["season"].max())
    s = g[(g["season"] == season)].dropna(subset=["spread_line"])
    unplayed = s[s["home_score"].isna()]
    if unplayed.empty:
        return season, int(s["week"].max())
    return season, int(unplayed["week"].min())


def show_week(g, season, week, args):
    prior = season_records(g, season - 1)
    srs_by_week = blended_asof_ratings(g, [season])
    rows = evaluate(g, season, week, args, prior, srs_by_week.get((season, week), {}))
    print(f"\n=== {season} week {week} -- road-dog small-spread system ===")
    print(f"filters: Sun/Mon, away=underdog, +{args.min_pts:g} to +{args.max_pts:g}, "
          f"prior win% <= {args.prior_max:g}, total < {args.total_max:g}\n")
    if not rows:
        print("  no games with posted lines for this week yet")
        return
    quals = graded = wins = 0
    for row in sorted(rows, key=lambda x: x["gameday"]):
        fails = [k for k, v in row["checks"].items() if not v]
        prior_s = f"{row['prior']:.3f}" if row["prior"] is not None else " n/a "
        srs_s = f"{row['srs']:+.1f}" if row["srs"] is not None else " n/a "
        if row["played"]:
            mid = f"{row['score']}, {row['total_str']}"
        else:
            mid = f"tot {row['total_line']:g}"
        if row["qualifies"]:
            quals += 1
            if row["bet"] in ("WON", "LOST"):
                graded += 1
                wins += row["bet"] == "WON"
            bet_s = f"  bet {row['bet']}" if row["bet"] else ""
            tag = f">>> QUALIFIES <<<{bet_s}"
        else:
            tag = f"fails: {', '.join(fails)}"
        print(f"  {row['gameday']} {row['weekday']:<9} {row['matchup']:<12} {mid:<37} "
              f"{row['dog']} +{row['pts']:g} (prior {prior_s}, srs {srs_s})  {tag}")
    print(f"\n  QUALIFIERS: {quals}", end="")
    print(f" | this week: {wins}-{graded - wins}" if graded else "")
    if quals == 0:
        print("  (normal -- this system only finds ~5-8 bets per season)")
    season_record(g, season, week, args, prior)


def season_record(g, season, through_week, args, prior):
    """The forward-test paper trail: every qualifier from week 1 through the
    requested week, graded. This is the log that decides whether the
    2000-2025 backtest was signal or noise."""
    bets = []
    for w in range(1, through_week + 1):
        for row in evaluate(g, season, w, args, prior):
            if row["qualifies"] and row["bet"] in ("WON", "LOST"):
                bets.append((w, row))
    if not bets:
        return
    wins = sum(b[1]["bet"] == "WON" for b in bets)
    n = len(bets)
    profit = sum(WIN_PAYOUT if b[1]["bet"] == "WON" else -100.0 for b in bets)
    print(f"\n  --- {season} forward test through week {through_week} ---")
    print(f"  record {wins}-{n - wins} ({100 * wins / n:.1f}% cover) | "
          f"flat -110 ROI {profit / n:+5.1f}% per bet")
    for w, row in bets:
        mark = "W " if row["bet"] == "WON" else "L "
        print(f"    {mark}wk{w:<2} {row['dog']} +{row['pts']:g} {row['matchup']} "
              f"(final {row['score']})")


def show_results(g, season, week, args):
    """Results-only view: one row per qualifying bet (no per-game checklist),
    graded in units (1u stake at -110, like the Action Network pick pages)."""
    prior = season_records(g, season - 1)
    if week is not None:
        weeks = [week]
    else:
        weeks = list(range(1, detect_current_week(g)[1] + 1))
    bets = []
    for w in weeks:
        for row in evaluate(g, season, w, args, prior):
            if row["qualifies"]:
                bets.append((w, row))
    scope = f"week {week}" if week is not None else f"weeks 1-{weeks[-1]}"
    print(f"\n=== {season} {scope} -- system results ===")
    if not bets:
        print("  no qualifying bets")
        return
    print(f"  {'wk':<3}{'date':<12}{'bet':<14}{'line':>5}  {'total':<18}{'final':<16}result")
    units = w = l = p_ = 0
    for wk, row in bets:
        tot = (row["total_str"].replace("total ", "") if row["total_str"]
               else f"{row['total_line']:g}")
        if row["bet"] == "WON":
            w += 1; units += WIN_PAYOUT / 100; res = f"WON  +{WIN_PAYOUT/100:.2f}u"
        elif row["bet"] == "LOST":
            l += 1; units -= 1.0; res = "LOST -1.00u"
        elif row["bet"] == "PUSH":
            p_ += 1; res = "PUSH 0.00u"
        else:
            res = "pending"
        print(f"  {wk:<3}{row['gameday']:<12}{row['matchup']:<14}+{row['pts']:g}   "
              f"{tot:<18}{row['score'] or '':<16}{res}")
    decided = w + l
    if decided:
        print(f"\n  record {w}-{l}" + (f"-{p_}" if p_ else "") +
              f" ({100 * w / decided:.1f}% cover) | units {units:+.2f}u | "
              f"ROI {100 * units / decided:+.1f}% per bet" +
              (f" | {p_} push" if p_ else ""))


def backtest(g, args):
    last = int(g["season"].max()) - 1
    seasons = list(range(args.start_year, last + 1))
    rows = []
    for season in seasons:
        prior = season_records(g, season - 1)
        s = g[g["season"] == season].dropna(subset=["spread_line", "total_line",
                                                    "home_score", "away_score"])
        for _, r in s.iterrows():
            if r["spread_line"] == 0:
                continue
            dog_home = r["spread_line"] < 0
            pts = abs(r["spread_line"])
            p = prior.get(r["home_team"] if dog_home else r["away_team"])
            if p is None or round(p, 3) > args.prior_max:
                continue
            if not (args.min_pts <= pts <= args.max_pts):
                continue
            if r["total_line"] >= args.total_max:
                continue
            if r["weekday"] not in ("Sunday", "Monday"):
                continue
            if dog_home and not args.include_home:
                continue
            margin = r["home_score"] - r["away_score"]  # home-positive
            beat = margin - r["spread_line"]            # >0: home covers
            rows.append({"season": season, "push": beat == 0,
                         "won": (beat < 0) if not dog_home else (beat > 0)})
    if not rows:
        sys.exit("no qualifying bets in backtest window")
    df = pd.DataFrame(rows)
    decided = df[~df["push"]]

    def stat_line(mask, label):
        sub = decided[mask.reindex(decided.index, fill_value=False)]
        if len(sub) == 0:
            return f"  {label}: no bets"
        pct = 100 * sub["won"].mean()
        roi = 100 * ((sub["won"].map({True: WIN_PAYOUT, False: -100})).sum()) / (100 * len(sub))
        verdict = "ABOVE breakeven" if pct >= BREAKEVEN else "below breakeven"
        return f"  {label}: {pct:5.1f}% cover | ROI {roi:+5.1f}% | n={len(sub)}  -- {verdict}"

    print(f"\n=== Backtest {seasons[0]}-{seasons[-1]} (flat -110; breakeven {BREAKEVEN}%) ===")
    print(f"filters: Sun/Mon, {'any dog' if args.include_home else 'away=underdog'}, "
          f"+{args.min_pts:g} to +{args.max_pts:g}, prior win% <= {args.prior_max:g}, "
          f"total < {args.total_max:g}")
    print(f"qualifying bets: {len(df)} ({int(df['push'].sum())} pushes)\n")
    allmask = pd.Series(True, index=df.index)
    print(stat_line(allmask, "pooled"))
    for lo, hi in [(seasons[0], 2009), (2010, 2019), (2020, seasons[-1])]:
        if lo <= hi:
            print(stat_line(df["season"].between(lo, hi), f"{lo}-{str(hi)[2:]}"))
    print("\nReminder: parameters were chosen post-hoc. This is the hypothesis")
    print("being forward-tested -- not a proven edge.")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=None, help="default: auto-detect")
    p.add_argument("--week", type=int, default=None, help="default: current week")
    p.add_argument("--backtest", action="store_true", help="validate historically")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--min-pts", type=float, default=1.0)
    p.add_argument("--max-pts", type=float, default=3.5)
    p.add_argument("--prior-max", type=float, default=0.294,
                   help="max prior-season win%% (0.294 ~ 5 wins in 17 games)")
    p.add_argument("--total-max", type=float, default=45.0)
    p.add_argument("--include-home", action="store_true",
                   help="also allow home dogs (historically much weaker)")
    p.add_argument("--results", "--result", action="store_true",
                   help="results-only view: one row per bet, graded in units")
    args = p.parse_args()

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
