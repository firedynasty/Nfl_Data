"""
SRS (Simple Rating System): one opponent-adjusted rating per team, in points.

    rating[team] = avg scoring margin[team] + avg(rating of opponents faced)

solved iteratively until the numbers stop moving, re-centered to league
mean 0 each pass. Needs only final scores, so it builds straight from
build_long_results() in nfl_box_score_analysis.py -- no play-by-play
download. Includes playoff games and ignores home/away splits, matching
Pro-Football-Reference's published SRS, which is the sanity-check target.

A rating of +6.2 reads as "about 6.2 points better than an average team" --
same units as a point spread, which is the whole point of the pipeline
(see intermediate/plan.md).

USAGE
-----
  python srs.py --season 2025                 # one season's SRS table
  python srs.py --season 2025 --cap 24        # cap each game's margin at +/-24 first
  python srs.py --season 2025 --validate      # in-sample check: SRS vs raw avg margin
  python srs.py --season 2026 --asof 3        # no-leakage blended ratings as of week 3
"""

import argparse
import os
import sys

import pandas as pd


def build_long_results(games, seasons):
    """Reshape a games table (one row per game, nflverse games.csv shape)
    into one row per team per game, with a W/L/T result column. Lives here
    (not in the NFL toolkit) because it's sport-agnostic: the college
    pipeline feeds the same shape from ESPN. nfl/nfl_box_score_analysis.py
    re-exports it so older imports keep working."""
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g["season"] = g["season"].astype(int)
    g = g[g["season"].isin(seasons)]

    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
    home = g[cols].copy()
    home["team"], home["opp"] = home["home_team"], home["away_team"]
    home["team_score"], home["opp_score"] = home["home_score"], home["away_score"]

    away = g[cols].copy()
    away["team"], away["opp"] = away["away_team"], away["home_team"]
    away["team_score"], away["opp_score"] = away["away_score"], away["home_score"]

    long = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "week", "team", "opp", "team_score", "opp_score"]
    ]
    long["result"] = long.apply(
        lambda r: "W" if r.team_score > r.opp_score else ("L" if r.team_score < r.opp_score else "T"),
        axis=1,
    )
    return long


def game_margins(long, season, cap=None):
    """One row per team-game with margin = team_score - opp_score,
    optionally capped at +/-cap so a single blowout can't dominate a rating."""
    g = long[long["season"] == season].copy()
    g["margin"] = g["team_score"] - g["opp_score"]
    if cap is not None:
        g["margin"] = g["margin"].clip(-cap, cap)
    return g[["team", "opp", "margin"]].reset_index(drop=True)


def solve_srs(margins, tol=1e-9, max_iter=10000, damping=0.5):
    """The iterative solve. Opponents are averaged over GAMES (facing the
    same team twice counts twice), not over distinct opponents -- that's
    standard SRS and it's what makes the schedule correction exact.

    The damped update (half-step toward the target each pass) is not
    optional: on tiny early-season samples (~1 game per team) the game
    graph is a near-perfect-matching and the undamped iteration period-2
    oscillates forever. The damped fixed point is identical."""
    teams = sorted(set(margins["team"]))
    opps = {t: list(margins.loc[margins["team"] == t, "opp"]) for t in teams}
    avg_margin = margins.groupby("team")["margin"].mean().to_dict()

    rating = {t: 0.0 for t in teams}
    for it in range(1, max_iter + 1):
        target = {t: avg_margin[t] + sum(rating[o] for o in opps[t]) / len(opps[t]) for t in teams}
        t_mean = sum(target.values()) / len(target)
        new = {t: rating[t] + damping * ((target[t] - t_mean) - rating[t]) for t in teams}
        n_mean = sum(new.values()) / len(new)
        biggest_change = max(abs((new[t] - n_mean) - rating[t]) for t in teams)
        rating = {t: new[t] - n_mean for t in teams}
        if biggest_change < tol:
            return rating, it
    print(f"warning: SRS did not converge within {max_iter} iterations", file=sys.stderr)
    return rating, max_iter


def solve_srs_asof(long, season, week, cap=None):
    """SRS solved only from games with `week < W` in `season` -- the
    no-leakage version: the rating knows nothing that happened at or
    after the week being predicted. Returns (ratings, games_played)."""
    g = long[(long["season"] == season) & (long["week"] < week)]
    if g.empty:
        return {}, {}
    rating, _ = solve_srs(game_margins(g, season, cap=cap))
    return rating, g.groupby("team").size().to_dict()


def blended_asof_ratings(games, seasons, cap=None, k=4.0, shrink=0.7):
    """Per-(season, week) ratings with no leakage, blended for the cold
    start. Early weeks have 0-2 games per team, where a raw as-of solve
    is noise, so the current season is ramped in against the prior full
    season (regressed toward 0 for year-over-year roster turnover):

        rating = g/(g+k) * srs_current + k/(g+k) * shrink * srs_prior

    week 1 (g=0) is entirely the regressed prior; the current season
    takes over as games accumulate. k = prior strength in pseudo-games,
    shrink = year-over-year regression -- both placeholder constants
    pending a backtest sweep, same discipline as the edge threshold."""
    seasons = sorted(seasons)
    long = build_long_results(games, [seasons[0] - 1] + seasons)
    out = {}
    for season in seasons:
        prior_games = long[long["season"] == season - 1]
        prior = (solve_srs(game_margins(prior_games, season - 1, cap=cap))[0]
                 if not prior_games.empty else {})
        cur_long = long[long["season"] == season]
        # Teams and weeks come from the SCHEDULE (scores or not), so the
        # table also covers weeks that haven't been played yet -- that's
        # what lets Phase 4 ask for ratings as of an upcoming week.
        sched = games[games["season"].astype(int) == season]
        teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
        for week in sorted(sched["week"].unique()):
            cur, played = solve_srs_asof(long, season, week, cap=cap)
            rating = {}
            for t in teams:
                p = prior.get(t, 0.0) * shrink
                g_played = played.get(t, 0)
                if g_played > 0 and t in cur:
                    w = g_played / (g_played + k)
                    rating[t] = w * cur[t] + (1 - w) * p
                else:
                    rating[t] = p
            out[(season, week)] = rating
    return out


def strength_of_schedule(margins, rating):
    """Average opponent rating per team from the final ratings.
    Identity check: srs = avg margin + sos, per the defining equation."""
    m = margins.assign(opp_rating=margins["opp"].map(rating))
    return m.groupby("team")["opp_rating"].mean()


def srs_table(margins_raw, rating):
    """The 32-row table: SRS next to raw average margin (mov) and strength
    of schedule, so the schedule correction is visible per team."""
    df = pd.DataFrame({
        "srs": pd.Series(rating),
        "mov": margins_raw.groupby("team")["margin"].mean(),
        "sos": strength_of_schedule(margins_raw, rating),
    }).sort_values("srs", ascending=False)
    return df.round(2)


def validate(games, season, rating, margins_raw):
    """In-sample check ONLY: season-total ratings have already seen every
    game, so this proves the math works, not that it predicts -- the real
    no-leakage bar is Phase 5's walk-forward. For each game, predict the
    home margin two ways (SRS difference vs raw-avg-margin difference, no
    home-field term in either) and see which tracks actual margins better.
    SRS should win; the gap is the schedule correction paying for itself."""
    raw_avg = margins_raw.groupby("team")["margin"].mean().to_dict()
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g = g[g["season"].astype(int) == season]
    g["actual"] = g["home_score"] - g["away_score"]
    g["srs_pred"] = g["home_team"].map(rating) - g["away_team"].map(rating)
    g["raw_pred"] = g["home_team"].map(raw_avg) - g["away_team"].map(raw_avg)

    out = pd.DataFrame({
        "corr_with_actual": [g["srs_pred"].corr(g["actual"]), g["raw_pred"].corr(g["actual"])],
        "mae_points": [(g["srs_pred"] - g["actual"]).abs().mean(),
                       (g["raw_pred"] - g["actual"]).abs().mean()],
    }, index=["SRS", "raw avg margin"]).round(3)
    print(f"\n=== Validation: per-game home-margin prediction, {season} (in-sample) ===")
    print(out.to_string())
    print("\nSRS should at least match raw margin here; the bar that matters is Phase 5.")


def main():
    p = argparse.ArgumentParser(description="SRS (Simple Rating System) ratings from final scores")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--cap", type=float, default=None,
                   help="cap each game's margin at +/-this before averaging (e.g. 24)")
    p.add_argument("--validate", action="store_true",
                   help="compare SRS vs raw avg margin on per-game prediction")
    p.add_argument("--asof", type=int, default=None, metavar="WEEK",
                   help="no-leakage blended ratings as of this week (games before it only)")
    p.add_argument("--k", type=float, default=4.0,
                   help="cold-start prior strength in pseudo-games (with --asof)")
    p.add_argument("--shrink", type=float, default=0.7,
                   help="year-over-year regression of prior-season SRS (with --asof)")
    args = p.parse_args()

    # NFL-only CLI: the games source lives in the nfl/ toolkit.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nfl"))
    from nfl_box_score_analysis import load_games
    games = load_games()

    if args.asof is not None:
        table = blended_asof_ratings(games, [args.season], cap=args.cap,
                                     k=args.k, shrink=args.shrink)
        key = (args.season, args.asof)
        if key not in table:
            sys.exit(f"week {args.asof} not found in season {args.season}")
        rating = table[key]
        print(f"\n=== SRS as of {args.season} week {args.asof} "
              f"(no leakage: games before week {args.asof} only) ===")
        print(f"cold-start blend: k={args.k:g} pseudo-games, prior-season shrink "
              f"{args.shrink:g}" + ("  <- week 1: ratings are prior season x shrink only"
                                    if args.asof == 1 else "") + "\n")
        df = (pd.Series(rating, name="srs_asof").rename_axis("team")
              .reset_index().sort_values("srs_asof", ascending=False))
        print(df.round(2).to_string(index=False))
        return

    long = build_long_results(games, [args.season])
    if long.empty:
        sys.exit(f"no completed games found for season {args.season}")

    margins = game_margins(long, args.season, cap=args.cap)
    rating, iters = solve_srs(margins)
    margins_raw = game_margins(long, args.season)

    cap_note = f", margins capped at ±{args.cap:g}" if args.cap else ""
    print(f"\n=== SRS — {args.season} season{cap_note} ===")
    print(f"converged in {iters} iterations | league avg "
          f"{sum(rating.values()) / len(rating):+.4f} (should be ~0)\n")
    print(srs_table(margins_raw, rating).to_string())

    if args.validate:
        validate(games, args.season, rating, margins_raw)


if __name__ == "__main__":
    main()
