#!/usr/bin/env python3
"""
Streamlit scoreboard built on nfl_box_score_analysis.py.

One page: pick a season/week and scroll through every game that week, each
with its final score and the box-score-style team stats (plays/yards/yards-
per-play, turnover margin, red zone trips/TD%/score%) listed right underneath
-- same stat logic as the CLI's efficiency-trend / turnover-trend /
redzone-trend commands, just computed per game instead of aggregated across
a season.

Usage:
    streamlit run streamlit_scoreboard.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from nfl_box_score_analysis import (
    build_long_results,
    first_downs_by_team_game,
    load_games,
    load_pbp,
    plays_yards_by_team_game,
    qb_hits_sacks_by_team_game,
    redzone_by_team_game,
    team_rushing_by_team_game,
    turnover_margin_by_team_game,
)
from predict_week import (
    build_week_table,
    detail_for_games,
    line_string,
    to_pbp_name,
    HFA_POINTS,
    MARGIN_STD,
)
from srs import blended_asof_ratings
from team_colors import BADGE_CSS, TEAM_ORDER, badge_html, nav_button_css
from team_composite import (
    add_game_performance_score,
    composite_srs,
    team_game_stats,
    team_season_table,
)

LOGOS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/logos.csv"
TEAMS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teams.csv"

st.set_page_config(page_title="NFL Scoreboard", page_icon="🏈", layout="wide")
st.markdown(
    '<style>section[data-testid="stSidebar"] { width: 220px !important; }</style>',
    unsafe_allow_html=True,
)
st.markdown(BADGE_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_games():
    return load_games()


@st.cache_data(show_spinner=False)
def cached_team_names() -> dict[str, str]:
    teams = pd.read_csv(TEAMS_URL)
    latest = teams.sort_values("season").drop_duplicates("team", keep="last")
    return dict(zip(latest["team"], latest["nickname"]))


@st.cache_data(show_spinner="Loading play-by-play for this season...")
def season_game_stats(season: int) -> pd.DataFrame:
    """One row per team-game: efficiency, turnover margin, and red zone stats."""
    games = cached_games()
    pbp = load_pbp(season)
    long = build_long_results(games, [season])

    eff = plays_yards_by_team_game(pbp)
    rz = redzone_by_team_game(pbp)
    to = turnover_margin_by_team_game(pbp, long)
    rush = team_rushing_by_team_game(pbp)
    fd = first_downs_by_team_game(pbp)
    pressure = qb_hits_sacks_by_team_game(pbp)

    df = to[["game_id", "week", "team", "opp", "team_score", "opp_score", "result",
             "own_giveaways", "opp_giveaways", "turnover_margin"]]
    df = df.merge(eff, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left").drop(columns=["posteam"])
    df = df.merge(
        rz[["game_id", "posteam", "rz_trips", "rz_td", "rz_scored"]],
        left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left",
    ).drop(columns=["posteam"])
    df = df.merge(
        rush[["game_id", "posteam", "rush_attempts", "rush_yards", "yds_per_attempt"]],
        left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left",
    ).drop(columns=["posteam"])
    df = df.merge(fd, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left").drop(columns=["posteam"])
    df["first_downs"] = df["first_downs"].fillna(0)
    df = df.merge(
        pressure, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left",
    ).drop(columns=["posteam"])
    return df


@st.cache_data(ttl=3600, show_spinner="Computing SRS ratings...")
def cached_srs_ratings(season: int):
    """No-leakage blended SRS per week for one season. Scores-only (no pbp),
    so this is fast; ttl lets newly played games flow into the ratings."""
    return blended_asof_ratings(cached_games(), [season])


# QB EPA blend constants, same placeholder status as SRS's k/shrink:
# QB_K_PSEUDO_DB = prior season's strength in pseudo-dropbacks (~4-5 games
# worth), QB_SHRINK = year-over-year regression TOWARD THE LEAGUE MEAN
# (new team/scheme/aging). Mean-regression matters: catch% shrunk toward
# 0 turns a 65% QB into 45%, which is nonsense -- toward the mean instead.
QB_K_PSEUDO_DB = 150.0
QB_SHRINK = 0.7


@st.cache_data(show_spinner="Loading QB stats for this season...")
def cached_qb_epa_asof(season: int) -> pd.DataFrame:
    """Per (week, QB): EPA/dropback through PRIOR weeks of the season,
    cold-start blended with the prior season -- same discipline as the SRS
    blend:  epa_db = db/(db+K) * current + K/(db+K) * shrink * prior.
    Week 1 is mostly last season (regressed); the current season takes
    over as dropbacks accumulate. QBs with no prior season (rookies) use
    current-only. CONTEXT DISPLAY ONLY: not part of the prediction model
    (srs-v1 stays pure SRS), so logs/grading are unaffected."""
    cur = None
    last_team = {}
    try:
        pbp = load_pbp(season)
        db = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_name"].notna()]
        last_team = (db.sort_values("week").groupby("passer_player_name")["posteam"]
                     .last().to_dict())  # for chart display; latest team if traded
        per = db.groupby(["passer_player_name", "week"]).agg(
            epa=("epa", "sum"), n=("epa", "count"),
            comp=("complete_pass", "sum"), att=("pass_attempt", "sum")).reset_index()
        frames = []
        for qb, grp in per.groupby("passer_player_name"):
            grp = grp.set_index("week").reindex(range(1, 23), fill_value=0)
            for col in ("epa", "n", "comp", "att"):
                grp[f"{col}_cum"] = grp[col].cumsum().shift(1).fillna(0)  # through prior weeks
            grp["passer_player_name"] = qb
            grp["week"] = grp.index
            frames.append(grp.reset_index(drop=True))
        cur = pd.concat(frames, ignore_index=True)
        del pbp
    except Exception as e:
        # This is the current season's own pbp -- a failure here means the
        # chart will look "empty" for a real reason (bad fetch, nflverse
        # hiccup, season not published yet), not because no QB has played.
        # Surface it instead of silently returning zero rows, which used to
        # be indistinguishable from a legitimate "no QB chart data yet".
        st.warning(f"QB stats: couldn't load {season} play-by-play ({e}). "
                   "QB chart / QB labels will be empty until this succeeds.")
        cur = pd.DataFrame(columns=["week", "passer_player_name", "team", "epa_cum",
                                    "n_cum", "comp_cum", "att_cum"])

    prior_epa, prior_catch = {}, {}
    try:
        pbp_prev = load_pbp(season - 1)
        db_prev = pbp_prev[(pbp_prev["qb_dropback"] == 1)
                           & pbp_prev["passer_player_name"].notna()]
        tot = db_prev.groupby("passer_player_name").agg(
            epa_sum=("epa", "sum"), n=("epa", "count"),
            comp=("complete_pass", "sum"), att=("pass_attempt", "sum"))
        lg_epa = tot["epa_sum"].sum() / tot["n"].sum()
        lg_catch = tot["comp"].sum() / tot["att"].sum()
        # Regress priors toward the LEAGUE MEAN, not toward 0 -- but QB_SHRINK
        # alone assumes the prior-season AVERAGE is itself reliable, which
        # fails for thin priors (a backup's 30-40 mop-up dropbacks is a wildly
        # noisy EPA/db). Scale the shrink by the prior sample's own size
        # (same K as the current-season cold-start weight below) so a QB with
        # only ~40 prior dropbacks gets pulled close to league-average before
        # QB_SHRINK is even applied, while a full prior season (500+
        # dropbacks) is barely touched by this extra factor. Without this, a
        # small unsustainable prior (e.g. Malik Willis's +0.74 EPA/db over 38
        # garbage-time snaps in 2025) dominated the blend for months into the
        # next season even after his real current-season play (-0.08) said
        # otherwise.
        prior_reliability = tot["n"] / (tot["n"] + QB_K_PSEUDO_DB)
        prior_epa = {qb: lg_epa + QB_SHRINK * prior_reliability[qb] * (v - lg_epa)
                     for qb, v in (tot["epa_sum"] / tot["n"]).items()}
        prior_catch = {qb: lg_catch + QB_SHRINK * prior_reliability[qb] * (v - lg_catch)
                       for qb, v in (tot["comp"] / tot["att"]).items()}
        del pbp_prev
    except Exception:
        pass  # no prior season available -> current-only for everyone

    cur["cur_epa_db"] = cur["epa_cum"] / cur["n_cum"].where(cur["n_cum"] > 0)
    cur["cur_catch"] = cur["comp_cum"] / cur["att_cum"].where(cur["att_cum"] > 0)
    cur["prior_epa_db"] = cur["passer_player_name"].map(prior_epa)
    cur["prior_catch"] = cur["passer_player_name"].map(prior_catch)
    cur["team"] = cur["passer_player_name"].map(last_team)
    w = cur["n_cum"] / (cur["n_cum"] + QB_K_PSEUDO_DB)
    # fill cur with 0 before multiplying: w is 0 exactly when cur is NaN,
    # but 0 * NaN = NaN would poison the blend (week 1 = pure prior)
    blended_epa = w * cur["cur_epa_db"].fillna(0) + (1 - w) * cur["prior_epa_db"]
    blended_catch = w * cur["cur_catch"].fillna(0) + (1 - w) * cur["prior_catch"]
    cur["epa_db"] = blended_epa.fillna(cur["cur_epa_db"])    # rookies: current-only
    cur["catch_pct"] = blended_catch.fillna(cur["cur_catch"])

    # QBs with a prior season but no dropbacks yet THIS season (injured
    # starters, mid-season takeovers): still show the regressed prior.
    have_cur = set(cur["passer_player_name"])
    prior_only = [qb for qb in prior_epa if qb not in have_cur]
    if prior_only:
        extra = pd.DataFrame({
            "week": [w_ for _ in prior_only for w_ in range(1, 23)],
            "passer_player_name": [qb for qb in prior_only for _ in range(1, 23)],
            "epa_db": [prior_epa[qb] for qb in prior_only for _ in range(1, 23)],
            "catch_pct": [prior_catch[qb] for qb in prior_only for _ in range(1, 23)],
            "n_cum": 0,
        })
        cur = pd.concat([cur, extra], ignore_index=True)
    return cur[["week", "passer_player_name", "team", "epa_db", "catch_pct", "n_cum"]]


def qb_lookup(qb_df, week, full_name):
    """(abbreviated name, epa_db or None, catch_pct or None) for one game's
    listed starter. None epa_db means genuinely no data (rookie with no
    dropbacks yet)."""
    abbr = to_pbp_name(full_name)
    if abbr is None:
        return None, None, None
    row = qb_df[(qb_df["week"] == week) & (qb_df["passer_player_name"] == abbr)]
    if row.empty or pd.isna(row["epa_db"].iloc[0]):
        return abbr, None, None
    catch = row["catch_pct"].iloc[0] if "catch_pct" in row.columns else None
    return abbr, row["epa_db"].iloc[0], (None if pd.isna(catch) else catch)


def qb_label(qb_df, week, full_name):
    """'J.Goff (+0.05, 68%)' style label; just the name if no trailing data."""
    abbr, epa_db, catch_pct = qb_lookup(qb_df, week, full_name)
    if abbr is None:
        return None
    if epa_db is None:
        return abbr
    inner = f"{epa_db:+.2f}"
    if catch_pct is not None:
        inner += f", {100 * catch_pct:.0f}%"
    return f"{abbr} ({inner})"


@st.cache_data(show_spinner="Computing Game Performance Score...")
def cached_composite_score(season: int) -> dict[str, float]:
    """team -> Game Performance Score (0-100), the QB chart's dot color."""
    tg = team_game_stats([season])
    t = team_season_table(tg)
    t["srs"] = t["team"].map(composite_srs(cached_games(), [season]))
    t = add_game_performance_score(t)
    return dict(zip(t["team"], t["game_performance_score"]))


# QB chart filter: minimum dropbacks THIS SEASON to appear on the scatter.
# Removes punters on fake punts, trick-play throwers, and one-play backups
# (a C.Johnston at -1.85 EPA/0% wrecked the whole x-axis otherwise).
# Same convention as build_qb_csv.py's min_dropbacks default.
MIN_CHART_DROPBACKS = 20


def latest_completed_week(games: pd.DataFrame, season: int) -> int:
    """Most recent week in this season with at least one final score, or 0
    if the season hasn't kicked off yet. This is what the QB chart's
    "Current" option resolves to."""
    sub = games[games["season"] == season]
    played = sub.dropna(subset=["home_score", "away_score"])
    return int(played["week"].max()) if len(played) else 0


def qb_chart_frame(season: int, through_week: int) -> pd.DataFrame:
    """One row per QB with enough volume THIS season (n_cum >=
    MIN_CHART_DROPBACKS), stats THROUGH `through_week` inclusive (i.e.
    counting that week's games), team record, and the team's Game
    Performance Score.

    cached_qb_epa_asof stores each week's cumulative total as of *before*
    that week (shifted by one, for no-leakage predictions), so "through
    week W inclusive" is that frame's row at week=W+1 -- translate here so
    every caller of this function can think in plain "through week W"
    terms instead of tripping over the shift."""
    qb = cached_qb_epa_asof(season)
    lookup_week = min(through_week + 1, 22)
    df = qb[(qb["week"] == lookup_week) & (qb["n_cum"] >= MIN_CHART_DROPBACKS)].dropna(
        subset=["epa_db", "catch_pct", "team"]).copy()
    if df.empty:
        return df
    long = build_long_results(cached_games(), [season])
    rec = long.groupby("team")["result"].apply(
        lambda s: f"{(s == 'W').sum()}-{(s == 'L').sum()}").to_dict()
    scores = cached_composite_score(season)
    df["record"] = df["team"].map(rec)
    df["score"] = df["team"].map(scores)
    return df.dropna(subset=["score"])


def repel_labels(xs, ys, iters=250):
    """Force-directed label placement in pure numpy (the adjustText idea,
    no matplotlib/adjustText dependency): labels start above their point,
    then iterate -- label-vs-label and label-vs-other-points repulsion,
    a weak pull home to the label's own point, clipped steps. Returns
    (label_x, label_y) in DATA coordinates, so Altair can draw both the
    labels and connector lines back to their dots."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    xr = (x1 - x0) or 1.0
    yr = (y1 - y0) or 1.0
    xn = (x - x0) / xr
    yn = (y - y0) / yr
    lx, ly = xn.copy(), np.clip(yn + 0.05, 0.0, 1.0)

    for _ in range(iters):
        # weak attraction home keeps every label tethered to its dot
        fx = (xn - lx) * 0.08
        fy = (yn - ly) * 0.08
        # label-vs-label repulsion (only when too close)
        dx = lx[:, None] - lx[None, :]
        dy = ly[:, None] - ly[None, :]
        d2 = dx * dx + dy * dy + 1e-9
        m = d2 < 0.004
        np.fill_diagonal(m, False)
        push = np.where(m, 0.004 / d2, 0.0)
        fx += (dx * push).sum(axis=1)
        fy += (dy * push).sum(axis=1)
        # label-vs-other-points repulsion (weaker)
        pdx = lx[:, None] - xn[None, :]
        pdy = ly[:, None] - yn[None, :]
        pd2 = pdx * pdx + pdy * pdy + 1e-9
        pm = pd2 < 0.002
        np.fill_diagonal(pm, False)
        ppush = np.where(pm, 0.002 / pd2, 0.0)
        fx += (pdx * ppush).sum(axis=1) * 0.5
        fy += (pdy * ppush).sum(axis=1) * 0.5
        lx = np.clip(lx + np.clip(fx * 0.03, -0.03, 0.03), 0.0, 1.0)
        ly = np.clip(ly + np.clip(fy * 0.03, -0.03, 0.03), 0.0, 1.0)

    return lx * xr + x0, ly * yr + y0


def render_qb_chart(season: int, through_week: int):
    """The QB EPA/catch% scatter (qb_support_chart.py's look), colored by
    Game Performance Score, rendered in-app with Altair (no matplotlib
    dependency on Streamlit Cloud). `through_week` is inclusive -- games
    from that week are counted."""
    df = qb_chart_frame(season, through_week)
    label = f"through week {through_week}" if through_week > 0 else "before week 1 (preseason)"
    st.markdown(f"**QB EPA/dropback & catch% — {label}, {season}** "
                "(color = Game Performance Score)")
    if df.empty:
        st.caption(f"No QB has {MIN_CHART_DROPBACKS}+ dropbacks {label} yet. "
                   "If games from this week are already final, try picking "
                   "an earlier week, or \"Current\" once this week's play-by-"
                   "play has been published by nflverse.")
        return
    base = alt.Chart(df).encode(
        x=alt.X("epa_db:Q", title="EPA / Dropback"),
        y=alt.Y("catch_pct:Q", title="CATCH %", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domainMin=0.5)),  # floor at 50%: sub-50% was
    )                                             # all outlier noise anyway
    lx, ly = repel_labels(df["epa_db"], df["catch_pct"])
    labels = df.assign(lx=lx, ly=ly)
    pts = base.mark_circle(size=130, stroke="black", strokeWidth=0.5).encode(
        color=alt.Color("score:Q", title="Game Perf",
                        scale=alt.Scale(scheme="redyellowgreen", domain=[20, 90])),
        tooltip=["passer_player_name", "team", "record",
                 alt.Tooltip("epa_db:Q", format=".3f"),
                 alt.Tooltip("catch_pct:Q", format=".1%"),
                 alt.Tooltip("score:Q", format=".0f")],
    )
    links = alt.Chart(labels).mark_rule(color="gray", strokeWidth=0.5, opacity=0.6).encode(
        x="epa_db:Q", y="catch_pct:Q", x2="lx:Q", y2="ly:Q",
    )
    txt = alt.Chart(labels).mark_text(fontSize=9).encode(
        x="lx:Q", y="ly:Q", text="passer_player_name:N",
    )
    st.altair_chart(pts + links + txt, use_container_width=True)
    st.caption(f"Hover any dot for team, record, and exact numbers. QBs with "
               f"<{MIN_CHART_DROPBACKS} dropbacks this season are hidden "
               f"(punters, trick plays, mop-up). Same construction as the "
               f"standalone QB chart, but as-of \"through week {through_week}\" "
               f"above, with the prior-season blend.")


def current_season_and_week(games: pd.DataFrame) -> tuple[int, int]:
    """Latest season, and the most recent week that has kicked off (so the
    default view is 'this week', including in-progress/upcoming games in it)."""
    season = int(games["season"].max())
    sub = games[games["season"] == season].copy()
    sub["gameday"] = pd.to_datetime(sub["gameday"])
    started = sub[sub["gameday"] <= pd.Timestamp.now()]
    week = int(started["week"].max()) if len(started) else 1
    return season, week


def day_label(gameday: pd.Timestamp) -> str:
    today = pd.Timestamp.now().normalize()
    delta = (gameday.normalize() - today).days
    if delta == 0:
        return "Today"
    if delta == -1:
        return "Yesterday"
    if delta == 1:
        return "Tomorrow"
    return gameday.strftime("%a, %b %-d")


def stats_table(game_stats: pd.DataFrame, game_id: str, away: str, home: str, away_label: str,
                home_label: str, srs_by_team: dict | None = None) -> pd.DataFrame:
    """srs_by_team: {team -> rating}, already as-of BEFORE this game's week
    (no leakage) -- same rating the Predictions tab would have shown ahead
    of kickoff, so it reads as "how good the numbers said they were", not
    a rating that's seen this game's own result."""
    g = game_stats[game_stats["game_id"] == game_id].set_index("team")
    srs_by_team = srs_by_team or {}
    rows = {}
    for team, label in ((away, away_label), (home, home_label)):
        srs = srs_by_team.get(team)
        srs_str = f"{srs:+.1f}" if srs is not None else "-"
        if team not in g.index:
            rows[label] = {"SRS": srs_str, "1st": "-", "Plays": "-", "Yards": "-", "Yds/Play": "-",
                            "TO's": "-", "Rush #": "-", "Yds": "-", "Yds/Att": "-",
                            "RZ TD": "-", "RZ Score": "-", "QB Hits": "-", "Sacks": "-"}
            continue
        r = g.loc[team]
        rz_trips = int(r["rz_trips"]) if pd.notna(r["rz_trips"]) else 0
        rz_td = int(r["rz_td"]) if pd.notna(r["rz_td"]) else 0
        rz_scored = int(r["rz_scored"]) if pd.notna(r["rz_scored"]) else 0
        rows[label] = {
            "SRS": srs_str,
            "1st": int(r["first_downs"]) if pd.notna(r["first_downs"]) else "-",
            "Plays": int(r["plays"]) if pd.notna(r["plays"]) else "-",
            "Yards": int(r["yards"]) if pd.notna(r["yards"]) else "-",
            "Yds/Play": f"{r['yards_per_play']:.1f}" if pd.notna(r["yards_per_play"]) else "-",
            "TO's": f"{r['turnover_margin']:+.0f}",
            "Rush #": int(r["rush_attempts"]) if pd.notna(r["rush_attempts"]) else "-",
            "Yds": int(r["rush_yards"]) if pd.notna(r["rush_yards"]) else "-",
            "Yds/Att": f"{r['yds_per_attempt']:.1f}" if pd.notna(r["yds_per_attempt"]) else "-",
            "RZ TD": f"{rz_td}/{rz_trips}" if rz_trips else "-",
            "RZ Score": f"{rz_scored}/{rz_trips}" if rz_trips else "-",
            "QB Hits": int(r["qb_hits"]) if pd.notna(r["qb_hits"]) else "-",
            "Sacks": int(r["sacks"]) if pd.notna(r["sacks"]) else "-",
        }
    return pd.DataFrame(rows).T


def _jump_to_team(abbr):
    """Nav grid click: switch to Team view pre-filtered to this team.
    Runs pre-script (on_click), so widget-state writes are legal."""
    st.session_state["selected_team"] = abbr
    st.session_state["view_by"] = "Team"
    st.session_state["team_filter"] = abbr


def main():
    st.title("🏈 NFL Scoreboard")
    st.caption("Score, then the box-score stats right underneath: plays/yards/yards-per-play, turnover margin, red zone conversion.")

    games = cached_games()
    names = cached_team_names()
    default_season, default_week = current_season_and_week(games)

    with st.sidebar:
        st.header("Setup")
        seasons_available = sorted(games["season"].unique(), reverse=True)
        season = st.selectbox("Season", seasons_available, index=seasons_available.index(default_season))

        view_by = st.segmented_control("View by", ["Week", "Team", "Predictions"],
                                       default="Week", key="view_by")

        season_games = games[games["season"] == season]
        # Predictions needs only final scores (SRS) -- skip the heavy pbp load
        game_stats = None if view_by == "Predictions" else season_game_stats(season)
        week = None
        team_filter = None
        qb_through_week = None
        if view_by == "Team":
            teams_available = sorted(
                set(season_games["home_team"]).union(season_games["away_team"]),
                key=lambda t: names.get(t, t),
            )
            default_team = "LAC" if "LAC" in teams_available else teams_available[0]
            team_filter = st.selectbox(
                "Team", teams_available,
                index=teams_available.index(default_team),
                format_func=lambda t: names.get(t, t),
                key="team_filter",
            )
        else:
            weeks_available = sorted(season_games["week"].unique())
            week = st.selectbox(
                "Week", weeks_available,
                index=weeks_available.index(default_week) if default_week in weeks_available else 0,
            )
            if st.button("📈 QB chart", use_container_width=True,
                         help="EPA/dropback vs catch% scatter, colored by "
                              "Game Performance Score"):
                st.session_state["show_qb_chart"] = not st.session_state.get("show_qb_chart", False)

            if st.session_state.get("show_qb_chart"):
                # Deliberately its own control, decoupled from the "Week"
                # selectbox above: that one drives which games are LISTED,
                # this one drives which games are COUNTED into the QB
                # stats. Conflating the two was the source of the chart
                # looking "broken" -- picking Week 3 games showed QB stats
                # that didn't yet include week 3 (the no-leakage shift used
                # for predictions leaking into a view that has no leakage
                # concern). "Current" always resolves to the latest week
                # with final scores, so it's the one that self-updates as
                # games finish -- pick this after a game to see it reflected.
                qb_week_options = ["Current"] + [str(w) for w in weeks_available]
                qb_week_choice = st.selectbox(
                    "QB chart: through week", qb_week_options, index=0,
                    key="qb_chart_week_choice",
                    help="Which games count toward the QB stats plotted. "
                         "\"Current\" = through the most recent final score.",
                )
                qb_through_week = (latest_completed_week(games, season)
                                   if qb_week_choice == "Current" else int(qb_week_choice))

        show_preds = False
        if view_by != "Predictions":
            show_preds = st.toggle(
                "Attach prediction row to games", value=False,
                help="Adds each game's SRS ratings, model line, market line, and "
                     "predicted winner under the score, plus each starting QB's "
                     "EPA/dropback next to the team name.",
            )

        st.divider()
        st.caption(
            "Note on QB Hits/Sacks: hits usually include sacks, but not with "
            "100% consistency in the raw nflverse charting -- the two are kept "
            "as separate reported columns rather than assuming one strictly "
            "contains the other."
        )

    if view_by == "Predictions":
        ratings = cached_srs_ratings(season)
        pred_df, detail = build_week_table(games, season, week, ratings, k=4.0, shrink=0.7,
                                           names=names)
        qb_df = cached_qb_epa_asof(season)
        pred_df["away_qb"] = [qb_label(qb_df, d.week, d.away_qb) or "-"
                              for d in detail.itertuples()]
        pred_df["home_qb"] = [qb_label(qb_df, d.week, d.home_qb) or "-"
                              for d in detail.itertuples()]

        st.caption(f"Season {season}, Week {week} — {len(pred_df)} games · "
                   f"SRS ratings as of before week {week} (no games from this week or later)")
        st.sidebar.download_button(
            "⬇️ Download this view (CSV)",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name=f"nfl_{season}_week{week:02d}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(pred_df, hide_index=True, use_container_width=True)
        if st.session_state.get("show_qb_chart"):
            # Match the model's own no-leakage info set: predictions for
            # week `week` only ever see games through week-1.
            render_qb_chart(season, max(week - 1, 0))
        st.caption(
            f"away_srs/home_srs: opponent-adjusted team rating in points "
            f"(0 = league average; HFA {HFA_POINTS:+.1f}, win-prob std {MARGIN_STD:.1f}). "
            f"model_line: SRS diff + home field. edge: model line minus market line, "
            f"home-positive (+ = model likes home more than the market)."
        )
        st.caption(
            "Reminder: in honest walk-forward backtests (2021-2025, spread and "
            "moneyline) no edge bucket cleared the 52.4% breakeven — this sheet "
            "says how good the teams are, not what to bet."
        )
        with st.expander("How are these predictions made?"):
            st.markdown(
                "**away_srs / home_srs** — each team's Simple Rating System number, "
                "in points: average scoring margin plus the average rating of the "
                "opponents faced (opponent-adjusted; 0 = league average). Uses only "
                "games *before* this week, blended with last season early in the year.\n\n"
                "**model_line** — home_srs − away_srs + 2.2 (home-field constant "
                "estimated from the 2021-2025 backtest): the margin the model "
                "considers fair, in betting notation.\n\n"
                "**market_line** — the sportsbook's actual spread.\n\n"
                "**edge** — model_line − market_line (home-positive): + means the "
                "model likes the home team *more than the market does*.\n\n"
                "**pred / win_pct** — the team the model thinks **wins the game "
                "outright** (bigger predicted margin), and its calibrated "
                "probability (normal curve, 13.4-point error std from the backtest). "
                "This is about *winning*, never about covering the spread.\n\n"
                "**away_qb / home_qb** — each team's listed starter and his "
                "EPA/dropback: trailing through prior weeks this season, "
                "blended with last season early in the year (rookies: this "
                "season only). Context only — NOT part of the model.\n\n"
                "**final / pred_right** — the actual score, and whether the "
                "predicted winner actually won."
            )
        return

    if view_by == "Team":
        shown_games = season_games[
            (season_games["home_team"] == team_filter) | (season_games["away_team"] == team_filter)
        ].copy()
        st.caption(f"Season {season}, {names.get(team_filter, team_filter)} — {len(shown_games)} games")
    else:
        shown_games = season_games[season_games["week"] == week].copy()
        st.caption(f"Season {season}, Week {week} — {len(shown_games)} games")
        if st.session_state.get("show_qb_chart") and qb_through_week is not None:
            render_qb_chart(season, qb_through_week)

    # Team jump grid: one colored button per team, no scrolling needed
    st.markdown("---")
    nav_cols = st.columns(8)
    for i, abbr in enumerate(TEAM_ORDER):
        nav_cols[i % 8].button(abbr, key=f"btn_{abbr}", on_click=_jump_to_team, args=(abbr,))
    st.markdown(nav_button_css(), unsafe_allow_html=True)
    sel = st.session_state.get("selected_team")
    st.caption(f"Selected team: **{sel}** (showing its games below)"
               if sel else "Tap a team to jump straight to its games.")
    st.markdown("---")

    shown_games["gameday"] = pd.to_datetime(shown_games["gameday"])
    shown_games = shown_games.sort_values(["week", "gameday", "gametime"])

    # Always computed (not just under show_preds): the box-score table's SRS
    # column wants it regardless of whether the prediction row is attached.
    srs_ratings = cached_srs_ratings(season)

    pred_by_id = None
    qb_df = None
    if show_preds:
        pred_by_id = detail_for_games(shown_games, srs_ratings).set_index("game_id")
        qb_df = cached_qb_epa_asof(season)

    export_df = game_stats[game_stats["game_id"].isin(shown_games["game_id"])].copy()
    export_df.insert(0, "season", season)
    export_df.insert(3, "team_name", export_df["team"].map(names))
    export_df.insert(5, "opp_name", export_df["opp"].map(names))

    # SRS as-of that game's week (no leakage) -- same number now shown on
    # every box-score card, so the export always carries it too.
    export_df["srs"] = export_df.apply(
        lambda r: srs_ratings.get((season, r["week"]), {}).get(r["team"]), axis=1
    ).round(3)

    if show_preds:
        # Mirror the "Attach prediction row to games" toggle in the CSV --
        # previously this toggle changed only the on-screen cards and the
        # download stayed the box-score-only columns no matter what.
        gcols = pred_by_id[["home", "pred_margin", "spread_line", "edge", "home_win_prob",
                            "pred_winner", "away_qb", "home_qb"]].reset_index()
        export_df = export_df.merge(gcols, on="game_id", how="left")
        is_home = export_df["team"] == export_df["home"]
        # model_line / market_line: standard sportsbook notation, per-team --
        # favorite negative, underdog positive (what you'd actually see
        # typed into DraftKings for that team), same convention line_string()
        # already uses for the on-screen "model BUF -3.4" text.
        export_df["model_line"] = (-export_df["pred_margin"]).where(is_home, export_df["pred_margin"]).round(2)
        export_df["market_line"] = (-export_df["spread_line"]).where(is_home, export_df["spread_line"])
        # edge is NOT a spread, so it intentionally does NOT follow the same
        # flip -- it's a value signal (positive = model likes THIS ROW'S
        # team more than the market does), so edge != model_line -
        # market_line by design. Flipping it the same way as the display
        # lines would invert its meaning on every favorite's row (a
        # favorite's positive-market/negative-model gap would read as
        # "value here" when the value is actually on the underdog).
        export_df["edge"] = export_df["edge"].where(is_home, -export_df["edge"]).round(2)
        export_df["win_prob"] = export_df["home_win_prob"].where(is_home, 1 - export_df["home_win_prob"]).round(3)
        export_df["model_favors_team"] = export_df["team"] == export_df["pred_winner"]
        export_df["qb_name"] = export_df["away_qb"].where(~is_home, export_df["home_qb"])
        qb_stats = export_df.apply(
            lambda r: pd.Series(qb_lookup(qb_df, r["week"], r["qb_name"]),
                                index=["qb", "qb_epa_db", "qb_catch_pct"]),
            axis=1,
        )
        export_df = pd.concat([export_df, qb_stats], axis=1)
        export_df = export_df.drop(columns=["home", "pred_margin", "spread_line",
                                            "away_qb", "home_qb", "qb_name"])

    export_scope = names.get(team_filter, team_filter) if view_by == "Team" else f"week{week}"
    st.sidebar.download_button(
        "⬇️ Download this view (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"nfl_{season}_{export_scope}_team_game_stats.csv",
        mime="text/csv",
        help="Every stat shown on the scoreboard cards, for exactly the games "
             "currently on screen -- handy for a groupby/mean in pandas or "
             "handing to Claude Code. Includes model/market/QB columns when "
             "\"Attach prediction row to games\" is on.",
        use_container_width=True,
    )

    for game in shown_games.itertuples():
        final = pd.notna(game.away_score) and pd.notna(game.home_score)
        away_win = final and game.away_score > game.home_score
        home_win = final and game.home_score > game.away_score
        away_label = names.get(game.away_team, game.away_team)
        home_label = names.get(game.home_team, game.home_team)

        with st.container(border=True):
            for team, label, score, won in (
                (game.away_team, away_label, game.away_score, away_win),
                (game.home_team, home_label, game.home_score, home_win),
            ):
                badge_col, name_col, score_col = st.columns([1, 6, 1])
                badge_col.markdown(badge_html(team), unsafe_allow_html=True)
                qb_str = ""
                if qb_df is not None:
                    full_qb = game.away_qb_name if team == game.away_team else game.home_qb_name
                    ql = qb_label(qb_df, game.week, full_qb)
                    if ql:
                        qb_str = f" · {ql}"
                name_col.markdown(f"{'**' if won else ''}{label}{'**' if won else ''}{qb_str}")
                score_text = f"{int(score)}" if pd.notna(score) else "-"
                score_col.markdown(f"{'**' if won else ''}{score_text}{'**' if won else ''}")

            status = "Final" if final else "Upcoming"
            time_suffix = f" {game.gametime}" if not final and pd.notna(game.gametime) else ""
            week_prefix = f"Week {game.week} · " if view_by == "Team" else ""
            st.caption(f"{week_prefix}{status} · {day_label(game.gameday)}{time_suffix}")

            if pred_by_id is not None and game.game_id in pred_by_id.index:
                p = pred_by_id.loc[game.game_id]
                model_line = line_string(game.home_team, game.away_team, p["pred_margin"])
                market_line = (line_string(game.home_team, game.away_team, p["spread_line"])
                               if pd.notna(p["spread_line"]) else "-")
                # edge in the same "TEAM -X.X" notation as model/market: which
                # team the model likes MORE than the market credits them, and
                # by how much (edge = model line minus market line).
                edge_line = (line_string(game.home_team, game.away_team, p["edge"])
                            if pd.notna(p["edge"]) else "-")
                win_pct = 100 * max(p["home_win_prob"], 1 - p["home_win_prob"])
                right = ""
                if final:
                    correct = (game.home_score > game.away_score) == (p["pred_winner"] == game.home_team)
                    right = f" · **{'✓ right' if correct else '✗ wrong'}**"
                st.markdown(
                    f"🔮 SRS {game.away_team} **{p['away_srs']:+.1f}** @ "
                    f"{game.home_team} **{p['home_srs']:+.1f}** · "
                    f"model **{model_line}** · market **{market_line}** · "
                    f"edge **{edge_line}** · "
                    f"pred **{p['pred_winner']}** {win_pct:.0f}%{right}"
                )

            if final:
                week_ratings = srs_ratings.get((season, game.week), {})
                st.table(
                    stats_table(game_stats, game.game_id, game.away_team, game.home_team,
                               away_label, home_label, week_ratings)
                )


if __name__ == "__main__":
    main()
