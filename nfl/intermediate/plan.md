# Plan: SRS rating → win probability → spread edge → backtest

No code in this file — steps, file layout, and exactly what you'd expect
to see if you ran each step, so it can be checked against reality once
built. See `intermediate/qa_log.md` for how we got here (the backtests
that justify each design choice below).

Context for why this plan looks the way it does: the DET/BUF box score
you pasted is a **post-game** recap — 25 lines, and you eyeballed
pressure (BUF 2 sacks/5yds vs DET 4 sacks/38yds) as the tiebreaker. The
whole point of this pipeline is that a **pre-game** view can't ever show
that box score (it hasn't happened yet) — it has to compress "how good is
each team" into a few trailing numbers *before* kickoff, one of which
(pressure rate) is already exactly the stat you noticed by hand.

---

## Phase 1 — SRS rating (season-total first, so it's checkable)

**Status: BUILT (2026-09-17)** — `srs.py`, validated against PFR's
published numbers (corr 0.9994, MAE 0.20 pts on regular-season scope;
PFR excludes playoffs, ours includes them). Details in
`STATS_REFERENCE.md` and `intermediate/qa_log.md`.

**Goal:** one number per team, in points, opponent-adjusted.

**Steps:**
1. New file `srs.py`. Note: SRS only needs final scores, not play-by-play
   — it can be built directly from `build_long_results()` in
   `nfl_box_score_analysis.py` (already gives `team`, `opp`, `team_score`,
   `opp_score` per game from `games.csv`). No pbp download needed for
   this piece, unlike everything else so far.
2. Implement the iterative solve: start every team at 0, repeat
   `rating[team] = avg_margin[team] + avg(rating[opponents faced])` until
   the numbers stop moving.
3. Optional refinement: cap each game's margin (e.g. ±24) before
   averaging, so one blowout doesn't distort a rating — standard SRS
   practice.
4. **Validate the same way `team_composite.py` validated its score**:
   correlation between SRS and actual point differential (should be
   high, since SRS is built from point differential — the real check is
   whether it beats *uncapped raw average margin* by correcting for
   schedule strength) and a sanity spot-check against Pro-Football-
   Reference's published SRS numbers for a known season, since PFR
   publishes this exact stat and it's free to eyeball.

**What you'd see if you ran this:** a 32-row table, one season at a time —
```
team   srs
KC     6.2
DEN   -2.1
...
```
League average should land at ~0.

---

## Phase 2 — Make SRS "as of a date" (no leakage)

**Status: BUILT (2026-09-17)** — `solve_srs_asof()` +
`blended_asof_ratings()` in `srs.py` (cold-start blend k=4, shrink=0.7,
placeholders pending sweep; damped solve fixes early-season oscillation).
Honest backtest run: **win acc 63.2%, but ATS inverted — 3+ bucket
45.6%, |edge|≥4 rule 46.5%, all below breakeven.** Raw SRS alone is not
a betting signal. Full read in `intermediate/qa_log.md`.

**Goal:** SRS computed only from games strictly before the game being
predicted — same discipline as everything else in this project.

**Steps:**
1. Extend `srs.py` with a function that takes a season + week and solves
   SRS using only games with `week < N` in that season (or blended with
   the prior full season when `N` is small — same cold-start problem
   already found in the week-1 backtests: with 0-2 games played, SRS is
   noise, and last season's SRS did **worse than picking the home team**
   in that exact test, so blending needs the same season-weight ramp
   discussed for the composite score, not a hard cutover).
2. Output one row per team **per week**, not one static number — a
   team's SRS should move as the season progresses.

**What you'd see:** a table like
```
season  week  team   srs_asof
2026    1     KC     4.8     <- from 2025 only (blended), 2026 has 0 games yet
2026    5     KC     7.1     <- now mostly from 2026's actual games
```

---

## Phase 3 — Predicted margin, win probability, market edge (one game)

**Status: folded into Phase 4 (2026-09-17)** — the per-game math lives in
`predict_week.py::build_week_table` (SRS diff + HFA 2.2, normal-CDF win
prob with std 13.44, edge vs `spread_line`; constants from the honest
2021-2025 backtest).

**Goal:** turn two teams' SRS-as-of numbers into the concrete output
discussed earlier — this is the "one game" version before Phase 4 makes
it a whole week.

**Steps:**
1. `predicted_home_margin = SRS_home - SRS_away + home_field_constant`
   — the constant should be **estimated from the data**, not guessed:
   average actual home margin minus average SRS-predicted margin (with
   HFA excluded), across a large sample of past games.
2. `win_prob = normal_cdf(predicted_home_margin / std_of_margin_error)`
   — `std_of_margin_error` also estimated from data: how far off SRS's
   predicted margins actually are from real results, historically (this
   number falls out of Phase 5's backtest, so Phase 3 and Phase 5 are
   circular in a good way — you calibrate one from the other).
3. `edge = predicted_home_margin - spread_line` (same home-positive sign
   convention as nflverse's `spread_line`, confirmed already).

**What you'd see, for one game:**
```
DEN @ KC
  SRS: KC +6.2   DEN -2.1
  Predicted margin (KC home): +8.3
  Win probability: KC 74%
  Market spread: KC +6.5
  Edge: +1.8 toward KC covering
```

---

## Phase 4 — Weekly output (`predict_week.py`)

**Status: BUILT (2026-09-17)** — `predict_week.py` (CLI, defaults to the
current season/week) plus a **Predictions view in
`streamlit_scoreboard.py`** (third option in the "View by" control;
skips the pbp load — SRS needs only scores). Smoke-tested with
Streamlit's AppTest on both views. Deploys on the next git push to the
Streamlit Cloud repo.

**Goal:** the thing you actually pull when you ask "week 2."

**Steps:**
1. New file `predict_week.py --season 2026 --week 2`.
2. Pull that week's schedule from `load_games()` (works whether the week
   is already played — for backtesting — or upcoming — for real use;
   unplayed games just have null scores, everything else is identical).
3. Look up each team's SRS-as-of (from Phase 2, using the week *before*
   this one) and run Phase 3's math per game.
4. Print/save a table, one row per game: away, home, predicted winner,
   win_prob, market spread, edge, suggested side.

**What you'd see:** exactly the table format already discussed —
```
week  away  home  home_win_prob  spread_line  edge   pick
2     DAL   NYG   38%            -3.0         +1.2   NYG (lean)
2     KC    PHI   61%             2.5         -0.8   pass (edge too small)
```

---

## Phase 5 — Backtest (this is what tells you whether any of this works)

**Status: harness BUILT, both runs done (2026-09-17)** —
`backtest_srs.py`. Leaky ceiling: win acc 70.4%, rule ≥4 at 71.1%.
Honest as-of run: win acc 63.2% but ATS inverted, rule ≥4 at 46.5% —
SRS alone doesn't beat the market. Constants (k/shrink/cap) untuned;
Phase 7 blend is the planned path to a real signal. Numbers +
interpretation in `intermediate/qa_log.md`.

**Goal:** the same discipline used for the win-prediction and QB-EPA
backtests earlier — pooled across many games, not one week, walk-forward
so nothing sees its own future.

**Steps:**
1. **Walk-forward, not a single split:** for every week of every season
   from 2021 onward (nflverse's earliest year), compute SRS-as-of using
   only games strictly before that week, generate that week's
   predictions, then grade against the real result. Repeat for every
   week — this produces hundreds/thousands of graded predictions instead
   of the 16-80 games used in the earlier by-hand backtests.
2. **Metrics to report, same as `grade_predictions.py` in Phase 6:**
   - Win accuracy vs. the home-baseline (52.5% pooled, per the wider
     backtest already run — SRS needs to beat that, not just beat 50%).
   - **Brier score** for calibration (do "70% confidence" games actually
     win ~70% of the time).
   - **ATS cover rate, bucketed by edge size** (0-1pt, 1-2pt, 2-3pt,
     3+pt) — this is what actually answers "how big an edge is worth
     betting," instead of guessing a round number.
3. **The real betting breakeven rate is 52.4%, not 50%** — standard
   -110 odds on both sides means a bettor needs to win ~52.38% of bets
   just to break even after the vig. Any cover-rate bucket needs to clear
   *that* bar, not 50%, to represent a real edge worth acting on. This
   number should be a named constant in whatever script reports backtest
   results, so it's never silently compared against the wrong bar.
4. Compare against the two already-established baselines from the
   earlier backtests (home-team-always, QB-EPA-alone) so SRS's marginal
   value is explicit, not just "SRS looks fine in isolation."

**What you'd see:** a report like
```
Walk-forward backtest, 2021-2025, N=1,280 graded games
Win accuracy: 58.3%  (home baseline: 52.5%, QB-EPA-alone: 57.8%)
Brier score: 0.231
ATS cover rate by edge bucket:
  0-1 pt edge:   50.8%  (n=410)  -- below breakeven, don't bet these
  1-2 pt edge:   52.1%  (n=340)  -- still below 52.4% breakeven
  2-3 pt edge:   54.6%  (n=290)  -- above breakeven
  3+  pt edge:   57.2%  (n=240)  -- above breakeven, strongest signal
```
That kind of table is what tells you, honestly, whether "bet when edge >
2 points" is a real rule or wishful thinking.

---

## Phase 6 — Logging + grading for ongoing use

**Status: BUILT (2026-09-17)** — `predict_week.py --log` appends unplayed
games to `predictions_log.csv` (refuses already-played games and
duplicates; `MODEL_VERSION = "srs-v1"` in `predict_week.py` — bump on
any model change). `grade_predictions.py` grades the log against actual
results per version (win acc, Brier, ATS cover, |edge|≥4 line, home
baseline). First live entries: 15 games logged for 2026 week 2 before
kickoff. The log file stays local (uncommitted) — commit it manually if
you want it backed up.

(Already discussed in `qa_log.md` — restated here as the last phase since
it's what turns this from a one-off backtest into something you check
every week going forward.)

1. `predictions_log.csv` — append-only, one row per game, written at
   prediction time (before kickoff), tagged with a `model_version`.
2. `grade_predictions.py` — run after games finish, joins the log back to
   `load_games()` on `game_id`, computes accuracy/Brier/cover-rate the
   same way Phase 5 does, but for live predictions instead of a
   historical backtest.
3. Every time SRS's constants get re-estimated (HFA, margin-error std) or
   a QB/pressure blend gets added on top, bump `model_version` so
   `grade_predictions.py` can compare versions head to head on the same
   graded weeks.

---

## Betting rule (decided)

The rule the weekly output should implement, confirmed with the user:

- **Spread bets: take any game where `|edge|` >= threshold, in whichever
  direction the edge points.** Threshold placeholder = **4 points**; the
  real number comes from Phase 5's edge-bucket cover rates (smallest
  bucket that clears the 52.4% breakeven). Keep it a parameter, not a
  constant baked into logic.
- **No "predicted winner only" filter.** An earlier idea — never bet the
  predicted loser +points — was dropped in favor of pure edge size. This
  means some bets will be on teams the model predicts lose outright
  (they cover without winning); that's intended.
- **Moneyline bets are a separate test:** model win probability vs.
  moneyline-implied probability (from `home_moneyline`/`away_moneyline`),
  not the point edge. **Graded 2026-09-17 (`backtest_moneyline.py`): no
  rule clears breakeven** — model is well-calibrated but its
  disagreements with the price are model error, not market error. See
  `intermediate/qa_log.md`.
- Phase 4's `pick` column = the edge side when `|edge|` clears the
  threshold, else `pass`. Expect only ~1-3 bets/week at a 4pt bar; a
  week of zero bets is the rule working, not a bug.

## Suggested build order

1. Phase 1 (SRS, season-total) — cheapest to build and sanity-check
   against PFR's published numbers.
2. Phase 5's walk-forward harness, run on the season-total SRS from step
   1 *without* the as-of-date logic yet — gives a rough first read fast.
3. Phase 2 (as-of-date SRS) — the real, no-leakage version.
4. Re-run Phase 5 on the as-of-date version — this is the number that
   actually matters, and it should be compared against step 2's rough
   version to see how much the leakage fix changes things.
5. Phase 3 + 4 (single game, then weekly table).
6. Phase 6 (logging/grading), once you're actually using weekly output
   for real decisions, not just backtesting.

Not in this plan: blending SRS with the QB EPA/dropback and pressure-rate
trailing features from earlier. That's a natural Phase 7 once SRS alone
has a real backtested number to beat — same "don't guess the weights,
fit them" lesson from the earlier hand-picked blend that underperformed
QB-EPA alone.

**Phase 7 status: BUILT and graded (2026-09-17)** — `backtest_blend.py`,
expanding-window OLS, team-level EPA. Blend adds +0.5pt win accuracy
(64.1%) but ATS still below breakeven everywhere; neither model finds
spread value vs the market. Full numbers + untried levers in
`intermediate/qa_log.md`.

## Phase 7b — per-QB composite in the model

**Status: BUILT and graded (2026-09-18) — does not win.**
`backtest_blend.py --qb`: `qb_season_asof()` + `qb_starter_features()`
lift `cached_qb_epa_asof()`'s as-of, cold-start-blended EPA/db + catch%
math (from display-only in `streamlit_scoreboard.py`) into the model,
keyed by the listed starter (`home_qb_name`/`away_qb_name` via
`to_pbp_name()`) instead of by team. M2 = M1 + `qb_epa_diff` +
`qb_catch_diff`, same expanding-window OLS, 2022-2025 (1,139 games):
win accuracy **63.8%** (M1: 64.1%, M0: 63.6%), Brier 0.223 (M1: 0.224),
ATS still **below breakeven in every bucket**, rule ≥4 at 48.7% (M1:
47.6%). Starter-level QB detail is a wash to a slight loss vs. team-level
EPA — the market already prices starter/injury information, same
conclusion the moneyline test reached. No `MODEL_VERSION` bump;
`predict_week.py` stays on `srs-v1`. Full numbers + interpretation
(including the catch%/EPA collinearity caveat on the fitted weights) in
`intermediate/qa_log.md`.

This was the highest-prior item on the exploration backlog below — it's
now closed with a real answer, not a guess.

## Exploration backlog (from post-game analyses, Sep 2026)

Ideas the user wants to test from watching SF@LA and DET@BUF box
scores. **Caution:** these were born from post-hoc readings of two
games — exactly how bad features get made ("what explained THIS game
must predict the NEXT one"). They go in the backlog to be TESTED by
the honest loop (as-of backtest vs M0/M1 on identical games → only if
it wins: MODEL_VERSION bump, log, grade), not to be believed. Ranked
by prior:

1. **Short-window pressure rate (trailing 1-2 games)** — "a protection
   weakness exposed last week gets exploited again" (the DET/BUF note).
   Distinct from the season-trailing pressure already in the blend.
   O-line injuries/breakdowns cluster in time; pressure-allowed is one
   of the more stable week-to-week team stats. Test: `pressure_diff_recent`
   (last 2 games, as-of) in `backtest_blend.py` FEATURES. Prior:
   MEDIUM-HIGH — best idea of the batch; but the market reads injury
   reports too.
2. **Recency-weighted SRS** — the umbrella version of the same
   instinct: weight recent games' margins more in the SRS solve (decay
   parameter in `game_margins` before `solve_srs`). This was already on
   the candidates list; the "short-term form" observation is another
   vote for it. Prior: MEDIUM.
3. **Trailing turnover margin** — "the turnover battle flipped the
   game." Honest prior: LOW — turnovers are high-variance and barely
   autocorrelated week to week; fumble recovery is luck, INT rate has
   only modest QB-driven stability. Cheap to test the same way; expect
   nothing, keep only if the backtest insists.
4. **Recent red-zone TD rate** — short-window version of the RZ
   component. Prior: LOW-MEDIUM — red-zone numbers are small-sample
   noisy even at season level; as a 1-2 game window it's mostly noise.
5. ~~**Phase 7b (per-QB starter EPA in the model)**~~ — **TESTED
   2026-09-18, did not win** (63.8% vs M1's 64.1%, no ATS value; see
   above). Was the highest-prior item; closed.

## Segment tests (Sep 2026) — slicing existing results, no new model

No public betting-split/handle data source found (the "fade the public"
idea from qa_log has no free historical data), so pivoted to what
`games.csv` already carries but the model ignores.

1. ~~**Divisional games vs non-division**~~ — **TESTED 2026-09-18,
   no edge.** Win acc is worse in division games (61.8% vs 65.3%, as
   expected) but ATS is below breakeven in BOTH (48.6% / 47.5%) — the
   market already prices the extra unpredictability. Closed.
2. ~~**Rest differential / short week / bye week**~~ — **TESTED
   2026-09-18, no edge.** Every bucket below breakeven (45-48%).
   Notable: home-off-bye win accuracy is 72.8% (model's best bucket)
   but that bucket's ATS is 45.7% — the model being MORE right doesn't
   help, because the line already reflects it. Closed.
3. ~~**Totals (over/under) market**~~ — **TESTED 2026-09-18, no edge.**
   `backtest_totals.py` (Phase 8): trailing points-for/points-against per
   team, as-of, cold-start blended (shrink toward each team's own
   prior-season rate, itself pulled toward that season's league average —
   the general form of the SRS/EPA `shrink * prior` formula for a stat
   that isn't already zero-centered), weights fit by expanding-window OLS.
   2021-2025, 1,424 games: cover rate 46.9%-51.3% across |edge| buckets,
   below breakeven everywhere pooled. Splitting under-only shows one
   bucket above breakeven (|edge|>=3: 54.8%, ROI +5.6%, n=104) but no
   monotonic pattern across buckets and a raw predicted-total cutoff table
   (55/49/52/55/53% under-rate across five bands) shows no usable
   threshold — reads as noise. "Bad teams -> under" doesn't survive
   honest as-of testing. Full numbers in `intermediate/qa_log.md`. Closed.
4. ~~**Weather-conditioned totals** (temp/wind/roof/surface)~~ — **TESTED
   2026-09-18, mostly nothing + ONE flagged signal.**
   `backtest_weather.py` (Phase 10): temp bands dead, roof priced,
   Thursday exactly 50.0%, division 51.9%. BUT wind is the project's
   first dose-response + era-stable signal: blind under at 11-16 mph is
   55.5% lifetime (n=945) and 59.1%/60.3% in the last two eras; the
   market lowers wind totals only ~half as much as scoring drops
   (fitted wind coef −0.23 pts/mph, market miss −1.4 to −1.6 pts).
   Flagged as a FORWARD-TEST candidate (blind under, outdoor, wind
   ≥11 mph), same standing as road_dog at this stage — not a rule
   until a graded paper trail exists. Full numbers in qa_log.
   **Forward-test tool BUILT 2026-09-18:**
   `systems/wind_under/wind_under_system.py` (weekly qualifiers via
   Open-Meteo stadium forecasts; `--backtest` pooled 55.2% under,
   ROI +7.0%, above breakeven in all four eras; paper trail 1-0 after
   week 1). Open question for the trail: does the 16+ mph band stay in
   the rule (recent-era decay) — qualifiers are band-tagged to decide.
5. ~~**Pressure as tiebreaker when QBs are even**~~ — **TESTED
   2026-09-18, no edge.** `backtest_pressure.py` (Phase 9): the literal
   claim holds (lower-prate team wins 55.2% when |epa_diff|≤0.05) but
   is NO stronger than unconditional (56.4%) — protection is a general
   quality marker, not a tiebreaker. When both model and market say the
   teams are even, it's 49.5% — a coin flip. ATS nowhere robust;
   dose-response not monotone. Closed.
