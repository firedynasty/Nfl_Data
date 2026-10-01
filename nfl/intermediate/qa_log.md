# NFL win/ATS prediction — session Q&A log

Working directory for everything below: `/Users/stanleytan/Documents/technical/github/nfl_data`

A running record of the questions asked and answers/findings reached while
figuring out how to turn the existing NFL box-score toolkit into a win
(and eventually ATS) prediction pipeline. Written in order. Marks which
parts are **built** (real code, run, checked in) vs. **discussed only**
(a plan, not yet code) so it's clear what's real vs. proposed.

---

**Q: What's already in `nfl_data`, and how far along is it?**

A: Much further along than a comparable NBA-data project checked at the
start of this session. Existing pipeline (all **built**, pre-dating this
session):
- `nfl_box_score_analysis.py` — pulls schedule + play-by-play straight from
  nflverse's GitHub releases (no scraping, no API key, no rate limits).
  Builder-pattern functions, one row per `game_id` + `posteam`: first
  downs, yards/play, red zone TD/score rate, team rushing, QB hits/sacks,
  turnover margin.
- `streamlit_scoreboard.py` — a UI on top of that (deployed at
  `nflseasondata.streamlit.app`): pick season/team/week in the sidebar,
  see played games with box scores, CSV export.
- `team_composite.py` — **Game Performance Score**: season-total stats per
  team, each percentile-ranked across the 32 teams, weighted into one
  0-100 number. Validated at 0.66 correlation with scoring margin at the
  time (7 components, pre-EPA).
- `qb_support_chart.py` — QB aDOT-vs-catch% scatter, colored by that score.
- Everything here is **descriptive/postdictive** — it explains a game
  *after* it happened. Nothing took stats from before week N to predict
  week N. That gap is the throughline for the rest of this log.

---

**Q: There was a note about "Tom Brady 2022" — what was that?**

A: A saved prompt at the bottom of `FILES.md` from an earlier session,
not yet run: build a 2022 QB CSV, join the Game Performance Score onto it,
chart it, to see where Brady/2022 Bucs land. It predates the win-prediction
idea — it's about visualizing one season, not predicting outcomes.

---

**Q: Would QB play (not just team box-score stats) add real predictive
accuracy? Would a classifier "pick it up" automatically anyway?**

A: Yes it should add signal, and no, a classifier trained only on
team-level stats wouldn't reconstruct it on its own — those stats blend
QB play with rushing/O-line/game-script noise, so QB skill specifically
is diluted. A model can't un-blend a signal it was never given directly.
Recommended **EPA/dropback** (Expected Points Added per pass attempt/
sack/scramble — a value stat) over **aDOT** (air yards per attempt — a
style stat, doesn't say if the QB is good, just how he throws) as the
actual predictive QB feature.

---

**Q: Is `team_composite.py`'s Game Performance Score what's used for the
chart's colors?**

A: Confirmed by reading both files: yes, exactly one thing — the dot
color (green=high, red=low via `RdYlGn`) and the number printed in each
QB's label. The chart's x/y axes come from the QB CSV directly, not from
the composite.

---

**Q: [Confirmed] Add EPA/dropback into the predictive composite.**

A: **Built.** Added `qb_epa_by_team_game()` to `team_composite.py` (same
local-builder pattern as the existing `pressure_allowed_by_team_game` —
composite-specific builders live in this file, not in
`nfl_box_score_analysis.py`). Added it as an 8th weighted component
(`epa_per_dropback`, weight 0.17), rebalancing the other 7 weights down to
keep the sum at 1.0. **Validated head-to-head**, not just eyeballed: same
2023-2025 data (2,600+ games), old weights (no EPA) scored 0.615
correlation with scoring margin; new weights (with EPA) scored 0.661 — a
real, sample-robust gain (a single-week-2026-only check showed a bigger
jump, 0.56→0.67, but flagged as too small a sample, 32 games, to trust
alone). Documented in `STATS_REFERENCE.md`.

---

**Q: New chart with the updated composite, and swap aDOT for EPA/dropback
on the x-axis?**

A: **Built.** No script existed to build the per-QB CSV at all — the old
`output/qb_adot_catchpct.csv` was made ad hoc in an earlier session.
Created `build_qb_csv.py` (per-QB dropbacks, EPA/dropback, completions,
catch%, team record, from nflverse play-by-play). Updated
`qb_support_chart.py`: x-axis is now EPA/dropback instead of aDOT, title/
labels updated, still colored by Game Performance Score (which now
includes EPA automatically). Ran the full pipeline end to end
(`build_qb_csv.py` → `team_composite.py --qb_csv` →
`qb_support_chart.py`) and confirmed the chart: high-EPA/high-catch% QBs
cluster green (T.Lawrence, J.Dart, C.Williams), low-EPA cluster red
(C.Rush, B.Nix, D.Watson) — sanity-checks correctly since EPA/dropback is
now literally one of the score's own components.
Output: `~/Downloads/qb_epa_catchpct_game_performance.png` (+ backing
CSVs in `~/Downloads/`).

---

**Q: What files need to change to turn this into an actual predictive
classifier (trailing window, not season totals)?**

A: **Discussed only — not yet built.**
- *Reused as-is:* `nfl_box_score_analysis.py` (builders already return
  one row per team-per-game, the right granularity) and
  `team_composite.py`'s `team_game_stats()` (same reason).
- *Needs modification:* `team_composite.py`'s `team_season_table()`
  collapses a whole season into one row — doesn't fit prediction. Add a
  sibling function, e.g. `rolling_team_features()`, that sorts by week
  per team and `.shift(1)`s before averaging so game N only sees games
  <N. Keep `team_season_table()` for the descriptive/chart use case.
- *New files needed:* `build_matchup_features.py` (home vs away trailing
  stats + `home_team_wins` label, one row per game), `train_win_
  classifier.py` (fit with a **time-respecting** train/test split — never
  shuffled, or the future leaks into training), `predict_week.py` (apply
  the trained model to an unplayed week's schedule).
- *Not needed for the classifier:* `build_qb_csv.py` /
  `qb_support_chart.py` stay on their own descriptive track — team-level
  QB EPA already flows into the classifier's input via
  `qb_epa_by_team_game` inside `team_composite.py`.

---

**Q: What's the actual end output? Like if I ask "week 2," what do I get?**

A: **Discussed only.** Per game: predicted winner + win probability, e.g.
`DAL @ NYG → NYG 62%`. Two delivery options, not mutually exclusive:
(1) CLI (`predict_week.py --season 2026 --week 2`) — build/validate this
first. (2) A new "Predictions" view added to the already-deployed
`streamlit_scoreboard.py` (`nflseasondata.streamlit.app`), same look as
the existing season/team/week picker but pointed at an unplayed week.

---

**Q: How would I save `home_win_prob` and post-analyze how correct the
model was, then tweak it?**

A: **Discussed only.** Three pieces:
1. **Log at prediction time** — append-only `predictions_log.csv`, one
   row per game, written *before* the game is played, tagged with a
   `model_version` string so later tweaks are attributable.
2. **Grade after results land** — `grade_predictions.py`: join the log
   back to `load_games()` (now scored) on `game_id`, compute accuracy,
   **Brier score** (`(prob - actual)²` — punishes overconfidence, not
   just wrong picks), a calibration check (do "70% confidence" picks
   actually win ~70% of the time), and a baseline comparison (does the
   model beat "always pick home team," historically ~57%?).
3. **Close the loop** — because every row carries `model_version`,
   grading can compare accuracy/Brier score across versions on the same
   graded weeks — the honest A/B test for whether a tweak actually helped.

---

**Q: It'd be interesting to predict week 1, then check how correct —
worth trying?**

A: **Built (a one-off backtest script, not saved as a permanent file).**
Genuinely the cleanest possible test: week 1 has zero in-season trailing
data, so the only honest input is *last* season's numbers — no rolling-
window leakage possible by construction. Ran it: 2025 season-end Game
Performance Score picked the winner in only **5/16 (31.2%)** of 2026 week
1 games — worse than "always pick the home team" (**10/16, 62.5%**), and
worse than a coin flip. Real finding, not noise: prior-season team
aggregates don't carry cleanly across an offseason (roster/coaching
turnover, mean reversion). Flagged home-field advantage as the baseline
any real model needs to clear.

---

**Q: What about combining Game Performance Score with the QB's own
EPA/dropback and catch%?**

A: **Built (backtest, not a permanent script).** Same 16 week-1-2026
games, several modes tested:

| Predictor | Accuracy |
|---|---|
| Team Game Performance Score alone (2025) | 31.2% |
| Returning QB's own 2025 EPA/dropback alone | 56.2% |
| Returning QB's own 2025 EPA/dropback + catch% | 62.5% (ties home baseline) |
| Hand-picked blend (team score + QB EPA, arbitrary weights) | 43.8% (worse than QB alone) |
| Home team always wins (baseline) | 62.5% |

Individual QB persistence across an offseason clearly beats team-level
aggregate persistence. Also an explicit lesson: my hand-picked blend
formula made things *worse* than the QB signal alone — evidence that
weighting needs to be **fit** (logistic regression) not eyeballed, not
evidence that combining signals is a bad idea. Caveat raised: only 16
games, one week.

---

**Q: More backtest, please (wider than one week).**

A: **Built.** Same approach across 5 season openers: 2021→2022,
2022→2023, 2023→2024, 2024→2025, 2025→2026 (80 games pooled — close to
the ceiling since nflverse's earliest season is 2021).

| transition | qb_epa_catch | team_score_only | home_baseline |
|---|---|---|---|
| 2021→2022 | 50% | 69% | 38% |
| 2022→2023 | 50% | 56% | 38% |
| 2023→2024 | 81% | 75% | 69% |
| 2024→2025 | 69% | 56% | 56% |
| 2025→2026 | 62% | 31% | 62% |
| **Pooled (80 games)** | **62.5%** | **57.5%** | **52.5%** |

This overturned the single-week read: pooled, QB EPA+catch (62.5%) beats
team score (57.5%) beats home-field alone (52.5%) — but the earlier
"home baseline ties QB signal at 62.5%" and "team score is useless at
31.2%" were both **artifacts of that one unusual week**, not stable
truths. No predictor wins every year (team score actually beats QB signal
in 2021→22 and 2022→23) — that year-to-year instability is itself the
argument for fitting a real regression on top of both signals rather than
trusting either alone, and for widening backtests before trusting a
single-week result generally.

---

**Q: [User pasted a detailed explainer on ATS/spread betting.] Is there a
metric that tells me how good/bad a team is, comparable to a betting
line?**

A: Two distinct problems clarified:
1. **Win prediction ≠ ATS.** Everything built so far predicts *who wins*
   (classification). ATS needs a *point margin* prediction compared
   against the market's line — a good win-predictor can still be useless
   against the spread if it doesn't estimate margin.
2. **The Game Performance Score isn't a power rating in the sense ATS
   needs.** It's a percentile blend (0-100), not opponent-adjusted (no
   strength-of-schedule correction) and not expressed in point-spread
   units (can't subtract two percentile scores and get a spread).
   Proposed **SRS (Simple Rating System)**: `team rating = avg scoring
   margin + avg opponent rating`, solved iteratively — comes out in
   points, directly comparable to a spread. Noted the data needed
   (`team_score`, `opp`, `opp_score` per game) already exists in
   `team_composite.py`'s `team_game_stats()`. **Discussed only, not yet
   built.**

---

**Q: Does nflverse have free historical betting lines (spread/total/
moneyline) to backtest SRS against?**

A: **Checked — yes, already present, no new data source needed.** The
exact same `games.csv` already loaded via `load_games()` in
`nfl_box_score_analysis.py` has `spread_line`, `home_moneyline`,
`away_moneyline`, `total_line`, `home_spread_odds`/`away_spread_odds`,
`under_odds`/`over_odds`, `home_qb_name`/`away_qb_name` — going back to
**2006, 5,312 games with spread data**.

**Sign-convention gotcha, verified empirically (not from docs — checked
directly against the data):** `spread_line` is from the **home team's
perspective, positive = home favored** — the opposite of the "-7" shorthand
used in normal betting talk. Confirmed by checking: when the home team is
a big moneyline favorite (`home_moneyline` < -300), average `spread_line`
is **+10.1** and the average actual home margin (`result` column,
`home_score - away_score`) is **+11.1** — consistent only if positive
`spread_line` means home favored. Easy sign-flip bug if built the naive
way. `result` is already signed and ready to compare directly against a
predicted margin.

**Next proposed step (not yet built):** build the SRS rating, with the
spread-line comparison wired in from the start (predicted home margin
minus `spread_line` = the "edge"; compare against `result` to see if that
edge predicts covers).

---

**Q: How should the betting rule handle "edge," and what counts as a bet?**

A: **Decided (recorded in `plan.md` → "Betting rule (decided)").** After
walking through what edge means (model's predicted margin minus the
spread — the size of the disagreement with the book, in points), the
settled rule is: **bet any game where `|edge|` >= threshold, in whichever
direction the edge points** — no "predicted winner only" filter (an
earlier idea of never betting the predicted loser +points was dropped).
Threshold placeholder = 4 points; the real number comes from Phase 5's
edge-bucket cover rates (smallest bucket clearing the 52.4% breakeven at
-110 odds). Moneyline bets are a separate test: model win probability vs.
moneyline-implied probability. Expect only ~1-3 qualifying bets/week at a
4-point bar.

---

**Q: Build Phase 1 — SRS rating (season-total).**

A: **Built.** `srs.py`: `game_margins()` (per team-game margin, optional
±cap) → `solve_srs()` (iterative: rating = avg margin + avg opponent
rating over games faced, re-centered to mean 0; converges in ~40
iterations) → 32-row table with `mov` and `sos` columns so the schedule
correction is visible per team. Validated three ways:
1. League average = 0 (by construction, confirmed both seasons run).
2. In-sample per-game home-margin prediction beats raw avg margin —
   2025: corr .604 vs .592, MAE 9.07 vs 9.18; 2024: corr .608 vs .603.
   Small, as expected in-sample; the real bar is Phase 5 walk-forward.
3. **PFR spot-check passed** against their published 2025 SRS (from a
   browser-saved standings page — PFR 403s automated fetches): corr
   **0.9994**, MAE **0.20 pts**, max abs diff **0.25** — but only when
   scoped to regular season. Finding: **PFR's SRS excludes playoff games;
   `srs.py` includes them** (kept — for prediction, every completed game
   is information; filter `game_type == 'REG'` only to match PFR).
Documented in `STATS_REFERENCE.md`. Next per the plan's build order:
Phase 5 walk-forward harness on season-total SRS (rough first read),
then Phase 2 (as-of-date SRS, no leakage).

---

**Q: Build the Phase 5 walk-forward harness, run on season-total SRS
(the leaky rough read, build-order step 2).**

A: **Built.** `backtest_srs.py`: walks season → week → games calling a
`rating_fn(season, week)` hook (the only leakage-sensitive piece — the
stub returns season-total SRS; Phase 2 swaps in as-of ratings behind the
same hook). HFA and margin-error std are **estimated from data**, not
guessed; `BREAKEVEN_COVER_PCT = 52.4` is a named constant per the plan.
ATS grading: bet = side the edge points to (the decided rule), pushes
excluded from cover rates.

Rough run, 2021–2025 pooled (1,424 games, 32 neutral-site):

| Metric | Leaky result |
|---|---|
| HFA / margin-err std | +2.10 pts / 11.61 pts (std low — leak shrinks residuals) |
| Win accuracy | **70.4%** (home baseline on same games: 54.5%) |
| Brier | 0.195 |
| ATS 0-1 pt edge | 50.0% (n=202) — below breakeven |
| ATS 1-2 pt | 57.6% (n=217) — above |
| ATS 2-3 pt | 64.8% (n=193) — above |
| ATS 3+ pt | 69.3% (n=779) — above |
| Betting rule \|edge\| ≥ 4 | **71.1%** (n=616) |

**Read: this is the ceiling, not the truth.** The leak shows up exactly
where theory says it should — big edges self-justify because the rating
already saw the blowouts that created them. The bucket monotonicity
(50→58→65→69%) is real structure the honest version should preserve in
weaker form; the absolute percentages will fall. The honest bar for
Phase 2's no-leakage re-run: 3+ bucket and the ≥4 rule line still
clearing 52.4%. Same harness, one-line swap (`rating_fn`), so the delta
between runs = the exact cost of the leakage fix.

---

**Q: Build Phase 2 — as-of-date SRS (no leakage) — and re-run the
backtest honestly.**

A: **Built, and the honest number changes everything.** `srs.py` gained
`solve_srs_asof()` (SRS from games with `week < W` only) and
`blended_asof_ratings()` — per-(season, week) ratings with a cold-start
blend: `rating = g/(g+k)·srs_current + k/(g+k)·shrink·srs_prior`
(k=4 pseudo-games, shrink=0.7, both flagged placeholders pending a
sweep). Two implementation finds: (1) the undamped iteration period-2
**oscillates forever** on ~1-game-per-team early-season graphs (near
perfect matchings) — fixed with a damped half-step update, verified the
season-total table is bit-identical afterward; (2) table keys come from
the schedule (not played games) so Phase 4 can ask about upcoming weeks.
`backtest_srs.py --asof` swaps the `rating_fn` hook, as designed.

Honest run, 2021–2025 (1,424 games), vs the leaky run:

| Metric | Leaky | **Honest (as-of)** |
|---|---|---|
| Win accuracy | 70.4% | **63.2%** (baseline 54.5%) |
| Brier / std | 0.195 / 11.6 | 0.229 / 13.44 (≈ canonical 13.5 ✓) |
| ATS 0-1 pt | 50.0% | 50.9% (n=283) |
| ATS 1-2 pt | 57.6% | 50.5% (n=295) |
| ATS 2-3 pt | 64.8% | 51.4% (n=251) |
| ATS 3+ pt | 69.3% | **45.6%** (n=562) |
| Rule \|edge\|≥4 | 71.1% | **46.5%** (n=398) |

**Read:** win accuracy holds up fine — SRS is a real rating. But ATS is
not just below breakeven, it's **inverted**: the bigger the model's
disagreement with the market, the worse it does. That's the classic
signature of "the market knows things the model doesn't" (injuries, QB
news, form) — at big disagreements the market is right, not the
season-average. Betting \|edge\|≥4 on raw SRS would lose ~11¢ per $1 at
-110. **The 71.1%→46.5% delta is exactly the leakage the plan warned
about** — and catching it before real money is the pipeline working as
intended. Bottom line: raw SRS alone is not a betting signal. Noted but
flagged as in-sample-only: "fade big SRS edges" (3+ bucket implies
~54.4%) — extracting a rule from the data that grades it is overfitting;
any such variant needs fit-on-2021-2023 / verify-on-2024-2025 discipline
first. Next candidates per plan: constant sweep (k/shrink/cap, recency
weighting), then Phase 7's QB-EPA blend (QB signals carried independent
predictive weight in the opener backtests).

---

**Q: Phase 7 — blend SRS with trailing QB EPA/dropback + pressure rate,
weights FITTED not guessed.**

A: **Built and honestly graded.** `backtest_blend.py`: trailing as-of
EPA/pressure features (team-game sums, same k/shrink cold-start blend
discipline as SRS; rates computed from trailing sums so dropback volume
weights correctly), expanding-window OLS — season S is predicted by a
model trained only on seasons < S. M0 = SRS-only refit through the same
window as the apples-to-apples baseline. Team-level EPA for now (per-QB
trailing with starter detection = Phase 7b).

Fitted weights (per eval season): `srs_diff` 0.52–0.66; `epa_diff`
**+9.8 to +13.4, stable** — a real, sensibly-signed contributor;
`prate_diff` unstable on the smallest train sample (+20.1 for 2022) then
stabilizing negative (−7.8 → −15.7 — sensible sign, but likely collinear
with EPA: sacks already tank EPA); `home_flag` 0.9 → 1.8 as the training
window grows.

Results, 2022–2025 (1,139 games, breakeven 52.4%):

| Metric | M0 SRS-only (fitted) | M1 blend (fitted) |
|---|---|---|
| Win accuracy | 63.6% | **64.1%** (baseline 55.2%) |
| Brier | 0.226 | 0.224 |
| ATS 0-1 / 1-2 / 2-3 / 3+ | 48.6 / 47.5 / 42.3 / 49.6 | 48.9 / 51.2 / 44.9 / 49.9 |
| Rule \|edge\|≥4 | 51.0% (n=245) | 47.6% (n=227) |

**Verdict:** the blend adds +0.5pt win accuracy and a hair of Brier —
EPA is real but a small add on top of SRS. ATS: **nothing clears 52.4%
in either model.** Two model classes now honestly tested, same
conclusion: ~64% win prediction (near Vegas-favorite level) but no
spread value vs the market. This reproduces the well-documented
efficiency of NFL lines against public stats — not a bug, the market
being sharp. (Bucket n's are 180–380, SE ~±3-4pt, so M0-vs-M1 bucket
differences are mostly noise; the consistent sub-50% pattern across
ALL buckets of both models is the signal.)

Untried levers, in order of expected value: (1) **moneyline prob-edge
test** — Phase 3's other branch, different market and math (model win
prob vs implied prob); (2) per-QB trailing (7b) for starter changes;
(3) recency weighting; (4) k/shrink sweep; (5) the fade hypothesis —
only as a pre-registered rule verified on a true holdout. 2026 is a
pristine holdout if picks get logged before kickoff (Phase 6).

---

**Q: Moneyline prob-edge test — does the model's win probability beat
the moneyline-implied probability?**

A: **Built and graded.** `backtest_moneyline.py`: M1 blend probabilities
vs. raw implied probs from `home_moneyline`/`away_moneyline` (raw =
vig included = the real breakeven; ties refunded). Feature frame cached
to `blend_features_cache.csv` so grading iterates in seconds. Two rules:
**Rule A** (price shop: bet the bigger edge each game) and **Rule B**
(choose prediction: bet the predicted winner, only when its price edge
clears a threshold). Calibration measured on the predicted winner with
NO edge selection — selecting by edge contaminates it with disagreement
error (first draft of this script made exactly that mistake; the
"terrible calibration" was the selection artifact, not the model).

Results, 2022–2025 (1,136 games with moneylines):

1. **Calibration is genuinely good.** Predicted-winner bands: model
   55.0→actual 57.0 (n=532); 64.8→67.5 (n=363); 74.4→71.7 (n=184);
   83.2→84.2 (n=57). Slightly UNDER-confident if anything. The model's
   probabilities are honest.
2. **Rule A is a catastrophe, instructively:** every bucket loses
   (overall −11.5% ROI; 5–8pt edge bucket −20.8%). Betting the bigger
   edge systematically selects underdogs the model overrates vs. the
   market — it SELECTS ON MODEL ERROR, the moneyline twin of the ATS
   inversion.
3. **Rule B is the vig and nothing else:** bet every predicted winner →
   −1.9% ROI (≈ the book's margin on a 64.1%-accurate picker). No edge
   bucket clears breakeven; rule ≥5pt → −7.3%. Most damning detail: in
   the 8+pt edge bucket the market implied 44.7% and the actual win rate
   was **44.6%** — the market was dead-on; the model's disagreement was
   pure error.

**Synthesis (now triple-confirmed):** the model is well-calibrated and
near-market quality at picking winners (~64%), but its disagreements
with the price are where IT is wrong, not the market — on both the
spread and the moneyline. The model can say who wins; it cannot find
where the price is wrong, because the price already contains everything
it knows plus injury/news it lacks. No tested rule — spread \|edge\|≥4,
ML price-shop, ML predicted-winner-at-a-price — clears breakeven.
Remaining candidates are thin: Phase 6 live logging on the 2026 holdout
(free honest grading of anything pre-registered), per-QB 7b (market
prices injuries instantly; unlikely), or a genuinely different market —
player props are the known soft spot but need player-level lines data
nflverse doesn't carry.

---

**Q: The user clarified the main goal: not a bet-signal factory — a way
to SEE how good each team is (SRS as a new stat, comparable to a betting
line). Put it in the Streamlit app?**

A: **Built.** `predict_week.py` (Phase 4, CLI-first per the project's
script-then-app pattern): one row per game for any season/week — both
teams' as-of SRS, model fair line vs. market line in betting notation
(BUF -9.7), edge (home-positive), predicted winner + calibrated win%,
and final score + ✓/✗ grading for played weeks. Constants HFA 2.2 /
std 13.44 from the honest backtest, named in one place. Then wired into
`streamlit_scoreboard.py` as a third "Predictions" view (View by
control), skipping the pbp load since SRS needs only scores; ratings
cached per season with a 1-hour ttl so new results flow in. Smoke-tested
both views with Streamlit's AppTest. First live read, 2026 week 1
(prior-season-only ratings): 9-7 on winners — fine for a cold start.
Every surface carries the honest-backtest reminder ("how good the teams
are, not what to bet"). Deploys to nflseasondata.streamlit.app on the
next git push.

---

**Q: Build Phase 6 (live logging + grading)? And how should future
tweaks work?**

A: **Built.** `predict_week.py --log` appends unplayed games to
`predictions_log.csv` with `MODEL_VERSION` (`srs-v1`, in
predict_week.py — bump on ANY model change) and timestamp. Discipline
enforced in code: already-played games refused (post-hoc), duplicates
refused. `grade_predictions.py` joins the log to results: per version —
win acc, Brier, ATS cover, |edge|≥4 line, home baseline, with the
"compare versions only on shared weeks" rule printed. First 15 live
predictions logged for 2026 week 2 before kickoff; grades land Sunday.
**Tweak loop (the user's workflow question):** (1) one-sentence
hypothesis, (2) honest backtest first — cheap, and if it doesn't beat
v1 there, stop; (3) if it passes, bump version and log live; (4) the
shared-weeks comparison in grade_predictions.py is the verdict that
counts. One tweak at a time; never tune ON the live log — it's the
holdout.

---

**Q: Readability requests + clarification: does `pred`/`pred_right` mean
who WINS (not who covers)?**

A: Confirmed: `pred` = predicted outright winner; the spread only exists
in `market_line`/`edge` and the sheet never grades covers. Then built:
(1) `final` now renders as `27(49ers)-7(Rams)` (nicknames, away-home);
(2) app "Attach prediction row to games" toggle — every game card in
Week/Team views gets a 🔮 line under the score (SRS pair, model vs
market line, pred + win%, ✓/✗ on played games); (3) "How are these
predictions made?" expander in the Predictions view + a Weekly
predictions section in STATS_REFERENCE.md documenting every column.
All app paths re-smoke-tested.

---

**Q: Add the QB to the predictions (the QB EPA/catch% chart is generated
in this project too)?**

A: **Built, as display context — explicitly NOT in the model** (srs-v1
stays pure SRS; log/grading unaffected, no version bump). Name bridge
`to_pbp_name()` in predict_week.py (games.csv full names → pbp style:
"Jared Goff" → "J.Goff", suffix-aware). Three surfaces: CLI table gains
away_qb/home_qb columns; app combine mode shows "Team · J.Goff (+0.02)"
next to each team name; Predictions table shows "J.Allen (+0.45)". EPA
is trailing through prior weeks (`cached_qb_epa_asof`, reindexed over
byes) — verified equal to the QB chart's x-axis (J.Goff +0.02, J.Allen
+0.45, C.Williams +0.45 after week 1). QBs with no prior dropbacks
(week 1, or not yet played this season) show name only — honest, not
zero-filled. Explainer + STATS_REFERENCE updated.

---

**Q: Can't the QB EPA trailing reach back to the previous season?**

A: **Yes — built.** Same cold-start discipline as the SRS blend:
`epa_db = db/(db+K)·current + K/(db+K)·shrink·prior`, K=150
pseudo-dropbacks (~4-5 games), shrink=0.7 (year-over-year regression;
both flagged placeholders like SRS's k/shrink). Week 1 is mostly the
regressed prior, current season takes over as dropbacks accumulate;
QBs with no prior (rookies) are current-only; QBs with a prior but no
dropbacks yet this season (injured starters) still show the regressed
prior. Two bugs found by testing, both fixed: (1) `0 × NaN = NaN`
poisoned the blend exactly when w=0 (week 1 showed nothing) — cur is
zero-filled before the multiply; (2) prior-only QBs were absent from
the lookup frame entirely. Verified: week 1 2026 shows regressed-2025
numbers for veterans (D.Maye +0.12, M.Stafford +0.15, B.Young −0.04);
week 2 blends week-1 actuals with the prior (J.Allen +0.15 vs. pure
week-1 +0.45). Justification already in the log: the opener backtests
found a returning QB's own prior-season EPA was the best week-1
predictor tested.

---

**Q: Add catch% too? And where would a QB composite (EPA + catch%) plug
into the MODEL in a future session?**

A: Catch% **built** (completions / pass attempts, `build_qb_csv.py`'s
definition = the chart's y-axis), displayed as "J.Goff (+0.10, 63%)".
Caught a real bug while building it: the 0.7 shrink can't apply to a
percentage — shrinking catch% toward 0 turns 65% into 45%. Both stats
now regress **toward the league mean** instead (`lg + 0.7·(v − lg)`,
league means computed from the prior season); verified J.Goff ~63%,
J.Allen ~62% vs. chart's ~65%/61%. For the model question: wrote a
**Phase 7b pointer section in `plan.md`** — the per-QB machinery now
exists as display code (`cached_qb_epa_asof` + `to_pbp_name`); a future
session lifts it into `backtest_blend.py` keyed by listed starters,
adds `qb_epa_diff`/`qb_catch_diff` to FEATURES, re-runs the honest
backtest, and only if it wins: bump MODEL_VERSION, log, grade.

---

**Q: Can the Game Performance Score include SRS? Plus app polish: team
color badges instead of logos, and a team-jump button grid.**

A: All three **built** (user said make the changes per recommendation).
(1) **SRS replaced win%** in the composite (its old 0.18 seat — SRS
supersedes binary W/L: it contains winning + margin + schedule
adjustment, the one thing the composite lacked). Completed seasons use
the full-window solve; in-progress season uses cold-start-blended as-of
ratings (`composite_srs()`). SRS is excluded from game-level validation
for the same circularity reason win% was. Sanity-ran 2023-2025 and
2026 paths, both healthy (numbers intentionally not shown per request).
(2) Score-row logos replaced with `team_colors.py` badge chips (tight
inline-flex spans per the user's spec; LA→LAR alias handled).
(3) Team-jump nav grid: 32 color-coded `st.button`s (CSS scoped via
`st-key-btn_*` classes) above the cards; click sets
`selected_team`/`view_by`/`team_filter` via on_click callback (legal
pre-run state writes) and lands on that team's Team view. AppTest
verified badges render and a BUF click switches views; note AppTest
itself can't serialize keyed segmented_controls (harness-only quirk,
browsers fine). Commit includes the previously-uncommitted EPA-session
changes in team_composite.py (user-approved).

---

**Q: Chart button near the week dropdown — QB EPA/catch% scatter
colored by Game Performance Score?**

A: **Built.** "📈 QB chart" button in the sidebar (appears wherever the
week dropdown does — Week and Predictions views) toggles an in-app
Altair chart: x = EPA/dropback, y = catch%, color = Game Performance
Score (0-100, red→green), per-QB labels, hover tooltips with team /
record / exact numbers. Data is the app's own trailing-blend QB frame
(`cached_qb_epa_asof`, now carrying team) at the selected week, plus a
new `cached_composite_score` (team_composite pipeline incl. the new SRS
component). Altair instead of matplotlib deliberately — Streamlit Cloud
has altair bundled, matplotlib isn't a declared dependency. Filters to
QBs with >= 20 dropbacks this season (`MIN_CHART_DROPBACKS`, same
convention as build_qb_csv's default) — added after a screenshot showed
C.Johnston (a PUNTER with one fake-punt dropback at -1.85 EPA, 0%
catch) stretching the x-axis to -2.0 and squashing all real QBs into a
clump; the filter also clears one-play backups that made labels
unreadable. Week 1 shows an honest "no QB numbers yet" note. Later
additions: catch% y-axis floored at 50% (every charted QB is >55%
anyway; sub-50% was all outlier noise), and **force-directed label
repulsion** (`repel_labels`, pure numpy — the adjustText idea with no
matplotlib dependency, keeping Altair tooltips): labels start above
their dot, iterate label-vs-label / label-vs-point repulsion plus a
weak pull home, then render at settled positions with gray connector
lines. Verified: min label separation ~doubles while labels stay
tethered to their dots.

---

**Q: [User pasted post-game box-score analyses — SF@LA and DET@BUF —
noting turnovers, red zone, and especially trailing pressure ("weakness
exposed last week gets exploited again") as things SRS/market missed.
Wants to explore in the future.]**

A: Recorded as an **exploration backlog in `plan.md`**, framed as
testable hypotheses with honest priors, ranked: (1) short-window
trailing pressure rate 1-2 games (MEDIUM-HIGH — protection breakdowns
cluster; distinct from the season-trailing pressure already in the
blend), (2) recency-weighted SRS (MEDIUM — the umbrella version),
(3) trailing turnover margin (LOW — turnovers barely autocorrelate;
classic post-hoc trap), (4) recent red-zone TD rate (LOW-MEDIUM —
small-sample noise), (5) Phase 7b per-QB starter EPA (already the
highest-prior model change). Meta-caution recorded with them: ideas
born from "what explained this game" must earn "predicts the next one"
through the standard loop — honest backtest first, MODEL_VERSION bump
and live log only if it wins.

---

**Q: Phase 7b — does lifting the per-QB starter EPA/db + catch% builder
(`cached_qb_epa_asof` in `streamlit_scoreboard.py`, display-only until
now) into the model actually beat M1's team-level blend?**

A: **Built and graded — it does not win.** `backtest_blend.py --qb`:
new `qb_season_asof()` + `qb_starter_features()` lift the same as-of,
cold-start-blended EPA/db and catch% math (k=150 pseudo-dropbacks,
shrink=0.7 toward league mean) but keyed by the STARTER listed in
`games.csv` (`home_qb_name`/`away_qb_name`, bridged via `to_pbp_name`)
instead of by team — the thing that should catch injuries/benchings
team-level trailing EPA can't. Missing starters (name mismatches, true
no-data rookies) fall back to that season/week's league average so no
game is dropped. M2 = M1's four features + `qb_epa_diff` +
`qb_catch_diff`, same expanding-window OLS discipline (2022-2025 eval,
1,139 games, breakeven 52.4%):

| Metric | M0 SRS-only | M1 blend | M2 blend + QB starter |
|---|---|---|---|
| Win accuracy | 63.6% | 64.1% | **63.8%** |
| Brier | 0.226 | 0.224 | 0.223 |
| ATS 0-1 / 1-2 / 2-3 / 3+ | 48.6/47.5/42.3/49.6 | 48.9/51.2/44.9/49.9 | 51.6/49.6/49.5/50.0 |
| Rule \|edge\|≥4 | 51.0% (n=245) | 47.6% (n=227) | 48.7% (n=232) |

Fitted `qb_epa_diff` is unstable and small (−0.06 in 2022's tiny train
window, +0.9 to +2.3 once more data accumulates — right sign, weak);
`qb_catch_diff` is large and positive throughout (+10 to +20) but that's
suspicious, not exciting: catch% is collinear with EPA/db (accurate
QBs both complete more and are worth more per dropback) and the
regression is likely splitting one real signal across two correlated
columns rather than finding a second one.

**Verdict: starter-level QB detail loses a hair of win accuracy vs. M1
(63.8% vs 64.1%), nudges Brier down a hair, and still clears breakeven
in ZERO ats buckets** — same conclusion as every model tried so far
(M0, M1, M2): the market already prices starter/injury information
the model is trying to add here, which is exactly the concern flagged
when 7b was first proposed ("market prices injuries instantly;
unlikely to help"). This closes the highest-prior item on the
exploration backlog with a real answer instead of a guess. Remaining
backlog items (short-window pressure, recency-weighted SRS, turnover
margin, red-zone rate) are all lower-prior than this one was and share
the same likely fate, but are still untested.

No `MODEL_VERSION` bump — `predict_week.py` stays on `srs-v1` (M1's
constants); this was a backtest-only experiment, nothing changed in
the live pipeline. `cached_qb_epa_asof` in `streamlit_scoreboard.py`
remains display-only, unaffected by this test.

---

**Q: User couldn't find historical betting-percentage/handle data (the
"fade the public" idea has no data source), pivoted to what's cheaply
testable. Any hidden pocket in the "no edge anywhere" conclusion —
divisional games, rest situations?**

A: **Built and graded — no pocket found.** `backtest_segments.py`: no
new model, just re-slices M1's already-graded 1,139 games (2022-2025,
from the cached feature frame) by columns `games.csv` carries but the
model never uses: `div_game`, `away_rest`/`home_rest`.

- **Divisional games:** win acc 61.8% (n=393) vs 65.3% non-division —
  division games ARE harder to predict outright, as expected. But ATS:
  48.6% division vs 47.5% non-division — **both below the 52.4%
  breakeven, no gap**. The market prices the extra unpredictability
  into the line just as well as it prices everything else.
- **Rest differential** (home_rest − away_rest, bucketed ≤−4 / −3..3 /
  ≥4) and **named situations** (short week ≤6 days, off bye ≥12 days,
  either side): **every single bucket below breakeven**, 45.0%–47.9%.
  Notable non-result: home-off-bye win accuracy is 72.8% (n=81, the
  model correctly predicts these wins far more than its 64% average)
  but that same bucket's ATS cover is 45.7% — the model gets MORE right
  about who wins on rest advantages, and it STILL doesn't turn into
  spread value, because the line already moved to reflect it. This is
  the cleanest single illustration yet of "market prices what the model
  knows" — not just true on average, true in the exact subset where the
  model is most confident.

**Verdict:** two more free checks, same universal conclusion. Explicit
caution logged in the script itself: this is post-hoc slicing against
one breakeven line with no multiple-comparisons correction, so treat
any future bucket that clears it as a re-test candidate, not a rule.
Remaining backlog, cheapest-to-most-expensive: totals market (highest
prior — genuinely different market, `total_line`/`over_odds`/
`under_odds` already in `games.csv`, unused; needs a new small
points-for/against trailing model since SRS predicts margin, not
scoring pace), then weather-conditioned totals once that model exists,
then the original short-window-pressure / recency-weighted-SRS /
turnover-margin / red-zone-rate items (all lower prior, all untested).

---

**Q: Totals (over/under) market — the highest-prior remaining backlog
item. User specifically wants to know whether two bad teams trend
toward the under, and what the cutoff would be.**

A: **Built and graded — no edge, no clean cutoff.** New
`backtest_totals.py` (Phase 8): trailing points-for/points-against per
team, as-of (no leakage), cold-start blended toward each team's
PRIOR-season rate shrunk toward that season's league-average points —
the general form of the same `shrink * prior` formula SRS/EPA use,
which only looked like "multiply by shrink" there because their league
average happens to be ~0; points obviously aren't. Weights fit by
expanding-window OLS (season S trained only on seasons < S), same
discipline as Phase 7. No play-by-play needed, so training reaches
back to 2000 (`total_line` has full coverage back to 1999).

Graded 2021-2025 (1,424 games):
- **ATS cover overall: 46.9%-51.3% across |edge| buckets — below
  breakeven everywhere.** Same conclusion as every other model tried
  (spread, moneyline, all segment slices).
- **Under-only split:** 0-1pt 53.0% (ROI +2.1%), 1-2pt 50.9%, 2-3pt
  45.7% (worst bucket), 3+pt 54.8% (ROI +5.6%, n=104). No monotonic
  pattern — the one bucket clearing breakeven has a thin sample and no
  causal story for why it'd clear while the 2-3pt bucket doesn't.
- **Raw cutoff test** (bucket by the model's predicted total itself,
  not edge vs. the market — i.e. "does a low predicted total actually
  mean more unders, regardless of pricing"): 55.2% / 48.6% / 51.5% /
  55.3% / 53.3% across five predicted-total bands. Bounces around
  48-55% with no trend — reads as noise, not a usable cutoff.

Fitted weights are directionally sensible (home_pf/away_pf ~0.58-0.62,
home_pa/away_pa ~0.32-0.37, intercept 1.5-3.3 — both a team's own
scoring and its opponent's scoring-allowed push the total the expected
direction), so the model isn't broken. It's the same story as SRS:
knowing team quality doesn't beat the closing total any more than it
beat the closing spread.

**Verdict: closes the highest-prior backlog item — "bad teams → under"
does not hold up as a standalone rule once as-of/no-leakage tested,
despite being an intuitive story.** Same universal conclusion as every
market tested so far: the market prices scoring-pace information as
well as it prices everything else. Remaining untested: weather-
conditioned totals (item 4, was gated on this model existing — now
unblocked if worth pursuing), and the original lower-prior items
(short-window pressure, recency-weighted SRS, turnover margin,
red-zone rate).


---

**Q: Two user hypotheses from watching box scores: (1) "a team that
gives fewer QB pressures usually wins if the two quarterbacks are about
the same" — pressure-as-tiebreaker; (2) can weather data (already in
`games.csv`: temp/wind/roof/surface — no separate pull needed) point at
under bets?**

A: **Both built and graded. Pressure: the literal claim is true but
useless — protection is a general quality marker, not a tiebreaker, and
the market already prices it. Weather: nothing anywhere EXCEPT wind —
the first signal in this project with dose-response AND era stability.**

`backtest_pressure.py` (Phase 9): no new model, slices the cached as-of
features (2021-2025, 1,424 games). Pick = team with the lower trailing
pressure-allowed rate.

- **Unconditional:** lower-prate team wins 56.4% (n=1,424) — protection
  quality does pick winners outright. But ATS 51.8%, ROI −0.1% — the
  spread already reflects it.
- **Condition A (|epa_diff| ≤ 0.05, "QBs even"):** win 55.2% — the
  tiebreaker version is NOT stronger than unconditional (56.4%).
  Pressure helps equally everywhere; it is not amplified when the QBs
  are even.
- **Both model AND market say even** (|epa_diff|≤0.08 AND |spread|≤3):
  49.5% win (n=307) — a pure coin flip. No tiebreaker power at all in
  the exact slice the hypothesis names.
- **Control (|epa_diff| > 0.12):** lower-prate team wins 61.6% —
  protection correlates with overall passing quality, which explains
  why it "works" unconditionally.
- **Dose-response not monotone** (condition B across |prate_diff|
  bands: 48.5/48.0/57.3/52.6). The above-breakeven pockets (55.0% at
  |epa|≤0.05, 55.8% at |spread|≤2.5, one dose band at 59.2%) read as
  noise across ~10 cuts, not structure.

`backtest_weather.py` (Phase 10): raw cuts 2000-2025 (7,017 games with
closing totals) + walk-forward pace+weather model (2021-2025, Phase 8
harness, expanding-window OLS, both models on identical games).

- **Temp: nothing** (47.8-52.4% under across five bands, no trend).
  **Roof:** indoor 48.2% / outdoor 51.3% — priced. **Thursday:** exactly
  50.0%. **Division:** 51.9%. All dead.
- **Wind: the exception.** Blind under by outdoor wind band: 0-6 mph
  48.3% (n=1,518) → 6-11 50.2% (n=2,088) → 11-16 55.5% (n=945) → 16+
  55.5% (n=457). Dose-response plus a plausible mechanism (passing
  efficiency and FG range both degrade). Era split in the 11-16 band:
  53.9 / 51.0 / 59.1 / 60.3% across 2000-06 / 07-13 / 14-19 / 20-25 —
  NOT decaying; market miss −1.6 and −1.4 pts in the two recent eras
  (the market lowers wind totals, but only about half as much as
  scoring actually drops). The 16+ band decays (56.6 → 51.8%) on thin
  per-era n (~85-130) — possibly books sharpening on high-profile wind
  games while 11-16 stays the quiet band.
- The walk-forward model fits wind at **−0.23 pts/mph, stable across
  all five expanding windows** (a 15-mph game ≈ −3.5 pts off the
  total), yet pace+weather still doesn't clear breakeven pooled (49.4%
  at |edge|≥2) — consistent with "the close adjusts for wind, just not
  enough at 11-16 mph." Under-only at |edge|≥3: 58.1% (n=93), thin,
  same caveat as Phase 8's 3+ bucket.

**Verdict:** pressure-as-tiebreaker is CLOSED, no edge — when both the
model and the market agree the teams are even, protection picks winners
at coin-flip rate. Wind is flagged as the project's first forward-test
candidate FROM THE TOTALS MARKET: blind under, outdoor games with wind
≥11 mph (especially 11-16), ~57-60% recent-era cover vs the 52.4%
breakeven. Same standing as road_dog had at this stage — honest
backtest signal, parameters eyeballed from the data, unproven until a
forward paper trail exists. Weather caveats: nflverse temp/wind is an
airport-level daily snapshot (2022 has a source gap, n=96 with wind),
not stadium-hour-precise. Next step if pursued: a small weekly
wind-under qualifier overlay with graded paper-trail logging, modeled
on road_dog_system.py.

---

**Q: Wind angle confirmed worth pursuing — build the weekly qualifier
tool ("wind angle is good yes").**

A: **Built — `systems/wind_under/wind_under_system.py`, modeled on
`road_dog_system.py`, with one design problem solved: games.csv's
`wind` is ACTUAL game-time wind (recorded post-game), so unplayed games
need a forecast.** The tool joins Open-Meteo (free, no key, global —
covers London/Germany/Mexico games) at per-stadium coordinates for the
kickoff hour (gametime in games.csv is ET; converted to UTC for the
hourly lookup). Qualifier rule: outdoor game (`roof` in
outdoors/open), wind >= 11 mph -> bet under. Retractable-roof games
(`roof` NaN until the team announces) FAIL safe with "roof TBD -- likely
closes in wind; re-check". The `--results`/`--week N` paper trail
grades on ACTUAL recorded wind, with the divergence caveat printed
(live picks are made on forecasts).

`--backtest` (2000-2025, actual `under_odds` where present, -110
fallback): **pooled 55.2% under, ROI +7.0%, n=1,307 decided — ABOVE
breakeven in ALL four eras** (54.8 / 52.7 / 56.8 / 57.7%), and above
in both bands (11-15: 55.7%, +7.9%; 16+: 54.4%, +5.2%). The 16+ band
is much weaker in the recent era split from Phase 10 (~51-52% in
2014-2025), so qualifiers are band-tagged in the weekly/results views
and the forward trail will decide whether 16+ stays in the rule.

First graded bet is already in the book: week 1 2026, DEN @ KC Monday
night, actual wind 13 mph, under 42.5 -> final 41, WON +0.91u. Week 2
view worked end-to-end with live forecasts (one qualifier flagged, one
at 10.2 mph just under the threshold — exactly the re-check-Sunday
case the tool warns about).

**Verdict: system is live for forward-testing, same standing as
road_dog — honest backtest signal (positive in every era, mechanism
understood), parameters chosen post-hoc, unproven until the forward
paper trail accumulates.** Expect volume to cluster in Nov-Dec (the
2025 slate shows ~20 qualifiers, almost all weeks 6-14). Remaining
honesty gap vs road_dog: forecasts aren't stored, so the graded set is
reconstructed from actual wind; storing each run's forecast qualifiers
(a predictions_log.csv-style append) is the cheap upgrade if the trail
gets serious.
