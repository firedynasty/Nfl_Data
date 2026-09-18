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
