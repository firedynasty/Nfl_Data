# Files

Every file in this repo, what it does, and the order the pipeline was built
in. Newest work first in the listing, then the story in build order.

```
-rw-r--r--  1 stanleytan  staff   3698 Sep 16 12:45 qb_support_chart.py
-rw-r--r--  1 stanleytan  staff   5498 Sep 16 12:30 game_performance_score.csv
-rw-r--r--  1 stanleytan  staff   9249 Sep 16 12:29 team_composite.py
-rw-r--r--  1 stanleytan  staff  14986 Sep 16 12:28 STATS_REFERENCE.md
-rw-r--r--  1 stanleytan  staff  10875 Sep 15 12:41 streamlit_scoreboard.py
-rw-r--r--  1 stanleytan  staff  21005 Sep 15 12:32 nfl_box_score_analysis.py
-rw-r--r--  1 stanleytan  staff  17683 Jul  5  2024  Untitled.ipynb
-rw-r--r--  1 stanleytan  staff  32858 Jul  5  2024  2022_NFL_Season_Scores.csv
-rw-r--r--  1 stanleytan  staff  33003 Jul  3  2024  2023_NFL_Season_Scores.csv
-rw-r--r--  1 stanleytan  staff   1047 Jul  5  2024  top_quarterbackers_2022.csv
-rw-r--r--  1 stanleytan  staff    409 Jul  5  2024  2023_Top_Pass_Yards_Players_with_Averages.csv
-rw-r--r--  1 stanleytan  staff   1592 Jul  4  2024  README.md
drwxr-xr-x  6 stanleytan  staff    192 Jul  5  2024  scrape_data/
drwxr-xr-x  6 stanleytan  staff    192 Jul  4  2024  Images/
drwxr-xr-x  5 stanleytan  staff    160 Sep 15 12:41 __pycache__/
```

## The pipeline in build order

### 1. `nfl_box_score_analysis.py` — the foundation (Python toolkit)

Everything starts here. Pulls schedule/results and play-by-play straight from
nflverse GitHub releases (no scraping, no API key). Defines the **builder
pattern** every later file reuses: one function per stat, each returning one
row per `game_id` + `posteam`.

```python
from nfl_box_score_analysis import load_pbp, first_downs_by_team_game

pbp = load_pbp(2026)            # one season of play-by-play
fd = first_downs_by_team_game(pbp)   # game_id, posteam, first_downs
```

Builders: `first_downs_by_team_game`, `plays_yards_by_team_game`,
`redzone_by_team_game`, `team_rushing_by_team_game`,
`qb_hits_sacks_by_team_game`, `turnover_margin_by_team_game`,
plus `build_long_results` (schedule → one row per team per game with W/L).

CLI commands (`boxscore`, `list-games`, `*-trend`) compare winners vs losers
for any stat across seasons:

```bash
python nfl_box_score_analysis.py boxscore --game_id 2025_02_PHI_KC
python nfl_box_score_analysis.py turnover-trend --seasons 2021 2022 2023 2024 2025
```

### 2. `streamlit_scoreboard.py` — the app (built on the toolkit)

```bash
streamlit run streamlit_scoreboard.py
```

The scoreboard UI: pick a season/week (or a team), scroll every game with
final score + the box-score stats underneath. Its `season_game_stats()`
merges the toolkit's builders into one row per team-game — the same merge
shape `team_composite.py` later copied. Sidebar has a **Download this view
(CSV)** button for exactly the games on screen.

### 3. `team_composite.py` — NEW: Game Performance Score (Sep 16)

Answers: *"is there one # where first downs is better, yds/play is better,
rush yds/att is better, RZ TD is better, and QB hits/sacks is less?"*

Seven components, each **percentile-ranked across the 32 teams** (0–100 =
"better than X% of the league"), then weighted:

```python
WEIGHTS = {
    "win_pct": 0.20,          # higher = better
    "to_margin_pg": 0.15,     # higher = better (giveaways already subtracted)
    "first_downs_pg": 0.13,   # higher = better
    "yards_per_play": 0.13,   # higher = better
    "rush_yds_per_att": 0.13, # higher = better
    "rz_td_pct": 0.13,        # higher = better
    "pressure_rate": 0.13,    # LOWER = better -> flipped (qb_hit|sack)/dropbacks
}
```

Validated against real results: **0.66 correlation with scoring margin,
winners avg 61.6 vs losers 40.5** (adding win% then turnovers moved it from
0.56 → 0.66). Edit `WEIGHTS` and rerun to tune.

```bash
# team table only
python team_composite.py --seasons 2026

# also join the score onto a QB CSV (needs a `team` column)
python team_composite.py --seasons 2026 --qb_csv ~/Downloads/qb_adot_catchpct.csv
```

### 4. `game_performance_score.csv` — generated output

One row per team: `team, games, game_performance_score`, all raw components
(first downs/gm, yds/play, rush yds/att, RZ TD%, pressure rate, win%, TO
margin/gm) plus each component's league percentile. Regenerate, don't edit.

### 5. `qb_support_chart.py` — NEW: the chart (Sep 16)

The aDOT vs catch% QB scatter, recolored by Game Performance Score (green =
good team game, red = bad). Labels include record (`G.Smith 1-0 (66)`) and
are placed with **force-directed repulsion** (adjustText) so they never
overlap — moved labels get a thin connector line back to their dot. Also
writes a tidy `*_data.csv` of exactly what the chart plots.

```bash
# needs the `test` conda env on this machine (base matplotlib is broken vs numpy 2.x)
/Users/stanleytan/anaconda3/envs/test/bin/python qb_support_chart.py \
    --csv ~/Downloads/qb_adot_catchpct_with_game_performance.csv \
    --out ~/Downloads/qb_adot_catchpct_game_performance.png
```

**Weekly refresh = two commands: rerun `team_composite.py --qb_csv ...`, then**
**`qb_support_chart.py`.**

```python
python team_composite.py --seasons 2026 --qb_csv ~/Downloads/qb_adot_catchpct.csv   
# base python, pandas only
/Users/stanleytan/anaconda3/envs/test/bin/python qb_support_chart.py                
# test env, matplotlib

```



### 6. `STATS_REFERENCE.md` — the docs

How every stat is built (the builder pattern, verified numbers against the
`2025_02_PHI_KC` reference game), including the full Game Performance Score
section. Read this before adding any new stat.

### 7. Legacy / exploration files (2024)

- `2022_NFL_Season_Scores.csv`, `2023_NFL_Season_Scores.csv` — full-season
  game scores from the pre-nflverse era
- `top_quarterbackers_2022.csv`, `2023_Top_Pass_Yards_Players_with_Averages.csv`
  — early QB stat pulls
- `Untitled.ipynb` — original exploration notebook
- `scrape_data/`, `Images/`, `README.md` — early scraping work and assets
- `__pycache__/`, `.ipynb_checkpoints/` — generated, ignore

## Recap of the Sep 16 session (where the score came from)

Starting from the question — *"is there a # that combines first downs,
yds/play, rush yds/att, RZ TD, QB hits/sacks?"* — we built:

1. `team_composite.py` — the Game Performance Score (7 components,
   percentile-ranked, weighted 20/15/13×5). Validated: 0.66 correlation
   with scoring margin, winners 61.6 vs losers 40.5.
2. Two iterations from feedback: added win% (dropped B.Young 82.4→71.3 after
   CAR's 0-1 start) and turnover margin (validation jumped 0.54→0.66).
3. `qb_support_chart.py` — aDOT/catch% chart recolored by the score, records
   in labels, force-directed non-overlapping labels, tidy `_data.csv` export.
4. Renamed Team Support Score → Game Performance Score everywhere;
   documented in `STATS_REFERENCE.md`.

## What you can reproduce

- **Any season 2021–2026** — nflverse publishes play-by-play for all of
  them; just change `--seasons`. Brady's final year (2022, TB) works.
- **Any QB CSV with a `team` column** — the composite joins onto it. The
  chart additionally needs `passer_player_name`, `adot`, `catch_pct`,
  `record_str` — all buildable from the same nflverse data.
- **Multi-season baselines** — `--seasons 2021 2022 2023` blends years (good
  for stable priors).

## Prompt for next session (Brady 2022)

> In `~/Documents/technical/github/nfl_data`:
> 1. Build a 2022 QB CSV matching my `qb_adot_catchpct.csv` format
>    (passer_player_name, team, dropbacks, adot = mean air_yards per attempt,
>    completions, catch_pct, plus team record columns) from nflverse 2022
>    play-by-play using `load_pbp(2022)` from `nfl_box_score_analysis.py`,
>    QBs with 100+ attempts, save to `~/Downloads/qb_adot_catchpct_2022.csv`.
> 2. Run `python team_composite.py --seasons 2022 --qb_csv ~/Downloads/qb_adot_catchpct_2022.csv`
> 3. Chart it: `/Users/stanleytan/anaconda3/envs/test/bin/python qb_support_chart.py --csv ~/Downloads/qb_adot_catchpct_2022_with_game_performance.csv --out ~/Downloads/qb_2022_game_performance.png`
>    (chart needs the `test` conda env — base matplotlib is broken)
>
> I want to see where Tom Brady and the 2022 Bucs land.

Swap `2022` → `2025` (or any year) in steps 1–2 for other seasons. The
broken-env detail is saved to memory, so a future session will know to use
the `test` env for charts without re-explaining.



