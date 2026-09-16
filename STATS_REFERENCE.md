# Stats reference

How each stat in `nfl_box_score_analysis.py` / `streamlit_scoreboard.py` is
built, so a new one can follow the same pattern.

## The pattern

Every stat is a **builder function** in `nfl_box_score_analysis.py` that takes
a season's play-by-play (`pbp`) and returns one row per `game_id` + `posteam`
(the team that had the ball on that play). To wire a new one in:

1. Write the builder in `nfl_box_score_analysis.py`, grouped by
   `["game_id", "posteam"]`.
2. Merge it into `season_game_stats()` in `streamlit_scoreboard.py`
   (`left_on=["game_id","team"], right_on=["game_id","posteam"]`, then drop
   `posteam`).
3. Add a formatted column to `stats_table()` (the per-game display table).
4. Sanity-check it against one real game before trusting it, e.g.:
   ```
   python3 -c "
   from nfl_box_score_analysis import load_pbp, my_new_builder
   pbp = load_pbp(2025)
   g = pbp[pbp['game_id']=='2025_02_PHI_KC']
   print(my_new_builder(g))
   "
   ```
   (`2025_02_PHI_KC`, PHI 20–17 KC, is the reference game used throughout —
   most numbers below were checked against it.)

## Stats added

### Plays / Yards / Yards per play (efficiency)
- **Builder:** `plays_yards_by_team_game(pbp)`
- **Logic:** filter to `play_type.isin(['run', 'pass'])` — this is the
  standard "offensive plays" definition; it excludes penalties-only
  (`no_play`), kneels, spikes, punts, kicks, and FG/XP attempts.
  `yards_per_play = yards / plays`.
- **CLI:** `efficiency-trend` (winners vs. losers). Verified against 2021–2025:
  winners averaged 62.6 plays / 365.9 yds / 5.86 ypp, losers 61.2 / 308.7 / 5.03.
- **UI columns:** `Plays`, `Yards`, `Yds/Play`.

```python
import pandas as pd

games = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv')

seasons = [2021, 2022, 2023, 2024, 2025, 2026]
frames = []
for s in seasons:
    url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.csv.gz'
    frames.append(pd.read_csv(url, compression='gzip', low_memory=False))
pbp = pd.concat(frames, ignore_index=True)

# Yards per play = total yards on run/pass plays / number of run/pass plays
# (excludes penalties-only plays, kneels, spikes, etc.)
scrim = pbp[pbp['play_type'].isin(['run', 'pass'])]
ypp = scrim.groupby(['game_id', 'posteam']).agg(
    plays=('play_type', 'count'),
    yards=('yards_gained', 'sum')
).reset_index()
ypp['yards_per_play'] = ypp['yards'] / ypp['plays']

# Reshape games.csv into one row per team per game with a W/L result
g = games.dropna(subset=['home_score', 'away_score']).copy()
g['season'] = g['season'].astype(int)
g = g[g['season'].isin(seasons)]
cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']

home = g[cols].copy()
home['team'], home['opp'] = home['home_team'], home['away_team']
home['team_score'], home['opp_score'] = home['home_score'], home['away_score']

away = g[cols].copy()
away['team'], away['opp'] = away['away_team'], away['home_team']
away['team_score'], away['opp_score'] = away['away_score'], away['home_score']

long = pd.concat([home, away], ignore_index=True)[
    ['game_id', 'season', 'week', 'team', 'opp', 'team_score', 'opp_score']
]
long['result'] = long.apply(
    lambda r: 'W' if r.team_score > r.opp_score else ('L' if r.team_score < r.opp_score else 'T'),
    axis=1
)

# Merge and compare
merged = long.merge(ypp, left_on=['game_id', 'team'], right_on=['game_id', 'posteam'], how='left')
merged = merged[merged['result'].isin(['W', 'L'])]

print(merged.groupby('result')[['plays', 'yards', 'yards_per_play']].agg(['mean', 'median']).round(2))
```



### First downs
- **Builder:** `first_downs_by_team_game(pbp)`
- **Logic:** `first_down == 1`, grouped and counted.
- **CLI:** `first-downs-trend`.
- **UI column:** `1st`.
- ```python
  import pandas as pd
  
  game_id = "2025_02_PHI_KC"   # swap in your game
  season = int(game_id.split("_")[0])
  
  pbp = pd.read_csv(
      f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
      compression='gzip', low_memory=False
  )
  g_pbp = pbp[pbp['game_id'] == game_id]
  
  first_downs = (
      g_pbp[g_pbp['first_down'] == 1]
      .groupby('posteam')
      .size()
      .reset_index(name='first_downs')
  )
  
  print(first_downs)
  ```
- 

### Turnover margin
- **Builder:** `turnover_margin_by_team_game(pbp, long)`
- **Logic:** `turnover = interception + fumble_lost` per play, summed per
  team-game as "giveaways." Merged onto `long` (the reshaped one-row-per-
  team-per-game schedule) twice — once for the team's own giveaways, once for
  the opponent's (= the team's takeaways). `margin = takeaways - giveaways`.
- **CLI:** `turnover-trend` (also buckets margin into positive/even/negative
  and reports win rate per bucket).
- **UI column:** `TO's` (shown as a signed number, e.g. `+1`, `-2`).

### Red zone: trips, TD rate, score rate
- **Builder:** `redzone_by_team_game(pbp)`
- **Logic:** group plays into drives (`fixed_drive`), take the max of
  `drive_inside20` per drive to flag red-zone drives, and the drive's final
  `fixed_drive_result`. A trip's `td` = result is "Touchdown"; `scored` =
  result is "Touchdown" or "Field goal". Aggregated to trips / td count /
  scored count per team-game.
- **CLI:** `redzone-trend`.
- **UI columns:** originally `Trips` / `TD%` / `Score%` (three columns), later
  collapsed to two fraction-style columns for space: **`RZ TD`** (`td/trips`,
  e.g. `2/2`) and **`RZ Score`** (`scored/trips`, e.g. `2/2`).

```python
import pandas as pd

game_id = "2025_02_PHI_KC"   # swap in your game
season = int(game_id.split("_")[0])

pbp = pd.read_csv(
    f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
    compression='gzip', low_memory=False
)
g_pbp = pbp[pbp['game_id'] == game_id]

drives = g_pbp.dropna(subset=['posteam', 'fixed_drive']).groupby(
    ['posteam', 'fixed_drive']
).agg(inside20=('drive_inside20', 'max'), result=('fixed_drive_result', 'first')).reset_index()

rz = drives[drives['inside20'] == 1].copy()
rz['td'] = (rz['result'] == 'Touchdown').astype(int)
rz['scored'] = rz['result'].isin(['Touchdown', 'Field goal']).astype(int)

summary = rz.groupby('posteam').agg(
    trips=('td', 'count'),
    td=('td', 'sum'),
    scored=('scored', 'sum'),
).reset_index()

# Fold trips into each ratio -> 2 columns instead of 3
summary['RZ TD'] = summary['td'].astype(str) + '/' + summary['trips'].astype(str) + ' (' + (summary['td']/summary['trips']*100).round(0).astype(int).astype(str) + '%)'
summary['RZ Score'] = summary['scored'].astype(str) + '/' + summary['trips'].astype(str) + ' (' + (summary['scored']/summary['trips']*100).round(0).astype(int).astype(str) + '%)'

print(summary[['posteam', 'RZ TD', 'RZ Score']])


```

redo without percentage 



```python

import pandas as pd

game_id = "2025_02_PHI_KC"   # swap in your game
season = int(game_id.split("_")[0])

pbp = pd.read_csv(
    f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
    compression='gzip', low_memory=False
)
g_pbp = pbp[pbp['game_id'] == game_id]

drives = g_pbp.dropna(subset=['posteam', 'fixed_drive']).groupby(
    ['posteam', 'fixed_drive']
).agg(inside20=('drive_inside20', 'max'), result=('fixed_drive_result', 'first')).reset_index()

rz = drives[drives['inside20'] == 1].copy()
rz['td'] = (rz['result'] == 'Touchdown').astype(int)
rz['scored'] = rz['result'].isin(['Touchdown', 'Field goal']).astype(int)

summary = rz.groupby('posteam').agg(
    trips=('td', 'count'),
    td=('td', 'sum'),
    scored=('scored', 'sum'),
).reset_index()

summary['RZ TD'] = summary['td'].astype(str) + '/' + summary['trips'].astype(str)
summary['RZ Score'] = summary['scored'].astype(str) + '/' + summary['trips'].astype(str)

print(summary[['posteam', 'RZ TD', 'RZ Score']])


```



### Team rushing (whole team, not just the lead back)
- **Builder:** `team_rushing_by_team_game(pbp)`
- **Logic:** `rush_attempt == 1`, summed per team-game for `rush_attempts` and
  `rush_yards`; `yds_per_attempt = rush_yards / rush_attempts`.
- **Distinct from:** `main_rusher_by_team_game(pbp, min_attempts=10)`, an
  earlier builder that isolates just the *lead* rusher per team-game (used by
  the CLI's `rushing-trend` command to study "does a workhorse back with a
  good day correlate with winning"). Team rushing is the whole backfield;
  main-rusher is one player.
- **UI columns:** `Rush #`, `Yds` (renamed from `Rush Yds` for space),
  `Yds/Att`.

```python

import pandas as pd

game_id = "2025_02_PHI_KC"   # swap in whatever game you're building the box score for
season = int(game_id.split("_")[0])

pbp = pd.read_csv(
    f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
    compression='gzip', low_memory=False
)
g_pbp = pbp[pbp['game_id'] == game_id]

# Team-level rushing: every rush attempt credited to that team, not just the lead back
rush = g_pbp[g_pbp['rush_attempt'] == 1]
team_rushing = rush.groupby('posteam').agg(
    rush_attempts=('rush_attempt', 'sum'),
    rush_yards=('rushing_yards', 'sum'),
).reset_index()
team_rushing['yds_per_attempt'] = (team_rushing['rush_yards'] / team_rushing['rush_attempts']).round(1)

print(team_rushing)


```



### QB hits & sacks
- **Builder:** `qb_hits_sacks_by_team_game(pbp)`
- **Logic:** filter to `qb_dropback == 1`, sum `qb_hit` and `sack` per
  team-game. The `team` side here is the team that dropped back to pass, so
  the numbers represent pressure taken *against* that team's QB (i.e.
  pressure applied by the opponent's defense).
- **Caveat (kept as a code comment and shown in the app sidebar):** hits
  usually include sacks in nflverse's charting, but not with 100%
  consistency — so the two are reported as separate columns rather than
  assuming one strictly contains the other.
- **UI columns:** `QB Hits`, `Sacks`.

```python
import pandas as pd

game_id = "2025_02_PHI_KC"   # swap in your game
season = int(game_id.split("_")[0])

pbp = pd.read_csv(
    f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
    compression='gzip', low_memory=False
)
g_pbp = pbp[pbp['game_id'] == game_id]

# Dropbacks only (pass attempts + sacks + scrambles)
db = g_pbp[g_pbp['qb_dropback'] == 1].copy()
db['pressure_proxy'] = ((db['qb_hit'] == 1) | (db['sack'] == 1)).astype(int)

pressure_allowed = (
    db.groupby('posteam')
    .agg(dropbacks=('pressure_proxy', 'count'), pressured=('pressure_proxy', 'sum'))
    .reset_index()
)
pressure_allowed['pressure_allowed_rate'] = (
    pressure_allowed['pressured'] / pressure_allowed['dropbacks'] * 100
).round(1)

print(pressure_allowed)
```

redo qb hits

```python

import pandas as pd

game_id = "2025_02_PHI_KC"   # swap in your game
season = int(game_id.split("_")[0])

pbp = pd.read_csv(
    f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
    compression='gzip', low_memory=False
)
g_pbp = pbp[pbp['game_id'] == game_id]

db = g_pbp[g_pbp['qb_dropback'] == 1].copy()
db['pressure_proxy'] = ((db['qb_hit'] == 1) | (db['sack'] == 1)).astype(int)

pressure_allowed = (
    db.groupby('posteam')
    .agg(dropbacks=('pressure_proxy', 'count'), pressured=('pressure_proxy', 'sum'))
    .reset_index()
)
pressure_allowed['Pressured'] = (
    pressure_allowed['pressured'].astype(str) + '/' + pressure_allowed['dropbacks'].astype(str)
)

print(pressure_allowed[['posteam', 'Pressured']])


```

qb hits in two columns

```python
import pandas as pd

game_id = "2025_02_PHI_KC"   # swap in your game
season = int(game_id.split("_")[0])

pbp = pd.read_csv(
    f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
    compression='gzip', low_memory=False
)
g_pbp = pbp[pbp['game_id'] == game_id]

db = g_pbp[g_pbp['qb_dropback'] == 1].copy()

hits_sacks = db.groupby('posteam').agg(
    qb_hits=('qb_hit', 'sum'),
    sacks=('sack', 'sum'),
).reset_index()

print(hits_sacks)
```



### Game Performance Score (composite)
- **Script:** `team_composite.py` (standalone; reuses the builders above)
- **Logic:** aggregates seven components to season totals per team —
  win percentage (ties = half a win), turnover margin/game (takeaways −
  giveaways, from `turnover_margin_by_team_game`; higher = better, no flip
  needed since giveaways are already subtracted inside the margin),
  first downs/game, yards per play, rush yards/attempt, RZ TD rate, and
  pressure allowed rate (`(qb_hit | sack) / dropbacks`, from the dropback
  side, so it's pressure the QB's own line gave up). Each component is
  **percentile-ranked across the 32 teams** (0–100, "better than X% of the
  league"), the pressure component is flipped (less = better), and the
  weighted average is the **Game Performance Score** (0–100). Percentiles
  instead of z-scores so one outlier team can't skew the scale. Weights
  are an editable `WEIGHTS` dict at the top of the script (default: win%
  .20, turnover margin .15, the five remaining stat components .13 each).
- **Validation caveat:** the game-level validation score excludes win% —
  at single-game level win% *is* the result, so including it would make
  the winners-vs-losers check circular. Only the stat components are
  validated there (renormalized to sum to 1).
- **NaN handling:** a team with zero red-zone trips gets the league-average
  RZ TD rate (neutral), so early-season samples don't produce NaN scores.
- **CLI:** `python team_composite.py --seasons 2026`
  Add `--qb_csv <path>` to join the score onto a per-QB CSV (on `team`) —
  built for attaching to QB-rating-style rows like `qb_adot_catchpct.csv`.
- **Validation:** the script also builds a game-level version of the score
  and reports its correlation with scoring margin plus winner/loser means
  (2026 wk1: corr 0.56, wins 59.0 vs losses 42.9).

## UI-only naming changes (no logic change)

Several columns were renamed purely for table width once first downs, QB
hits, and sacks pushed the table past what's comfortably visible:

| Original label     | Shortened to |
|---------------------|--------------|
| Turnover Margin      | TO's         |
| Rush Att             | Rush #       |
| Rush Yds             | Yds          |
| RZ Trips             | (merged into RZ TD / RZ Score fractions) |
| RZ TD% / RZ Score%   | RZ TD / RZ Score |
| 1st Downs            | 1st          |

## Exporting

`streamlit_scoreboard.py`'s sidebar has a **"Download this view (CSV)"**
button. It exports `season_game_stats(season)` filtered down to whichever
games are currently on screen (one week, or one team's full season) — one row
per team per game, every column above plus `team_name`, `opp_name`,
`team_score`, `opp_score`, and `result` (W/L/T). That's the shape to load into
pandas for a `groupby(...).mean()` — e.g. average QB hits allowed by team.
