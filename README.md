# nfl_data

SRS (Simple Rating System) pipelines for football, one per sport, sharing
the same math. Both turn final scores into opponent-adjusted team ratings
in points, predict upcoming games with no look-ahead leakage, and log
pre-kickoff predictions that the `markets` repo's `edge.py` reads as its
Model signal.

## Layout

```
shared/   sport-agnostic math: srs.py (ratings + long-results reshape),
          backtest_common.py (walk-forward backtest machinery)
nfl/      NFL pipeline: nflverse data -> predict_week.py -> predictions_log.csv
          (plus the streamlit app, backtests, systems/, and legacy exploration)
ncaa/     College (FBS) pipeline: ESPN scoreboard -> predict_cfb_week.py
          -> cfb_predictions_log.csv (cache: cfb_games_cache.csv)
```

Run scripts from inside their sport folder (relative cache/log paths):

```bash
cd nfl  && python predict_week.py --log          # this week's NFL sheet + log
cd ncaa && python predict_cfb_week.py --log      # this week's college sheet + log
cd ncaa && python predict_cfb_week.py --team ND  # one matchup
cd nfl  && streamlit run streamlit_scoreboard.py # the app
```

Both predict scripts default to the week in progress (or the next one once
the last is fully final). Same honesty rule everywhere: the backtests say
the models rank teams well but don't beat the market, so the model line is
a veto check, not a bet signal.

See `FILES.md` for the full file-by-file guide and `nfl/STATS_REFERENCE.md`
for how every NFL stat is built.
