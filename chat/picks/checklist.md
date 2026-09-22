# Weekly picks checklist

Goal: back the team most likely to win, and only at a price at or below
its fair probability. The market (Yahoo moneyline, devigged) is the most
reliable win probability we have: 2021-2025 it picked 66.5% of winners
(Brier 0.211) vs SRS 63.2% (0.229) and the SRS+QB blend 64.1% (0.224).
The model and QB data are a veto check, not the picker.

## Tue-Thu: before buying

1. **QB data.** Refresh the QB chart:
   ```bash
   python team_composite.py --seasons 2026 --qb_csv ...
   /Users/stanleytan/anaconda3/envs/test/bin/python qb_support_chart.py --csv ... --out ...
   ```
   Confirm the **starting QB for both teams**. A backup starting is a reason to skip.
2. **Streamlit.** Run `streamlit run streamlit_scoreboard.py` and look at SRS, edge and Game Performance Score.
3. **Log the model's picks before the first kickoff** (Thursday):
   `python predict_week.py --season 2026 --week N --log`
4. **Scrape the Yahoo lines:** `python3 systems/public_money/public_money_system.py --week N`
5. **Get the fair %** by devigging each moneyline pair (`yahoo_ml_prob` in
   `kalshi_advisor.py --dry-run`). Each side's implied prob ÷ the sum of both sides.
6. **Choose picks.** Pick the market favorite only when:
   - the model picks the same team, and
   - the model and the market are **within about 10 points** of each other, and
   - both starting QBs are normal starters.

   A 15+ point gap between the model and the market means **skip**. The market is usually right there (for example, JAX in Week 2).
7. **Price check.** Buy only when the **actual cost is at fair or 1¢ over**.
   On Kalshi, the actual cost is **Bought ÷ To Pay**, which includes the fee. It runs about 1-2¢
   above the "Odds" column. Aim for a displayed price about 1-2¢ **below** fair. Compare Kalshi and
   DraftKings and use the cheaper one. Try a limit order first.
   If every price is over fair, passing on the week is fine.
8. **Same stake every pick** (about $20). For smaller swings, bet less per game instead of selling early.
9. **Write the one-line "why" before confirming**, and name a signal from steps 1-6.
   Example: "BUF: market 75%, model 75%, Allen healthy, bought 75¢."
   "Money is on them" doesn't count on its own.

## Holding: when to sell

10. **Default: hold to settlement.** No auto-sell at +X%. At a fair price,
    selling and holding come out about even on average, and the sell price
    is usually a few cents below the real value (Week 2: bought at 69¢, could sell at 64¢).
    A price rising during the game is just the score, not free profit.
11. **Sell before kickoff only if:**
    - your **starting QB is ruled out** or there's other major news (cut the loss), or
    - the price jumps well above fair **with no news** (sell into the overpricing).
12. Check injury news once on Sunday morning.

## Mon-Tue: after the games

13. `python grade_predictions.py`
14. `python3 systems/public_money/public_money_system.py --results --week N`
15. Paste the settled Kalshi table into `my_positions.txt`, then run
    `python3 systems/kalshi_advisor/kalshi_advisor.py --file my_positions.txt`
    to grade your notes.
16. Record the **full week's profit or loss** in `chat/picks/review.md`, including losses and early sells.
