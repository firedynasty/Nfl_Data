# Picks review log

Newest week first. Fill in the result and P&L each Monday or Tuesday.

---

## Week 3, 2026: picks

Chosen 2026-09-22 using [checklist.md](checklist.md). Yahoo scraped 2026-09-22, and the
model was logged (srs-v1) before Thursday's kickoff.

Planned: $20 each. Bought: 2026-09-22 on Kalshi. "Cost/contract" is
Bought ÷ To Pay, which includes Kalshi's fee.

| Pick | Stake | To Pay | Fair (Yahoo) | Model | Max | Odds shown | Cost/contract | Result | P&L |
|---|---|---|---|---|---|---|---|---|---|
| BUF vs LAC | $9.44 | $12.37 | 75.2% | 75% | 76¢ | 75% | 76.3¢ | | |
| SF vs ARI | $19.99 | $24.94 | 78.5% | 86% | 79¢ | 79% | 80.2¢ | | |
| DET vs NYJ | $39.99 | $53.08 | 73.2% | 81% | 74¢ | 74% | 75.3¢ | | |
| **Total** | **$69.42** | | | | | | | | |

After fees, all three are 1.1-2.1 points over fair, for an expected cost of about -$1.70. The stakes aren't equal:
DET is about 4× BUF. All three winning pays +$20.97. A DET loss alone is -$39.99.

**Why:** in all three, the market favorite and the model agree, the gap is under 10 points, and both starting QBs are normal.
SF's opponent ARI is starting Brissett, their backup.

**Passed on:**
- NO (61%) and LA (56%): they pass every check, but they're close to coin flips.
- KC (85%): at 85¢ the payout is too small.
- PHI @ CHI: CHI is starting backup QB Bagent, and the model has CHI 88% vs the market's PHI 65%.
- SEA (backup QB Drew Lock), MIN, BAL and CAR: the model is 15-29 points above the market.
- JAX/NE, HOU/IND and CIN/PIT: the model and the market pick opposite teams.

**Prices seen:** DraftKings was 2-4.5¢ over fair (SF 83, BUF 77, DET 75, NO 64,
LA 59), so we skipped it. Kalshi was at fair or 1¢ over, so we're buying there.

**Plan:** hold to settlement. Sell before kickoff only if a starting QB is ruled out.

**Expectation:** about 2 of 3 wins on average. There's only about a 43% chance that all three win.

---

## Week 2, 2026: review

| Pick | Paid | Price | Result | P&L |
|---|---|---|---|---|
| BUF over DET | $19.44 | 70% | Won 41-31 | +$7.56 |
| DAL over WAS | $19.32 | 67% | Won 37-20 | +$8.68 |
| SEA at ARI | $24.15 | 67% | Won 31-7 | +$10.85 |
| NE over PIT | $19.88 | 69% | Won 20-3 | +$8.12 |
| CHI over MIN | $19.60 | 68% | Lost 3-9 | -$19.60 |
| JAX at DEN | $19.80 | 43% | Lost 13-20 | -$19.80 |
| NYJ vs GB | $19.68 | sold early | Lost 17-20 | -$3.36 |
| **Total** | **$141.87** | | | **-$7.55** |

**What went right**
- Went 4-for-5 on 67-70% favorites. About 3.4 wins were expected, so that's normal variance, not an edge.
- Selling NYJ early was right. They lost, and the sale got back $16.32 instead of $0.

**What to fix**
1. **Paid over fair on every favorite.** Counting the fee (Bought ÷ To Pay), the cost was
   3.5-4.3 points over Yahoo's fair price: CHI 70.0¢ vs 65.8, DAL 69.0¢ vs 64.7, SEA 69.0¢ vs 65.1,
   NE 71.0¢ vs 67.5.
2. **"Money is on them" was the main reason, but it was read backwards.** The big-money bettors
   were leaning against CHI (-15), DAL (-12) and NE (-8). JAX (+16) was the one
   pick they agreed with, and it lost. Following that signal went 22-26 for the week.
3. **Trusted the model over the market on JAX.** The model said 64% and the market said 43%.
   The backtest says large model-vs-market gaps favor the market.
4. **QB changes were missed.** SEA started Drew Lock and MIN started Carson Wentz.
5. **Selling mid-week costs about 5%** (bought at about 69¢, could sell at about 64¢).

**Model (grade_predictions.py):** 53.3% right on winners and 53.3% against the spread over 15 games.
On 4+ point disagreements with the spread it went 25% (n=4). That's too small a sample to judge.
