"""
Kalshi advisor -- reconcile a Kalshi NFL position slate against four
independent signals (Kalshi's own live price, Yahoo's moneyline-implied
probability, Yahoo's public bet%/money% split, and this repo's SRS model),
then ask an LLM to write a candid hold/sell/pass verdict per position.

THE HONESTY RULE (same discipline as predict_week.py and
public_money_system.py): this repo's own honest walk-forward backtest
(2021-2025, see intermediate/qa_log.md) found the SRS model does NOT clear
the 52.4% moneyline/spread breakeven, and the bet%/money% split has no free
historical source so it CAN'T be backtested here either (see
public_money_system.py's own docstring). So model_win_prob and the
"sharp money" read are both noisy opinions, not a proven edge -- the prompt
sent to the LLM says this explicitly, and tells it to weight the two real
markets (Kalshi, Yahoo ML) over the model, using a big model/market gap or
a lopsided public split as a "double-check this" flag rather than a reason
to trade against consensus.

INPUTS
------
1. A pasted Kalshi "My Positions" table (see PASTE FORMAT below), from a
   file (--file) or stdin.
2. predictions_log.csv -- this repo's own SRS win-probability log, written
   by `predict_week.py --log`. Most recent logged row per team is used.
3. public_money_log.csv -- Yahoo moneyline snapshot, written by
   `systems/public_money/public_money_system.py`. Most recent scraped
   "Money Line" pair per matchup is devigged into a fair probability, and
   its bets_pct/money_pct columns are carried through as-is (money% well
   below bets% on a side = the "fade the public" tell).
4. game_performance_score.csv -- team_composite.py's Game Performance
   Score, joined in as descriptive context only (it's a percentile blend
   of box-score stats, not a probability).

Any of 2-4 that's missing or stale is fine -- the advisor just tells the
LLM what it does and doesn't have for that position.

PASTE FORMAT (copy straight from the Kalshi positions page)
-------------------------------------------------------------
Matchup    Odds    Status    Bought    To Pay / Paid    Sell Price    Game Time    Placed
CHI Bears (vs MIN)    68%    Open    $19.60    To Pay: $28.00    Sell $18.20    Sun 10:00 AM    Sep 18, 2026, 10:25:00 PM

Columns just need 2+ spaces or a tab between them (however Kalshi's page
copies out). Rows that don't parse are skipped with a warning, not fatal.

USAGE
-----
  export OPENAI_API_KEY=sk-...
  python3 systems/kalshi_advisor/kalshi_advisor.py --file my_positions.txt
  pbpaste | python3 systems/kalshi_advisor/kalshi_advisor.py         # stdin
  python3 systems/kalshi_advisor/kalshi_advisor.py --file bets.txt --dry-run
      # no OpenAI call -- just the merged kalshi/model/yahoo odds table
  python3 systems/kalshi_advisor/kalshi_advisor.py --file bets.txt --model gpt-4o
  python3 systems/kalshi_advisor/kalshi_advisor.py --file bets.txt --no-log

This is a decision aid, not a signal with a proven edge -- see the honesty
rule above. It logs every run to kalshi_advisor_log.csv (append-only, pass
--no-log to skip) so you can look back at what it said vs what happened.

INTUITION JOURNAL
------------------
The first time it sees an OPEN position, it prompts you (on the terminal,
not stdin, so `pbpaste | ...` still works) for why you like it, and saves
that note to kalshi_advisor_notes.csv keyed to the position. Leave it blank
to skip. Once the position resolves (SOLD/WON/LOST) and you re-run the
advisor on the same slate, the saved note is handed back to the LLM, which
grades it GOOD/MIXED/BAD against the signals you actually had at entry
time -- not just against whether you happened to win. Pass --no-notes to
turn this off entirely.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PREDICTIONS_LOG = os.path.join(REPO_ROOT, "predictions_log.csv")
PUBLIC_MONEY_LOG = os.path.join(REPO_ROOT, "public_money_log.csv")
GAME_PERFORMANCE_CSV = os.path.join(REPO_ROOT, "game_performance_score.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "kalshi_advisor_log.csv")
NOTES_PATH = os.path.join(os.path.dirname(__file__), "kalshi_advisor_notes.csv")

BREAKEVEN_NOTE = (
    "This repo's own honest walk-forward backtest (2021-2025) found the SRS "
    "model does not clear the 52.4% moneyline/spread breakeven -- treat "
    "model_win_prob as one noisy opinion, not a proven edge. A large gap "
    "between model_win_prob and the two market prices is a reason to look "
    "closer (injury news, a QB change, a bad matchup for the model's "
    "inputs), not a reason to trade against two independent markets that "
    "agree with each other."
)

NICKNAME_TO_ABBR = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}

# --- parsing the pasted Kalshi slate -------------------------------------

MATCHUP_RE = re.compile(r"^(.*?)\s*\(vs\.?\s*([A-Za-z]{2,3})\)\s*$")
MONEY_RE = re.compile(r"\$([\d,]+\.?\d*)")


def team_abbr_from_name(name):
    for nick, abbr in NICKNAME_TO_ABBR.items():
        if nick.lower() in name.lower():
            return abbr
    return None


def parse_slate(text):
    """Yields one dict per parseable position row. Skips the header row and
    anything without at least a matchup, an odds%%, and a status."""
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("matchup"):
            continue
        fields = [f.strip() for f in re.split(r"\t|\s{2,}", line) if f.strip()]
        if len(fields) < 3:
            print(f"warning: skipping unparseable line: {raw_line!r}", file=sys.stderr)
            continue

        matchup_raw, odds_raw, status = fields[0], fields[1], fields[2]
        m = MATCHUP_RE.match(matchup_raw)
        team_name, opp_abbr = (m.group(1), m.group(2).upper()) if m else (matchup_raw, None)
        team_abbr = team_abbr_from_name(team_name)
        odds_match = re.match(r"(\d+(?:\.\d+)?)%", odds_raw)
        if team_abbr is None or odds_match is None:
            print(f"warning: skipping unrecognized line: {raw_line!r}", file=sys.stderr)
            continue
        kalshi_prob = float(odds_match.group(1)) / 100.0

        def money(idx):
            if idx >= len(fields):
                return None
            mm = MONEY_RE.search(fields[idx])
            return float(mm.group(1).replace(",", "")) if mm else None

        bought = money(3)
        payout_field = fields[4].lower() if len(fields) > 4 else ""
        to_pay = money(4) if "to pay" in payout_field else None
        paid = money(4) if "paid" in payout_field else None
        sell_price = money(5) if len(fields) > 5 else None
        game_time = fields[6] if len(fields) > 6 else None
        placed = fields[7] if len(fields) > 7 else None

        rows.append({
            "team_abbr": team_abbr, "opp_abbr": opp_abbr, "matchup_raw": matchup_raw,
            "kalshi_prob": kalshi_prob, "status": status,
            "bought": bought, "to_pay": to_pay, "paid": paid, "sell_price": sell_price,
            "game_time": game_time, "placed": placed,
        })
    return rows


# --- "why did you like this" journal -----------------------------------
#
# The idea: when a position is still OPEN, ask the trader (once, at a
# terminal, not via stdin so piping a pasted slate still works) why they
# like it, and stash that note keyed to the position. Next time this script
# runs against the *same* position after it's resolved (SOLD/WON/LOST), the
# note gets handed back to the LLM so it can give a candid "was this good
# intuition, in hindsight" read instead of just a bare settled/unsettled
# verdict.

def note_key(bet):
    """(team, opp, placed) -- placed timestamp is set once when the bet is
    placed and doesn't change as the position moves from Open to
    Sold/Won/Lost, so it's a stable key across a position's lifecycle."""
    return (bet["team_abbr"], bet["opp_abbr"] or "", bet["placed"] or "")


def load_notes(path=NOTES_PATH):
    """key -> most recently saved note text."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return {}
    out = {}
    for _, r in df.iterrows():
        out[(r["team_abbr"], r["opp_abbr"], r["placed"])] = r["note"]
    return out


def save_note(key, matchup_raw, note, path=NOTES_PATH):
    team_abbr, opp_abbr, placed = key
    row = pd.DataFrame([{
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "team_abbr": team_abbr, "opp_abbr": opp_abbr, "placed": placed,
        "matchup_raw": matchup_raw, "note": note,
    }])
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def prompt_tty(prompt_text):
    """Prompts on the controlling terminal even when stdin was already
    consumed piping in the pasted slate. Returns None (no prompt asked,
    e.g. non-interactive/cron use) rather than raising if there's no tty."""
    try:
        with open("/dev/tty") as tty_in, open("/dev/tty", "w") as tty_out:
            tty_out.write(prompt_text)
            tty_out.flush()
            line = tty_in.readline()
    except OSError:
        return None
    return line.strip() or None


def collect_notes(bets, no_notes=False):
    """For each OPEN position with no saved note yet, ask why the trader
    likes it and save the answer. Returns {bet_index: note_text} for every
    bet that has a note (freshly asked or previously saved)."""
    notes = load_notes()
    result = {}
    for i, b in enumerate(bets):
        key = note_key(b)
        if key not in notes and not no_notes and b["status"].strip().lower() == "open":
            answer = prompt_tty(
                f"\nWhy do you like {b['matchup_raw']} ({b['kalshi_prob'] * 100:.0f}%)? "
                f"(Enter your reasoning, or blank to skip): "
            )
            if answer:
                save_note(key, b["matchup_raw"], answer)
                notes[key] = answer
        if key in notes and notes[key]:
            result[i] = notes[key]
    return result


# --- the three signals -----------------------------------------------

def load_model_probs():
    """team_abbr -> most recently logged SRS win probability, from
    predictions_log.csv (predict_week.py --log)."""
    if not os.path.exists(PREDICTIONS_LOG):
        return {}
    df = pd.read_csv(PREDICTIONS_LOG)
    if df.empty:
        return {}
    out = {}
    for _, r in df.sort_values("logged_at").iterrows():
        for team, opp, is_home in ((r["home"], r["away"], True), (r["away"], r["home"], False)):
            prob = r["home_win_prob"] if is_home else 1 - r["home_win_prob"]
            out[team] = {
                "model_win_prob": round(float(prob), 4),
                "season": int(r["season"]), "week": int(r["week"]),
                "opponent": opp, "logged_at": r["logged_at"],
            }
    return out


ML_SIDE_RE = re.compile(r"^([A-Z]{2,3})\s+([+-]\d+)$")


def american_to_prob(odds):
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def load_yahoo_signals():
    """team_abbr -> Yahoo Money Line snapshot from public_money_log.csv
    (public_money_system.py), most recently scraped pair per matchup:
    - yahoo_ml_prob / yahoo_odds: devigged fair probability + the raw
      American price it came from.
    - public_bets_pct / public_money_pct: Yahoo's "Full Game Public Bet %"
      table for this side -- % of tickets vs % of dollars on it.
    - public_sharp_diff: money_pct - bets_pct. Meaningfully negative means
      the ticket-heavy public is piling on this side while fewer, bigger
      (sharper) bets lean the other way -- the classic "fade the public"
      tell public_money_system.py is built around."""
    if not os.path.exists(PUBLIC_MONEY_LOG):
        return {}
    df = pd.read_csv(PUBLIC_MONEY_LOG)
    ml = df[df["market"] == "Money Line"].copy()
    if ml.empty:
        return {}
    latest = ml.groupby("matchup")["scraped_at"].transform("max")
    ml = ml[ml["scraped_at"] == latest]

    out = {}
    for matchup, grp in ml.groupby("matchup"):
        sides = []
        for _, r in grp.iterrows():
            sm = ML_SIDE_RE.match(str(r["side"]).strip())
            if sm:
                sides.append((sm.group(1), float(sm.group(2)), r))
        if len(sides) != 2:
            continue
        (t1, o1, r1), (t2, o2, r2) = sides
        p1_raw, p2_raw = american_to_prob(o1), american_to_prob(o2)
        total = p1_raw + p2_raw
        for team, odds, praw, r in ((t1, o1, p1_raw, r1), (t2, o2, p2_raw, r2)):
            bets_pct, money_pct = float(r["bets_pct"]), float(r["money_pct"])
            out[team] = {
                "yahoo_ml_prob": round(praw / total, 4), "yahoo_odds": odds, "matchup": matchup,
                "public_bets_pct": bets_pct, "public_money_pct": money_pct,
                "public_sharp_diff": round(money_pct - bets_pct, 2),
            }
    return out


def load_composite():
    """team_abbr -> team_composite.py's Game Performance Score + inputs,
    descriptive context only (not a probability)."""
    if not os.path.exists(GAME_PERFORMANCE_CSV):
        return {}
    df = pd.read_csv(GAME_PERFORMANCE_CSV)
    return {
        r["team"]: {
            "game_performance_score": round(float(r["game_performance_score"]), 1),
            "srs": round(float(r["srs"]), 2),
            "win_pct": round(float(r["win_pct"]), 2),
        }
        for _, r in df.iterrows()
    }


def build_context(bet, model_probs, yahoo_probs, composite, user_note=None):
    ctx = dict(bet)
    ctx["model"] = model_probs.get(bet["team_abbr"])
    ctx["yahoo_ml"] = yahoo_probs.get(bet["team_abbr"])
    ctx["composite"] = composite.get(bet["team_abbr"])
    if bet["opp_abbr"]:
        ctx["opp_model"] = model_probs.get(bet["opp_abbr"])
        ctx["opp_composite"] = composite.get(bet["opp_abbr"])
    ctx["user_note"] = user_note
    return ctx


# --- the LLM call -----------------------------------------------

SYSTEM_PROMPT = f"""You are a disciplined sports-market analyst helping a
single retail trader review their own open Kalshi NFL "will X win"
positions. You are not placing bets or giving financial advice -- you are
reconciling signals the trader already has and giving a candid read.

{BREAKEVEN_NOTE}

For each position you're given:
- kalshi_prob: the live Kalshi contract price (implied probability) -- the
  position's actual current market.
- model: this repo's own SRS power-rating win probability for the same
  team, if logged (may be missing/stale).
- yahoo_ml: Yahoo's sportsbook moneyline consensus, if scraped (may be
  missing/stale) -- yahoo_ml_prob is the devigged fair probability,
  yahoo_odds is the actual American price (e.g. -220, +125) that prob
  was computed from, and public_bets_pct/public_money_pct/public_sharp_diff
  are Yahoo's "Full Game Public Bet %" split for this side (diff =
  money% - bets%; meaningfully negative means the ticket-heavy public is
  piling on this side while fewer, bigger bets lean the other way -- the
  classic "fade the public" tell, but this repo has no free historical
  source for it so it's never been backtested -- treat it as a soft flag,
  same as the model).
- composite: team_composite.py's Game Performance Score (0-100 percentile
  blend of first downs/play efficiency/rush efficiency/red zone/pressure/
  turnovers/win%) for descriptive context, not a probability.
- the position's own economics (bought price, payout if it resolves yes,
  current sell price, status).
- user_note: the trader's OWN words, written at entry time, on why they
  liked this position (may be null if they didn't write one). This is
  their stated intuition, not a signal -- don't treat it as evidence, but
  DO engage with it directly in your reasoning.

Weight kalshi_prob and yahoo_ml_prob most heavily -- they are two
independent, efficient markets that usually agree with each other. Use
model, public_sharp_diff, and composite only as secondary sanity checks.
When the model disagrees sharply with both markets, or the public split is
lopsided against the side you're holding, call that out explicitly as
"double-check this" (injury news, a QB change, something the model's
box-score inputs can't see) -- never as a reason to trade against two
markets that agree.

HARD RULE: if kalshi_prob and yahoo_ml_prob are within ~5 points of each
other (they agree), your fair_prob_estimate MUST stay within that same
range and your verdict MUST be HOLD or, if you're not holding, PASS --
never BUY_MORE and never SELL on the strength of model/public_sharp_diff
alone. model and public_sharp_diff can raise your confidence in a verdict
the two markets already support, or lower it (nudge toward PASS), but they
can never flip the direction of your call away from what kalshi_prob and
yahoo_ml_prob agree on.

For OPEN positions, decide one of: HOLD (current price roughly matches your
fair-value estimate, or is still favorable), SELL (fair value has clearly
dropped below the current sell price -- cut it), BUY_MORE (fair value is
clearly above kalshi_prob -- rare, explain why), or PASS (signals conflict
or are too thin for a real opinion -- say so honestly instead of forcing a
take). For SOLD/WON/LOST positions, use verdict "SETTLED" and give a
retrospective: did they win or lose (status tells you), and -- ONLY if
user_note is non-null -- was their stated intuition actually good, judged
against the signals they had at entry time (kalshi_prob/yahoo_ml/model/
public_sharp_diff), not just against whether they happened to win. A
correct guess for a bad reason is BAD intuition; a loss despite sound
reasoning that the market/model also supported is GOOD intuition gone
wrong on variance -- say so plainly, don't just grade on the outcome.

For every position set intuition_grade based on user_note:
- null if user_note is null/missing (nothing to grade)
- "GOOD" if the stated reasoning was sound given what was knowable then
- "MIXED" if partly right, partly shaky, or right for an incomplete reason
- "BAD" if the reasoning was contradicted by the signals they had, or the
  win/loss was really about something else -- say what, plainly

Return ONLY a JSON object shaped exactly:
{{"positions": [
  {{"team": "...", "verdict": "HOLD|SELL|BUY_MORE|PASS|SETTLED",
   "confidence_1to5": 1-5, "fair_prob_estimate": 0.0-1.0 or null,
   "intuition_grade": "GOOD|MIXED|BAD|null",
   "reasoning": "2-3 sentences, concrete, cites the actual numbers you were given, and directly addresses user_note if one was given"}}
]}}
One object per position, in the SAME ORDER they were given."""


def call_openai(contexts, model):
    from openai import OpenAI
    client = OpenAI()
    payload = {"positions": contexts}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Here are the positions, in order:\n"
                                         + json.dumps(payload, indent=2, default=str)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        sys.exit(f"model did not return valid JSON:\n{content}")
    return data.get("positions", [])


def print_odds_table(contexts):
    print(f"\n{'Team':5} {'Opp':4} {'Status':7} {'Kalshi':>7} {'Model':>7} {'YahooOdds':>9} {'YahooML':>8} "
          f"{'Bet%':>6} {'Mon%':>6} {'SharpΔ':>7} {'GPscore':>8}")
    for c in contexts:
        if c.get("user_note"):
            print(f"  note [{c['team_abbr']}]: {c['user_note']}")
        model_p = c["model"]["model_win_prob"] if c["model"] else None
        yahoo = c["yahoo_ml"]
        yahoo_p = yahoo["yahoo_ml_prob"] if yahoo else None
        yahoo_odds = yahoo["yahoo_odds"] if yahoo else None
        bets_pct = yahoo["public_bets_pct"] if yahoo else None
        money_pct = yahoo["public_money_pct"] if yahoo else None
        sharp_diff = yahoo["public_sharp_diff"] if yahoo else None
        gp = c["composite"]["game_performance_score"] if c["composite"] else None
        model_str = "-".rjust(7) if model_p is None else f"{model_p * 100:6.1f}%"
        odds_str = "-".rjust(9) if yahoo_odds is None else f"{yahoo_odds:+9.0f}"
        yahoo_str = "-".rjust(8) if yahoo_p is None else f"{yahoo_p * 100:7.1f}%"
        bets_str = "-".rjust(6) if bets_pct is None else f"{bets_pct:5.1f}%"
        money_str = "-".rjust(6) if money_pct is None else f"{money_pct:5.1f}%"
        sharp_str = "-".rjust(7) if sharp_diff is None else f"{sharp_diff:+7.1f}"
        gp_str = "-".rjust(8) if gp is None else f"{gp:8.1f}"
        print(f"{c['team_abbr']:5} {c['opp_abbr'] or '-':4} {c['status']:7} "
              f"{c['kalshi_prob'] * 100:6.1f}% {model_str} {odds_str} {yahoo_str} "
              f"{bets_str} {money_str} {sharp_str} {gp_str}")


def log_results(contexts, verdicts, model_name, path=LOG_PATH):
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for c, v in zip(contexts, verdicts):
        rows.append({
            "logged_at": now, "team": c["team_abbr"], "opponent": c["opp_abbr"], "status": c["status"],
            "kalshi_prob": c["kalshi_prob"],
            "model_win_prob": c["model"]["model_win_prob"] if c["model"] else None,
            "yahoo_ml_odds": c["yahoo_ml"]["yahoo_odds"] if c["yahoo_ml"] else None,
            "yahoo_ml_prob": c["yahoo_ml"]["yahoo_ml_prob"] if c["yahoo_ml"] else None,
            "public_bets_pct": c["yahoo_ml"]["public_bets_pct"] if c["yahoo_ml"] else None,
            "public_money_pct": c["yahoo_ml"]["public_money_pct"] if c["yahoo_ml"] else None,
            "public_sharp_diff": c["yahoo_ml"]["public_sharp_diff"] if c["yahoo_ml"] else None,
            "game_performance_score": c["composite"]["game_performance_score"] if c["composite"] else None,
            "bought": c["bought"], "to_pay": c["to_pay"], "paid": c["paid"], "sell_price": c["sell_price"],
            "ai_model": model_name, "verdict": v.get("verdict"),
            "confidence_1to5": v.get("confidence_1to5"),
            "fair_prob_estimate": v.get("fair_prob_estimate"), "reasoning": v.get("reasoning"),
            "user_note": c.get("user_note"), "intuition_grade": v.get("intuition_grade"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"\nLogged {len(df)} verdicts to {os.path.abspath(path)}")


def main():
    p = argparse.ArgumentParser(
        description="Reconcile a pasted Kalshi NFL position slate against this repo's "
                     "SRS model + a Yahoo moneyline snapshot, and ask an LLM for a verdict.")
    p.add_argument("--file", help="path to a text file with the pasted Kalshi table (default: stdin)")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                   help="OpenAI chat model (env OPENAI_MODEL, default gpt-4o -- "
                        "override with whatever's current/strongest on your account, "
                        "e.g. --model gpt-5)")
    p.add_argument("--dry-run", action="store_true",
                   help="skip the OpenAI call, just print the merged odds table")
    p.add_argument("--no-log", action="store_true",
                   help="don't append results to kalshi_advisor_log.csv")
    p.add_argument("--no-notes", action="store_true",
                   help="don't prompt for (or use) 'why did you like this' notes on open positions")
    args = p.parse_args()

    text = open(args.file).read() if args.file else sys.stdin.read()
    bets = parse_slate(text)
    if not bets:
        sys.exit("no parseable positions found")

    notes = collect_notes(bets, no_notes=args.no_notes)
    contexts = [
        build_context(b, load_model_probs(), load_yahoo_signals(), load_composite(),
                      user_note=notes.get(i))
        for i, b in enumerate(bets)
    ]

    print_odds_table(contexts)
    print(f"\n{BREAKEVEN_NOTE}")

    if args.dry_run:
        print("\n(--dry-run: no OpenAI call made)")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("\nOPENAI_API_KEY is not set -- export it, or re-run with --dry-run "
                  "to just see the merged odds table.")

    verdicts = call_openai(contexts, args.model)
    if len(verdicts) != len(contexts):
        print(f"warning: model returned {len(verdicts)} verdicts for {len(contexts)} "
              f"positions -- pairing what matches", file=sys.stderr)

    print("\n=== AI verdicts ===")
    for c, v in zip(contexts, verdicts):
        print(f"\n{c['matchup_raw']}  [{c['status']}]")
        print(f"  -> {v.get('verdict')} (confidence {v.get('confidence_1to5')}/5, "
              f"fair_prob~{v.get('fair_prob_estimate')})")
        if v.get("intuition_grade"):
            print(f"     intuition: {v.get('intuition_grade')}")
        print(f"     {v.get('reasoning')}")

    if not args.no_log:
        log_results(contexts, verdicts, args.model)


if __name__ == "__main__":
    main()
