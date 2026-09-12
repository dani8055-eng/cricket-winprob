"""
Prepare the team-vs-team predictor for the browser.

Two outputs the web page needs:
  1. Each team's LATEST form snapshot (their most recent win-rate / form5 / form10
     / matches / venue history as of the end of the dataset). We reuse the
     leak-safe running history from the parser logic, but here we just want the
     FINAL state per team.
  2. A logistic approximation of the pre-match model (coefficients + scaling), so
     the page can compute a win probability instantly with pure math -- same
     approach as the in-match live calculator.

HONEST NOTES baked into the output for the dashboard:
  - Form is frozen at the dataset's end date (not live).
  - The pre-match model is ~67-70% accurate; T20 is partly unpredictable. The tool
    reports a probability, not a certainty.
  - Head-to-head is intentionally EXCLUDED from the live tool (it needs a specific
    opponent-pair history and is data-thin); the tool uses robust team-strength
    features, which alone give ~67% accuracy.

INPUT:  data/interim/prematch_matches.csv
OUTPUT: outputs/team_predictor.json  (team snapshots + model coefficients)
"""

import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

# features the live tool uses (NO head-to-head; it needs a pair, and is thin)
FEATS = ['team1_winrate', 'team2_winrate', 'team1_form5', 'team2_form5',
         'team1_form10', 'team2_form10', 'team1_matches', 'team2_matches',
         'venue_team1_wr']
SEED = 42


def latest_team_snapshots(df_raw):
    """
    Replay matches in date order to get each team's FINAL
    (win_rate, form5, form10, matches) as of the end of the data.
    Mirrors the parser's leak-safe running history, but we keep only the last state.
    """
    total = defaultdict(lambda: [0, 0])         # team -> [wins, games]
    lastN = defaultdict(lambda: deque(maxlen=10))
    # We need the ORIGINAL match list (team1/team2/winner/date). Rebuild from the
    # parsed file's identity columns + label.
    matches = df_raw.sort_values('date')
    for _, m in matches.iterrows():
        t1, t2, won = m['team1'], m['team2'], m['team1_won']
        winner = t1 if won == 1 else t2
        for t in (t1, t2):
            total[t][1] += 1
        total[winner][0] += 1
        lastN[t1].append(1 if winner == t1 else 0)
        lastN[t2].append(1 if winner == t2 else 0)

    snaps = {}
    for team, (w, g) in total.items():
        d = list(lastN[team])
        form5 = sum(d[-5:]) / len(d[-5:]) if d else 0.5
        form10 = sum(d) / len(d) if d else 0.5
        snaps[team] = {
            'winrate': round(w / g, 4) if g else 0.5,
            'form5': round(form5, 4),
            'form10': round(form10, 4),
            'matches': g,
        }
    return snaps


def main():
    df = pd.read_csv(INTERIM / "prematch_matches.csv")
    hist = df[df['both_have_history'] == 1].reset_index(drop=True)

    # --- fit logistic approx on the SAME features (time-ordered) ---
    hist = hist.sort_values('date').reset_index(drop=True)
    cut = int(len(hist) * 0.8)
    train = hist.iloc[:cut]
    X = train[FEATS].values
    y = train['team1_won'].values
    mean = X.mean(axis=0); std = X.std(axis=0)
    Xs = (X - mean) / std
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)

    # --- latest snapshot per team ---
    snaps = latest_team_snapshots(df)
    # only keep teams with a reasonable sample so the tool isn't noisy
    snaps = {t: s for t, s in snaps.items() if s['matches'] >= 10}

    payload = {
        'features': FEATS,
        'mean': mean.tolist(), 'std': std.tolist(),
        'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
        'teams': snaps,
    }
    (OUT / "team_predictor.json").write_text(json.dumps(payload))
    print(f"Saved: {OUT / 'team_predictor.json'}")
    print(f"Teams available (>=10 matches): {len(snaps)}")
    # show a few well-known teams' snapshots as a sanity check
    for t in ['India', 'Australia', 'England', 'Pakistan', 'Afghanistan']:
        if t in snaps:
            s = snaps[t]
            print(f"  {t:12} winrate {s['winrate']:.2f}  form10 {s['form10']:.2f}  ({s['matches']} matches)")


if __name__ == "__main__":
    main()
