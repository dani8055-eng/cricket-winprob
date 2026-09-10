"""
Prepare a compact JSON of test-match chases for the replay dashboard.

Joins the model's out-of-sample predictions (outputs/test_predictions.csv) with
the ball-by-ball context (data/interim/chase_ballbyball.csv) to get team names
and the running score, then writes a small JSON the HTML dashboard can load.

To keep the file small and the dashboard snappy, we include a curated set of
matches: a mix of close finishes and blowouts makes the win-prob line most
interesting to watch. We pick matches by how much the win-prob line *moved*
(most dramatic swings first), plus keep it to a manageable number.

OUTPUT: outputs/replay_data.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
OUT = Path("outputs")

N_MATCHES = 60   # how many matches to include in the dashboard


def main():
    pred = pd.read_csv(OUT / "test_predictions.csv")
    ctx = pd.read_csv(INTERIM / "chase_ballbyball.csv")

    # context columns we want for a readable broadcast-style display
    keep = ['match_id', 'ball_no', 'chasing_team', 'bowling_team',
            'runs_so_far', 'target', 'wickets_fallen']
    ctx = ctx[keep]

    df = pred.merge(ctx, on=['match_id', 'ball_no'], how='left')

    # "drama" = how much the model's win-prob swung over the match
    swing = df.groupby('match_id')['winprob_model'].agg(lambda s: s.max() - s.min())
    # also require a reasonable length (full-ish chases) so the line is meaningful
    length = df.groupby('match_id')['ball_no'].max()
    good = length[length >= 30].index
    swing = swing.loc[swing.index.isin(good)].sort_values(ascending=False)

    chosen = list(swing.head(N_MATCHES).index)

    matches = []
    for mid in chosen:
        m = df[df['match_id'] == mid].sort_values('ball_no')
        if m.empty:
            continue
        row0 = m.iloc[0]
        matches.append({
            'id': str(mid),
            'chasing': str(row0['chasing_team']),
            'bowling': str(row0['bowling_team']),
            'target': int(row0['target']),
            'won': int(row0['chasing_won']),
            # per-ball series (rounded to keep the file small)
            'ball': m['ball_no'].astype(int).tolist(),
            'runs': m['runs_so_far'].astype(int).tolist(),
            'wkts': m['wickets_fallen'].astype(int).tolist(),
            'need': m['runs_needed'].astype(int).tolist(),
            'wp':   [round(float(x), 3) for x in m['winprob_model']],
            'naive':[round(float(x), 3) for x in m['winprob_naive']],
        })

    # sort the dropdown by a friendly label
    matches.sort(key=lambda x: (x['chasing'], x['bowling']))
    (OUT / "replay_data.json").write_text(json.dumps(matches))
    print(f"Saved: {OUT / 'replay_data.json'}  ({len(matches)} matches)")
    print("Sample match:", matches[0]['chasing'], "chasing", matches[0]['target'],
          "vs", matches[0]['bowling'], "-> won" if matches[0]['won'] else "-> lost")


if __name__ == "__main__":
    main()
