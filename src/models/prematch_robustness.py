"""
Robustness check for the pre-match predictor.

The main model got ~70% accuracy, but head-to-head (h2h_team1) dominated feature
importance. Head-to-head can be noisy (many team pairs have few prior meetings),
so we honestly test how much the result depends on it:

  1. Model WITHOUT h2h -> does accuracy hold up, or collapse?
  2. How many prior meetings back each test-match's h2h value? (is it real signal
     or often based on 1-2 games?)

This tells us whether the ~70% is robust team-strength signal or is being propped
up by a thin, possibly-memorised feature.

INPUT:  data/interim/prematch_matches.csv
OUTPUT: prints comparison
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

INTERIM = Path("data/interim")

ALL = ['team1_winrate', 'team2_winrate', 'team1_form5', 'team2_form5',
       'team1_form10', 'team2_form10', 'h2h_team1',
       'team1_matches', 'team2_matches', 'venue_team1_wr']
NO_H2H = [f for f in ALL if f != 'h2h_team1']
SEED = 42


def fit_eval(train, test, feats):
    m = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric='logloss', random_state=SEED, n_jobs=-1)
    m.fit(train[feats].values, train['team1_won'].values)
    p = m.predict_proba(test[feats].values)[:, 1]
    acc = accuracy_score(test['team1_won'].values, (p >= 0.5).astype(int))
    auc = roc_auc_score(test['team1_won'].values, p)
    return acc, auc


def main():
    df = pd.read_csv(INTERIM / "prematch_matches.csv").sort_values('date').reset_index(drop=True)
    df = df[df['both_have_history'] == 1].reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]

    print("=" * 56)
    print("ROBUSTNESS: does the ~70% survive without head-to-head?")
    print("=" * 56)
    a_all, u_all = fit_eval(train, test, ALL)
    a_no, u_no = fit_eval(train, test, NO_H2H)
    print(f"WITH head-to-head:    {100*a_all:5.1f}% acc  (AUC {u_all:.3f})")
    print(f"WITHOUT head-to-head: {100*a_no:5.1f}% acc  (AUC {u_no:.3f})")
    drop = 100*(a_all - a_no)
    print(f"Drop from removing h2h: {drop:+.1f} points")
    if a_no >= 0.62:
        print(">> Robust: strong team-strength signal even without h2h.")
    else:
        print(">> The result leans heavily on head-to-head; report that honestly.")

    # how thin is the h2h signal? we can't recount meetings here without the raw
    # history, but h2h_team1 == 0.5 means 'no prior meetings' (our default).
    no_h2h_info = (test['h2h_team1'] == 0.5).mean()
    print(f"\nShare of test matches with NO prior head-to-head (h2h=0.5 default): "
          f"{100*no_h2h_info:.0f}%")
    print("(For those, the h2h feature carries no real information.)")


if __name__ == "__main__":
    main()
