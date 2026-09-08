"""
Train a T20 chase win-probability model, and check whether it is CALIBRATED.

The model predicts, for each ball of a run-chase, the probability that the
chasing team goes on to win. Trained with XGBoost on match-situation features.

THE TWO THINGS THAT MAKE THIS HONEST:

1. MATCH-LEVEL SPLIT (no leakage). Every ball in one match shares the same final
   outcome. If we split balls randomly, balls from the same match land in both
   train and test and the model just memorises "match X was won". So we split by
   MATCH: whole matches go to train OR test, never both. This is the cricket
   version of the look-ahead discipline.

2. CALIBRATION. A win-prob model is only trustworthy if its numbers mean what
   they say: among all balls where it says ~70%, the team should actually win
   ~70% of the time. We measure this directly (reliability table) and compare the
   model against a NAIVE baseline to show the difference honesty makes.

FEATURES (kept simple and meaningful):
  balls_left, wickets_in_hand, runs_needed, and a derived
  required_run_rate = runs_needed / overs_left  -- the number fans actually watch
  (an example of extracting a meaningful quantity from raw columns).

INPUT:  data/interim/chase_ballbyball.csv
OUTPUT: outputs/winprob_model_metrics.txt (printed), outputs/calibration.csv
        models saved implicitly via predictions file for the dashboard later.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ['balls_left', 'wickets_in_hand', 'runs_needed', 'required_run_rate']
SEED = 42


def add_features(df):
    """Engineer required run rate; clean the rare edge cases."""
    df = df.copy()
    # drop the odd negative-wickets edge rows (rare data quirks)
    df = df[(df['wickets_in_hand'] >= 0) & (df['wickets_in_hand'] <= 10)]
    overs_left = df['balls_left'] / 6.0
    # required run rate = runs still needed / overs remaining.
    # guard against divide-by-zero on the very last ball.
    df['required_run_rate'] = np.where(overs_left > 0,
                                       df['runs_needed'] / overs_left,
                                       df['runs_needed'] * 6.0)  # cap: all needed off <1 over
    return df


def match_level_split(df, test_frac=0.2, seed=SEED):
    """Assign whole matches to train or test -- never split a match across both."""
    rng = np.random.default_rng(seed)
    matches = df['match_id'].unique()
    rng.shuffle(matches)
    n_test = int(len(matches) * test_frac)
    test_matches = set(matches[:n_test])
    is_test = df['match_id'].isin(test_matches)
    return df[~is_test].copy(), df[is_test].copy()


def naive_winprob(df):
    """
    A simple, honest baseline a commentator might use:
    win prob rises with wickets in hand and falls as required run rate climbs.
    Purely rule-of-thumb (logistic on just RRR and wickets), NOT tuned -- it
    exists to show what an 'intuitive' estimate looks like vs a calibrated model.
    """
    # crude score -> squashed to 0..1
    score = 2.0 * (df['wickets_in_hand'] - 5) / 5.0 - (df['required_run_rate'] - 8) / 4.0
    return 1 / (1 + np.exp(-score))


def calibration_table(y_true, y_prob, bins=10):
    """Reliability: for each predicted-probability bucket, the actual win rate."""
    df = pd.DataFrame({'p': y_prob, 'y': y_true})
    df['bucket'] = pd.cut(df['p'], np.linspace(0, 1, bins + 1), include_lowest=True)
    tab = df.groupby('bucket').agg(n=('y', 'size'),
                                   predicted=('p', 'mean'),
                                   actual=('y', 'mean')).dropna()
    return tab


def main():
    df = pd.read_csv(INTERIM / "chase_ballbyball.csv")
    df = add_features(df)
    print(f"Rows after cleaning: {len(df):,} across {df['match_id'].nunique():,} matches")

    train, test = match_level_split(df)
    print(f"Train matches: {train['match_id'].nunique():,}  |  "
          f"Test matches: {test['match_id'].nunique():,}  (no match in both)")

    Xtr, ytr = train[FEATURES].values, train['chasing_won'].values
    Xte, yte = test[FEATURES].values, test['chasing_won'].values

    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr)
    p_model = model.predict_proba(Xte)[:, 1]
    p_naive = naive_winprob(test).values

    # --- metrics ---
    print("\n" + "=" * 60)
    print("MODEL vs NAIVE  (test set, held-out matches)")
    print("=" * 60)
    for name, p in [('XGBoost model', p_model), ('Naive rule', p_naive)]:
        auc = roc_auc_score(yte, p)
        brier = brier_score_loss(yte, p)   # lower = better calibrated
        ll = log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))
        print(f"{name:<16} ROC-AUC {auc:.3f} | Brier {brier:.3f} (lower=better) | LogLoss {ll:.3f}")

    # --- calibration table for the model ---
    print("\nCALIBRATION (model): does a stated probability match the real win rate?")
    tab = calibration_table(yte, p_model)
    print(f"{'predicted':>10}{'actual':>10}{'n':>9}")
    for _, r in tab.iterrows():
        print(f"{r['predicted']:>10.2f}{r['actual']:>10.2f}{int(r['n']):>9}")
    tab.to_csv(OUT / "calibration.csv")

    # save test predictions for the eventual live/replay dashboard
    outdf = test[['match_id', 'ball_no', 'balls_left', 'wickets_in_hand',
                  'runs_needed', 'required_run_rate', 'chasing_won']].copy()
    outdf['winprob_model'] = p_model
    outdf['winprob_naive'] = p_naive
    outdf.to_csv(OUT / "test_predictions.csv", index=False)
    print(f"\nSaved calibration.csv and test_predictions.csv to outputs/")


if __name__ == "__main__":
    main()
