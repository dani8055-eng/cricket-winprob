"""
Fit a lightweight logistic win-probability model for the browser 'live mode'.

The full analysis uses XGBoost. But a web page can't run XGBoost directly, so for
the live calculator we fit a simple LOGISTIC REGRESSION on the same features and
embed its handful of coefficients in the dashboard's JavaScript -- pure math,
instant, self-contained, no server.

This script:
  1. fits the logistic model (same match-level split, no leakage),
  2. reports how closely it matches the full XGBoost model (so the approximation
     is stated honestly, not hidden),
  3. prints the coefficients to paste into the dashboard.

Features (same as the main model):
  balls_left, wickets_in_hand, runs_needed, required_run_rate

We standardise features so the coefficients are on a comparable scale; the script
prints BOTH the standardisation constants (mean/std) and the coefficients, which
is everything the browser needs to reproduce the prediction exactly.

INPUT:  data/interim/chase_ballbyball.csv
OUTPUT: prints coefficients + accuracy comparison; saves outputs/live_model.json
"""

import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

INTERIM = Path("data/interim")
OUT = Path("outputs")

FEATURES = ['balls_left', 'wickets_in_hand', 'runs_needed', 'required_run_rate']
SEED = 42


def add_features(df):
    df = df.copy()
    df = df[(df['wickets_in_hand'] >= 0) & (df['wickets_in_hand'] <= 10)]
    overs_left = df['balls_left'] / 6.0
    df['required_run_rate'] = np.where(overs_left > 0,
                                       df['runs_needed'] / overs_left,
                                       df['runs_needed'] * 6.0)
    return df


def match_split(df, test_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    matches = df['match_id'].unique().copy()
    rng.shuffle(matches)
    test = set(matches[:int(len(matches) * test_frac)])
    m = df['match_id'].isin(test)
    return df[~m].copy(), df[m].copy()


def main():
    df = add_features(pd.read_csv(INTERIM / "chase_ballbyball.csv"))
    train, test = match_split(df)

    Xtr, ytr = train[FEATURES].values, train['chasing_won'].values
    Xte, yte = test[FEATURES].values, test['chasing_won'].values

    # standardise (store mean/std so the browser can reproduce it)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    Xtr_s = (Xtr - mean) / std
    Xte_s = (Xte - mean) / std

    logit = LogisticRegression(max_iter=2000)
    logit.fit(Xtr_s, ytr)
    p_logit = logit.predict_proba(Xte_s)[:, 1]

    # full model for comparison
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=SEED, n_jobs=-1)
    xgb.fit(Xtr, ytr)
    p_xgb = xgb.predict_proba(Xte)[:, 1]

    print("=" * 58)
    print("SIMPLE (logistic) vs FULL (XGBoost) on held-out matches")
    print("=" * 58)
    print(f"{'':16}{'ROC-AUC':>10}{'Brier':>10}")
    print(f"{'Simple logistic':16}{roc_auc_score(yte,p_logit):>10.3f}{brier_score_loss(yte,p_logit):>10.3f}")
    print(f"{'Full XGBoost':16}{roc_auc_score(yte,p_xgb):>10.3f}{brier_score_loss(yte,p_xgb):>10.3f}")
    # how close are the two models' probabilities?
    corr = np.corrcoef(p_logit, p_xgb)[0, 1]
    mad = np.mean(np.abs(p_logit - p_xgb))
    print(f"\nAgreement between the two: correlation {corr:.3f}, "
          f"avg abs difference {100*mad:.1f} percentage points")
    print("(The simpler model is what the live browser calculator uses.)")

    # export everything the browser needs
    payload = {
        'features': FEATURES,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'coef': logit.coef_[0].tolist(),
        'intercept': float(logit.intercept_[0]),
    }
    (OUT / "live_model.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved coefficients to {OUT / 'live_model.json'}")
    print("\nCoefficients (standardised features):")
    for f, c in zip(FEATURES, logit.coef_[0]):
        print(f"  {f:20} {c:+.4f}")
    print(f"  {'intercept':20} {logit.intercept_[0]:+.4f}")


if __name__ == "__main__":
    main()
