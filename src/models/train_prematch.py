"""
Pre-match T20 winner predictor -- and an honest measure of how predictable
T20 outcomes even are.

Predicts whether team1 wins, from pre-match strength features (form, head-to-head,
venue, experience) built leak-free by parse_prematch.py.

HONEST FRAMING (the point of this project):
  T20 is a famously high-variance format. So the interesting question is not
  "can I predict winners" but "HOW MUCH of the result is explainable by team
  strength, versus irreducible randomness?" We answer that by comparing:
    - a coin-flip baseline (50%),
    - a simple rule ("team with better recent form wins"),
    - the model.
  The gap between the model and 50% is the honest, quantified predictability.

TIME-ORDERED SPLIT (no leakage): we sort by date and train on the EARLIER
matches, test on the LATER ones -- mirroring real forecasting (predict the future
from the past), and matching how the features were built.

INPUT:  data/interim/prematch_matches.csv
OUTPUT: prints metrics + feature importance; saves outputs/prematch_metrics.txt
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ['team1_winrate', 'team2_winrate', 'team1_form5', 'team2_form5',
            'team1_form10', 'team2_form10', 'h2h_team1',
            'team1_matches', 'team2_matches', 'venue_team1_wr']
SEED = 42


def main():
    df = pd.read_csv(INTERIM / "prematch_matches.csv").sort_values('date').reset_index(drop=True)
    # focus on matches where both teams have some history (real signal)
    df = df[df['both_have_history'] == 1].reset_index(drop=True)
    print(f"Matches with history for both teams: {len(df):,}")

    # time-ordered split: first 80% train, last 20% test
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f"Train (earlier): {len(train):,}  |  Test (later): {len(test):,}")

    Xtr, ytr = train[FEATURES].values, train['team1_won'].values
    Xte, yte = test[FEATURES].values, test['team1_won'].values

    # ---- baselines ----
    # 1. coin flip
    base_coin = 0.5
    # 2. simple rule: team with better recent form (form10) wins
    rule_pred = (test['team1_form10'] > test['team2_form10']).astype(int)
    # ties in form -> predict team1 (arbitrary); accuracy of the rule:
    acc_rule = accuracy_score(yte, rule_pred)

    # ---- model ----
    model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    acc_model = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, p)
    brier = brier_score_loss(yte, p)

    print("\n" + "=" * 58)
    print("HOW PREDICTABLE IS A T20 MATCH? (test = later matches)")
    print("=" * 58)
    print(f"Coin flip baseline:          {100*base_coin:5.1f}% accuracy")
    print(f"'Better recent form' rule:   {100*acc_rule:5.1f}% accuracy")
    print(f"XGBoost model:               {100*acc_model:5.1f}% accuracy")
    print(f"                             (ROC-AUC {auc:.3f}, Brier {brier:.3f})")
    edge = 100*(acc_model - 0.5)
    print(f"\nModel's edge over a coin flip: +{edge:.1f} percentage points")
    print("Interpretation: team strength explains part of the outcome, but a large")
    print("share of a T20 result is genuinely unpredictable from prior form.")

    # ---- what matters ----
    imp = pd.DataFrame({'feature': FEATURES, 'importance': model.feature_importances_}) \
        .sort_values('importance', ascending=False)
    print("\nMost useful features:")
    for _, r in imp.head(6).iterrows():
        print(f"  {r['feature']:18} {r['importance']:.3f}")

    with open(OUT / "prematch_metrics.txt", "w") as fh:
        fh.write(f"coin {base_coin:.3f}\nrule {acc_rule:.3f}\n"
                 f"model_acc {acc_model:.3f}\nauc {auc:.3f}\nbrier {brier:.3f}\n")
    print(f"\nSaved: {OUT / 'prematch_metrics.txt'}")


if __name__ == "__main__":
    main()
