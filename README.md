# Cricket Win-Probability (T20 Chases)

A live-style **win-probability model for T20 run chases** — and an honest check of
whether its probabilities actually mean what they say.

Given the situation at any ball of a chase (runs needed, balls left, wickets in
hand), the model estimates the chance the chasing team wins. Trained on **5,507
completed T20 matches** (~583,000 balls) from [Cricsheet](https://cricsheet.org).

## The question

Broadcast win-probability graphics look authoritative, but are they *calibrated* —
when they say "70%", does the team really win 70% of the time? This project builds
a proper model and measures that directly, against a naive rule-of-thumb.

## How it's built (and kept honest)

- **Match-level train/test split.** Every ball in a match shares the same final
  outcome, so splitting balls randomly would leak the answer. Whole matches go to
  train *or* test, never both.
- **Meaningful features.** Balls left, wickets in hand, runs needed, and the
  derived *required run rate* (runs needed ÷ overs left) — the number fans watch.
- **Calibration check.** A reliability table compares predicted probability to the
  actual win rate in each bucket.

## Results (held-out matches)

| | ROC-AUC | Brier score (lower = better) |
|---|---|---|
| **XGBoost model** | **0.908** | **0.124** |
| Naive rule-of-thumb | 0.831 | 0.209 |

The model is both **more accurate** and **far better calibrated** than the naive
rule. In the reliability table, predicted probability closely tracks the actual
win rate across the whole range (e.g. it says 0.85 → teams win 0.85). A slight
under-confidence in the middle buckets is reported honestly rather than hidden.

## Honest limits

- Wides/no-balls are currently counted as balls of position (a small
  simplification; treating them as not advancing the legal-ball count is future
  work).
- Chase (2nd innings) only for now; first-innings win probability is a harder,
  separate problem and a natural extension.
- Matches with no clear winner (ties / no-result / D-L / super overs) are excluded.

## Repository

```
src/data/parse_chase.py     Cricsheet JSON -> clean ball-by-ball chase table
src/models/train_winprob.py Model, match-level split, calibration analysis
data/interim/               parsed table
outputs/                    calibration table, test predictions
```

## Data

Ball-by-ball JSON from **Cricsheet** (cricsheet.org), T20 match archive. The raw
files are not committed (large); download the T20 JSON set and point the parser at
the folder.

---

*Independent research project. Emphasis on honest probabilities over impressive-
looking-but-miscalibrated ones.*
