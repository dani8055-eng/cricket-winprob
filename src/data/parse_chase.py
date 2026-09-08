"""
Parse Cricsheet T20 JSON match files into a clean ball-by-ball CHASE table.

Each output row = one legal delivery of the SECOND innings (the run-chase),
capturing the match situation at that ball and whether the chasing team won.
This is the modelling base for a chase win-probability model.

WHY THE CHASE (2nd innings): the chasing team knows exactly what it needs, so
each ball has a well-defined state (runs needed, balls left, wickets in hand).
That makes win probability a clean, honest question.

CRICKET-SPECIFIC DECISIONS (the careful bit):
  - TARGET = total runs scored in the 1st innings + 1. We sum every delivery's
    runs.total across the first innings.
  - RUNS SO FAR in the chase = cumulative runs.total up to and including this ball.
  - WICKETS FALLEN = cumulative count of deliveries that carried a 'wickets' entry.
  - BALLS: T20 is 20 overs x 6 = 120 balls. We number legal deliveries 1..N.
    (Cricsheet lists each delivery in order; extras like wides/no-balls still
    appear as deliveries. For a first clean version we count every listed
    delivery as one ball of position. A later refinement can treat wides/no-balls
    as not advancing the legal-ball count -- flagged as future work.)
  - LABEL: did the team batting SECOND win? From info.outcome.winner.
    Matches with no clear winner (ties, no-result, D/L, super overs) are SKIPPED
    for this clean first version -- documented, not hidden.

Uses a dictionary (hash map) to look up each match's outcome quickly -- exactly
the "hash map" data structure: O(1) lookups by key instead of scanning a list.

INPUT:  a directory of Cricsheet T20 .json files
OUTPUT: data/interim/chase_ballbyball.csv
"""

import json
import sys
from pathlib import Path
import pandas as pd

RAW_DIR = Path.home() / "Downloads" / "t20s_json"   # where the JSON files live
OUT = Path("data/interim"); OUT.mkdir(parents=True, exist_ok=True)
TOTAL_BALLS = 120  # 20 overs * 6


def innings_total(innings):
    """Sum every delivery's total runs across one innings."""
    total = 0
    for over in innings["overs"]:
        for d in over["deliveries"]:
            total += d["runs"]["total"]
    return total


def parse_match(path):
    """Return a list of per-ball dict rows for the chase, or [] if unusable."""
    try:
        m = json.load(open(path))
    except Exception:
        return []

    info = m.get("info", {})
    if info.get("match_type") != "T20":
        return []
    innings = m.get("innings", [])
    if len(innings) < 2:
        return []  # need a completed chase

    # --- who won? clean winner only ---
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    if not winner:
        return []  # tie / no result / abandoned -> skip in this clean version

    first, second = innings[0], innings[1]
    chasing_team = second.get("team")
    target = innings_total(first) + 1
    chasing_won = 1 if winner == chasing_team else 0

    match_id = Path(path).stem
    teams = [first.get("team"), second.get("team")]

    rows = []
    runs = 0
    wkts = 0
    ball_no = 0
    for over in second["overs"]:
        for d in over["deliveries"]:
            ball_no += 1
            runs += d["runs"]["total"]
            if "wickets" in d:
                wkts += len(d["wickets"])
            balls_left = max(TOTAL_BALLS - ball_no, 0)
            rows.append({
                "match_id": match_id,
                "chasing_team": chasing_team,
                "bowling_team": teams[0],
                "ball_no": ball_no,
                "balls_left": balls_left,
                "runs_so_far": runs,
                "wickets_fallen": wkts,
                "wickets_in_hand": 10 - wkts,
                "target": target,
                "runs_needed": max(target - runs, 0),
                "chasing_won": chasing_won,
            })
    return rows


def main():
    raw = RAW_DIR
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1])
    files = sorted(raw.glob("*.json"))
    print(f"Found {len(files)} match files in {raw}")

    all_rows = []
    used, skipped = 0, 0
    for i, f in enumerate(files, 1):
        r = parse_match(f)
        if r:
            all_rows.extend(r); used += 1
        else:
            skipped += 1
        if i % 500 == 0:
            print(f"  ...{i}/{len(files)} processed", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "chase_ballbyball.csv", index=False)
    print(f"\nSaved: {OUT / 'chase_ballbyball.csv'}")
    print(f"Matches used (clean chases): {used}   skipped: {skipped}")
    print(f"Total ball rows: {len(df):,}")
    if len(df):
        print(f"Chases won by the batting-second team: "
              f"{100*df.groupby('match_id')['chasing_won'].first().mean():.1f}% of matches")


if __name__ == "__main__":
    main()
