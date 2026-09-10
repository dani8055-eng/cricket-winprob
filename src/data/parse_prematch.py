"""
Build a match-level table for the PRE-MATCH win predictor (T20).

One row per match, with features that describe each side's strength BEFORE the
match, and the label: did team1 win?

THE CRITICAL DISCIPLINE — NO LOOK-AHEAD:
  To predict a match on date D, every feature must use only matches BEFORE D.
  We therefore process matches in strict DATE ORDER and, for each match, compute
  each team's form/history from the games already seen. Then we record the
  outcome and only afterwards update the running history. A team's features for
  their 2019 game use only their pre-2019 games -- never the future. This is the
  cricket analogue of lagging macro data: features reflect what was knowable at
  the time, so the eventual accuracy is honest.

FEATURES (per match, from history so far):
  - team1_winrate, team2_winrate         overall win rate to date
  - team1_form5, team2_form5             win rate over each team's last 5 games
  - team1_form10, team2_form10           win rate over last 10 games
  - h2h_team1                            team1's win rate in prior meetings vs team2
  - team1_matches, team2_matches         games played to date (experience/warmup)
  - same_venue_team1_wr                  team1 win rate at this venue (if any history)
Matches where either side has NO prior history are still kept (features default
to 0.5 = "no information"), and a flag `both_have_history` marks the ones with
real signal, so we can evaluate on the informative subset too.

Team identity: internationals and leagues are mixed (as chosen). Each team's
history is tracked by its name string as it appears in the data.

Ties / no-results are dropped (need a clear winner).

INPUT:  a directory of Cricsheet T20 .json files (default ~/Downloads/t20s_json)
OUTPUT: data/interim/prematch_matches.csv
"""

import json
import sys
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd

RAW_DIR = Path.home() / "Downloads" / "t20s_json"
OUT = Path("data/interim"); OUT.mkdir(parents=True, exist_ok=True)


def load_matches(raw):
    """Read all matches, keep the fields we need, sorted by date."""
    rows = []
    files = sorted(raw.glob("*.json"))
    print(f"Reading {len(files)} files...", flush=True)
    for i, f in enumerate(files, 1):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        info = m.get("info", {})
        if info.get("match_type") != "T20":
            continue
        teams = info.get("teams", [])
        if len(teams) != 2:
            continue
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner:  # tie / no result
            continue
        dates = info.get("dates", [])
        if not dates:
            continue
        rows.append({
            "match_id": f.stem,
            "date": dates[0],
            "team1": teams[0],
            "team2": teams[1],
            "winner": winner,
            "venue": info.get("venue", "unknown"),
        })
        if i % 1000 == 0:
            print(f"  ...{i} files", flush=True)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"Usable matches with a clear winner: {len(df)}")
    return df


def wr(wins, games):
    return wins / games if games > 0 else 0.5  # 0.5 = no information


def main():
    raw = RAW_DIR
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1])
    matches = load_matches(raw)

    # running history (updated AFTER each match is recorded -> no leak)
    total = defaultdict(lambda: [0, 0])          # team -> [wins, games]
    lastN = defaultdict(lambda: deque(maxlen=10))  # team -> recent results (1/0)
    h2h = defaultdict(lambda: [0, 0])            # (a,b) -> [a_wins, meetings]
    venue_wl = defaultdict(lambda: [0, 0])       # (team,venue) -> [wins, games]

    out_rows = []
    for _, m in matches.iterrows():
        t1, t2, venue = m["team1"], m["team2"], m["venue"]

        def form(team, n):
            d = list(lastN[team])
            d = d[-n:]
            return sum(d) / len(d) if d else 0.5

        h1w, hm = h2h[(t1, t2)]
        both = total[t1][1] > 0 and total[t2][1] > 0

        out_rows.append({
            "match_id": m["match_id"], "date": m["date"],
            "team1": t1, "team2": t2, "venue": venue,
            "team1_winrate": wr(*total[t1][::-1][::-1]) if False else wr(total[t1][0], total[t1][1]),
            "team2_winrate": wr(total[t2][0], total[t2][1]),
            "team1_form5": form(t1, 5), "team2_form5": form(t2, 5),
            "team1_form10": form(t1, 10), "team2_form10": form(t2, 10),
            "h2h_team1": (h1w / hm) if hm > 0 else 0.5,
            "team1_matches": total[t1][1], "team2_matches": total[t2][1],
            "venue_team1_wr": wr(venue_wl[(t1, venue)][0], venue_wl[(t1, venue)][1]),
            "both_have_history": int(both),
            "team1_won": int(m["winner"] == t1),
        })

        # --- NOW update history (after recording, so features stayed leak-free) ---
        w = m["winner"]
        for t in (t1, t2):
            total[t][1] += 1
            venue_wl[(t, venue)][1] += 1
        total[w][0] += 1
        venue_wl[(w, venue)][0] += 1
        lastN[t1].append(1 if w == t1 else 0)
        lastN[t2].append(1 if w == t2 else 0)
        h2h[(t1, t2)][1] += 1; h2h[(t2, t1)][1] += 1
        if w == t1: h2h[(t1, t2)][0] += 1
        else: h2h[(t2, t1)][0] += 1

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT / "prematch_matches.csv", index=False)
    print(f"\nSaved: {OUT / 'prematch_matches.csv'}  ({len(df):,} matches)")
    print(f"Matches where both teams have prior history: "
          f"{df['both_have_history'].sum():,} "
          f"({100*df['both_have_history'].mean():.0f}%)")
    print(f"Overall team1 win rate (sanity, ~0.5 expected): "
          f"{df['team1_won'].mean():.3f}")


if __name__ == "__main__":
    main()
