import pandas as pd
from collections import Counter
import math
import random

TEAMS = [
    # seed, team name
    (1, "Cornell"),
    (2, "Maryland"),
    (3, "Princeton"),
    (4, "Ohio State"),
    (5, "Penn State"),
    (6, "Syracuse"),
    (7, "Duke"),
    (8, "North Carolina"),
    (9, "Richmond"),
    (10, "Georgetown"),
    (11, "Harvard"),
    (12, "Colgate"),
    (13, "Notre Dame"),
    (14, "Towson"),
    (15, "Air Force"),
    (16, "Albany"),
]

#Ratings
RATINGS = {
    "Cornell": 2008,
    "Maryland": 1923,
    "Princeton": 1870,
    "Ohio State": 1826,
    "Penn State": 1848,
    "Syracuse": 1817,
    "Duke": 1765,
    "North Carolina": 1766,
    "Richmond": 1815,
    "Georgetown": 1723,
    "Harvard": 1767,
    "Colgate": 1666,
    "Notre Dame": 1806,
    "Towson": 1613,
    "Air Force": 1535,
    "Albany": 1567,
}

# Slight upset/noise factor (in Elo points) added per game
RATING_NOISE_SD = 15


def _win_prob(team_a: str, team_b: str) -> float:
    """Probability team_a beats team_b from Elo ratings (with small Gaussian noise)."""
    ra = RATINGS[team_a]
    rb = RATINGS[team_b]
    if RATING_NOISE_SD > 0:
        ra += random.gauss(0, RATING_NOISE_SD)
        rb += random.gauss(0, RATING_NOISE_SD)
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))


def _seed_order(pairings):
    """
    Given a list of (seed, name), return the order for the round
    standard bracketing
    """
    pairings = sorted(pairings, key=lambda x: x[0])
    n = len(pairings)
    ordered = []
    for i in range(n // 2):
        ordered.append(pairings[i])           # high seed
        ordered.append(pairings[-(i + 1)])    # corresponding low seed
    return ordered


def _play_round(teams_in_round):
    """Play one round; teams_in_round is list of (seed, name) in match order (pairs play)."""
    winners = []
    for i in range(0, len(teams_in_round), 2):
        seed_a, name_a = teams_in_round[i]
        seed_b, name_b = teams_in_round[i + 1]
        p = _win_prob(name_a, name_b)
        winner = (seed_a, name_a) if random.random() < p else (seed_b, name_b)
        winners.append(winner)
    return winners


def simulate_bracket():
    """
    Simulate the full tournament once and return the champion team name (str).
    Uses global TEAMS and RATINGS. Assumes len(TEAMS) is a power of two.
    """
    n = len(TEAMS)
    assert n > 1 and (n & (n - 1) == 0), "Number of teams must be a power of two."
    # First round pairing: standard 1vN, 2vN-1, ...
    current = _seed_order(TEAMS)
    while len(current) > 1:
        current = _play_round(current)
    # current holds one (seed, name)
    return current[0][1]


def run_large_simulation(n_runs=1_000_000, batch_size=100_000, out_path="lacrosse_predictor_1M.xlsx"):
    """
    Run many tournament simulations, aggregate champions, and save a table to Excel.
    Handles the case when n_runs is not an exact multiple of batch_size.
    """
    assert batch_size > 0, "batch_size must be positive"
    all_results = Counter()

    full_batches = n_runs // batch_size
    remainder = n_runs % batch_size

    for _ in range(full_batches):
        batch = [simulate_bracket() for _ in range(batch_size)]
        all_results.update(batch)

    if remainder:
        batch = [simulate_bracket() for _ in range(remainder)]
        all_results.update(batch)

    df = pd.DataFrame([
        {"Team": team, "Championships": count, "Win %": round(100 * count / n_runs, 2)}
        for team, count in sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    ])
    df.to_excel(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    # Example: 100k runs, batches of 1k
    run_large_simulation(n_runs=100_000, batch_size=1_000, out_path="lacrosse_predictor_100k.xlsx")
