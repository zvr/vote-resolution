#!/usr/bin/env python3
"""Tally preferential votes from a text file.

Each line has the form:  Voter: choice1, choice2, ...   (best to worst)
Blank lines and lines starting with '#' are ignored. Choices left off a
ballot are treated as ranked below all the choices that appear on it.

The winner is computed with the Schulze method (a Condorcet method).
"""
import argparse
import sys
from collections import Counter


def parse_votes(path, sep=","):
    ballots = []  # list of (voter, [choices])
    voters = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                sys.exit(f"{path}:{lineno}: missing ':' separator")
            voter, _, rest = line.partition(":")
            voter = voter.strip()
            choices = [c.strip() for c in rest.split(sep) if c.strip()]
            if not voter:
                sys.exit(f"{path}:{lineno}: empty voter name")
            if voter in voters:
                sys.exit(f"{path}:{lineno}: duplicate vote from '{voter}'")
            if len(set(choices)) != len(choices):
                sys.exit(f"{path}:{lineno}: '{voter}' lists a choice more than once")
            voters.add(voter)
            ballots.append((voter, choices))
    return ballots


def pairwise(ballots, candidates):
    """d[a][b] = number of voters who prefer a over b."""
    d = {a: {b: 0 for b in candidates} for a in candidates}
    for _, choices in ballots:
        rank = {c: i for i, c in enumerate(choices)}
        unranked = len(choices)
        for a in candidates:
            for b in candidates:
                if a != b and rank.get(a, unranked) < rank.get(b, unranked):
                    d[a][b] += 1
    return d


def schulze(d, candidates):
    """Return candidates grouped into ranked tiers (best first)."""
    p = {a: {b: 0 for b in candidates} for a in candidates}
    for a in candidates:
        for b in candidates:
            if a != b and d[a][b] > d[b][a]:
                p[a][b] = d[a][b]
    for k in candidates:
        for i in candidates:
            if i == k:
                continue
            for j in candidates:
                if j == i or j == k:
                    continue
                p[i][j] = max(p[i][j], min(p[i][k], p[k][j]))
    wins = {a: sum(1 for b in candidates if a != b and p[a][b] > p[b][a])
            for a in candidates}
    tiers = {}
    for a, w in wins.items():
        tiers.setdefault(w, []).append(a)
    return [sorted(tiers[w]) for w in sorted(tiers, reverse=True)]


def instant_runoff(ballots, candidates):
    """Print each elimination round (instant-runoff / IRV).

    Each round counts every ballot for its highest-ranked remaining choice.
    A choice with more than half of the live ballots wins; otherwise the
    choice(s) with the fewest votes are eliminated. If all remaining choices
    are tied, they share the win.
    """
    remaining = set(candidates)
    rnd = 1
    while True:
        counts = {c: 0 for c in remaining}
        for _, choices in ballots:
            for c in choices:
                if c in remaining:
                    counts[c] += 1
                    break
        live = sum(counts.values())
        print(f"Round {rnd}: ({live} live ballots, {len(ballots) - live} exhausted)")
        for c in sorted(counts, key=lambda c: (-counts[c], c)):
            print(f"  {c:<20} {counts[c]}")
        top = max(counts.values())
        low = min(counts.values())
        if top * 2 > live:
            winners = [c for c in counts if counts[c] == top]
            print(f"  -> {winners[0]} has a majority")
            return winners
        if top == low:
            winners = sorted(counts)
            print(f"  -> all remaining choices tied: {', '.join(winners)}")
            return winners
        out = sorted(c for c in counts if counts[c] == low)
        print(f"  -> eliminated: {', '.join(out)}")
        remaining -= set(out)
        rnd += 1


def main():
    parser = argparse.ArgumentParser(description="Tally preferential votes.")
    parser.add_argument("file", help="votes file (one 'Voter: a, b, c' per line)")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="print the instant-runoff elimination rounds")
    parser.add_argument("-s", "--separator", default=",", metavar="C",
                        help="character separating choices on a line (default: ',')")
    args = parser.parse_args()
    if len(args.separator) != 1 or args.separator == ":":
        parser.error("separator must be a single character other than ':'")
    try:
        ballots = parse_votes(args.file, args.separator)
    except OSError as e:
        sys.exit(f"error: {e}")
    if not ballots:
        sys.exit("error: no votes found")

    candidates = sorted({c for _, ch in ballots for c in ch})
    if not candidates:
        sys.exit("error: no choices found")

    print("=== Statistics ===")
    print(f"Voters:     {len(ballots)}")
    print(f"Choices:    {len(candidates)} ({', '.join(candidates)})")
    partial = sum(1 for _, ch in ballots if len(ch) < len(candidates))
    print(f"Incomplete ballots: {partial}")

    firsts = Counter(ch[0] for _, ch in ballots if ch)
    print("\n=== First-choice votes ===")
    for c in sorted(candidates, key=lambda c: (-firsts[c], c)):
        print(f"{c:<20} {firsts[c]}")

    if args.progress:
        print("\n=== Progress (instant-runoff rounds) ===")
        instant_runoff(ballots, candidates)

    d = pairwise(ballots, candidates)
    print("\n=== Head-to-head (row preferred over column) ===")
    width = max(len(c) for c in candidates) + 2
    print(" " * width + "".join(f"{c[:width - 1]:>{width}}" for c in candidates))
    for a in candidates:
        print(f"{a:<{width}}" + "".join(
            f"{'-' if a == b else d[a][b]:>{width}}" for b in candidates))

    tiers = schulze(d, candidates)
    print("\n=== Result (Schulze method) ===")
    place = 1
    for tier in tiers:
        print(f"{place}. {', '.join(tier)}" + (" (tie)" if len(tier) > 1 else ""))
        place += len(tier)

    winners = tiers[0]
    if len(winners) == 1:
        print(f"\nWinner: {winners[0]}")
    else:
        print(f"\nTie for first place: {', '.join(winners)}")


if __name__ == "__main__":
    main()
