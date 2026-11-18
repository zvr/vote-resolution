#!/usr/bin/env python3
"""Tally preferential votes from a text file (or stdin).

Each line has the form:  Voter: choice1, choice2, ...   (best to worst)
Blank lines and lines starting with '#' are ignored. Choices left off a
ballot are treated as ranked below all the choices that appear on it.

Two counting methods are available:
  schulze  a Condorcet method (default); with several seats, the top N of
           the Schulze ranking are elected.
  stv      single transferable vote with a Droop quota and fractional
           (Gregory) surplus transfers. With one seat this is the
           instant-runoff method ('irv' is accepted as an alias).
"""

# Copyright (C) 2026, Alexios Zavras
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import random
import sys
from collections import Counter
from fractions import Fraction


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def clean(text):
    """Trim and collapse internal whitespace."""
    return " ".join(text.split())


def parse_votes(lines, source, sep=",", case_sensitive=False, lenient=False):
    """Return a list of (voter, [choices]).

    Names are compared ignoring case unless case_sensitive is set; the first
    spelling seen of a choice is used everywhere. Malformed lines are fatal
    unless lenient is set, in which case they are reported and skipped.
    """
    def problem(lineno, msg, action):
        if not lenient:
            sys.exit(f"error: {source}:{lineno}: {msg}")
        warn(f"{source}:{lineno}: {msg} ({action})")

    def key(name):
        name = clean(name)
        return name if case_sensitive else name.casefold()

    ballots = []
    voters = set()
    spelling = {}  # normalized choice -> first spelling seen
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        voter, colon, rest = line.partition(":")
        if not colon:
            problem(lineno, "missing ':' separator", "line skipped")
            continue
        if not key(voter):
            problem(lineno, "empty voter name", "line skipped")
            continue
        if key(voter) in voters:
            problem(lineno, f"duplicate vote from '{clean(voter)}'", "line skipped")
            continue
        choices, seen = [], set()
        for raw in rest.split(sep):
            if not key(raw):
                continue
            if key(raw) in seen:
                problem(lineno, f"'{clean(voter)}' lists '{clean(raw)}' more than once",
                        "repeat ignored")
                continue
            seen.add(key(raw))
            choices.append(spelling.setdefault(key(raw), clean(raw)))
        voters.add(key(voter))
        ballots.append((clean(voter), choices))
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
    """Return (tiers, strongest paths); tiers are lists of candidates, best first."""
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
    return [sorted(tiers[w]) for w in sorted(tiers, reverse=True)], p


class TieBreaker:
    """Deterministic tie-breaking.

    Ties are resolved by earlier rounds (for eliminations), then by the
    head-to-head score (total number of pairwise preferences), and finally by
    a random draw if a seed was given or by alphabetical order otherwise.
    """

    def __init__(self, strength, seed=None):
        self.strength = strength
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else None

    @property
    def fallback(self):
        return f"random draw (seed {self.seed})" if self.rng else "alphabetical order"

    def rank_tied(self, tied):
        """Order tied candidates best first."""
        tied = sorted(tied)
        if self.rng:
            self.rng.shuffle(tied)
        return sorted(tied, key=lambda c: -self.strength[c])  # stable

    def pick_lowest(self, tied, history):
        """Choose which of the tied candidates to eliminate; returns (choice, reason)."""
        tied = sorted(tied)
        for counts in reversed(history):
            low = min(counts[c] for c in tied)
            tied = [c for c in tied if counts[c] == low]
            if len(tied) == 1:
                return tied[0], "fewest votes in an earlier round"
        low = min(self.strength[c] for c in tied)
        lowest = [c for c in tied if self.strength[c] == low]
        if len(lowest) == 1:
            return lowest[0], "lowest head-to-head score"
        return self.rank_tied(lowest)[-1], self.fallback


def fmt(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{float(x):.2f}"


def stv(ballots, candidates, seats, tb, progress):
    """Single transferable vote; returns the elected candidates in order.

    Each round, every ballot counts for its highest-ranked hopeful choice. A
    choice with more than live/(seats left + 1) votes is elected and its
    surplus is passed on proportionally; if nobody qualifies, the weakest
    choice is eliminated. With one seat this is instant-runoff voting.
    """
    weights = [Fraction(1)] * len(ballots)
    hopeful = set(candidates)
    elected = []
    history = []
    rnd = 0
    say = print if progress else (lambda *a: None)
    while len(elected) < seats:
        rnd += 1
        seats_left = seats - len(elected)
        counts = {c: Fraction(0) for c in hopeful}
        owners = [None] * len(ballots)
        for i, (_, choices) in enumerate(ballots):
            for c in choices:
                if c in hopeful:
                    counts[c] += weights[i]
                    owners[i] = c
                    break
        live = sum(counts.values())
        quota = live / (seats_left + 1)
        say(f"Round {rnd}: {fmt(live)} live votes, {fmt(len(ballots) - live)} "
            f"exhausted, quota {fmt(quota)}, {seats_left} seat(s) left")
        for c in sorted(counts, key=lambda c: (-counts[c], c)):
            say(f"  {c:<20} {fmt(counts[c])}")

        winners = sorted((c for c in hopeful if counts[c] > quota),
                         key=lambda c: (-counts[c], c))
        if winners:
            for c in winners:
                ratio = (counts[c] - quota) / counts[c]
                for i, o in enumerate(owners):
                    if o == c:
                        weights[i] *= ratio
                hopeful.remove(c)
                elected.append(c)
                say(f"  -> elected: {c} (surplus {fmt(counts[c] - quota)} transferred)")
        elif len(hopeful) <= seats_left:
            rest = sorted(hopeful, key=lambda c: (-counts[c], c))
            elected.extend(rest)
            say(f"  -> remaining choices elected: {', '.join(rest)}")
            break
        else:
            low = min(counts.values())
            tied = [c for c in counts if counts[c] == low]
            out, reason = tb.pick_lowest(tied, history)
            note = (f" (tie among {', '.join(sorted(tied))}, broken by {reason})"
                    if len(tied) > 1 else "")
            say(f"  -> eliminated: {out}{note}")
            hopeful.remove(out)
        history.append(counts)
    return elected


def print_matrix(candidates, m):
    width = max(len(c) for c in candidates) + 2
    print(" " * width + "".join(f"{c[:width - 1]:>{width}}" for c in candidates))
    for a in candidates:
        print(f"{a:<{width}}" + "".join(
            f"{'-' if a == b else m[a][b]:>{width}}" for b in candidates))


def main():
    parser = argparse.ArgumentParser(description="Tally preferential votes.")
    parser.add_argument("file", nargs="?", default="-",
                        help="votes file, one 'Voter: a, b, c' per line "
                             "(default or '-': standard input)")
    parser.add_argument("-m", "--method", choices=["schulze", "stv", "irv"],
                        default="schulze",
                        help="counting method (default: schulze; irv is an alias of stv)")
    parser.add_argument("-n", "--seats", type=int, default=1, metavar="N",
                        help="number of winners to elect (default: 1)")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="show the steps of the chosen method "
                             "(stv: the rounds; schulze: the strongest paths)")
    parser.add_argument("-s", "--separator", default=",", metavar="C",
                        help="character separating choices on a line (default: ',')")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="seed for random tie-breaking "
                             "(default: break remaining ties alphabetically)")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="treat names that differ only in case as different")
    parser.add_argument("--lenient", action="store_true",
                        help="warn about and skip malformed lines instead of exiting")
    args = parser.parse_args()
    if len(args.separator) != 1 or args.separator == ":":
        parser.error("separator must be a single character other than ':'")
    if args.seats < 1:
        parser.error("seats must be at least 1")
    method = "stv" if args.method == "irv" else args.method

    if args.file == "-":
        if sys.stdin.isatty():
            parser.error("no input file given and standard input is a terminal")
        ballots = parse_votes(sys.stdin, "<stdin>", args.separator,
                              args.case_sensitive, args.lenient)
    else:
        try:
            with open(args.file, encoding="utf-8") as f:
                ballots = parse_votes(f, args.file, args.separator,
                                      args.case_sensitive, args.lenient)
        except OSError as e:
            sys.exit(f"error: {e}")
    if not ballots:
        sys.exit("error: no votes found")

    candidates = sorted({c for _, ch in ballots for c in ch})
    if not candidates:
        sys.exit("error: no choices found")
    if args.seats > len(candidates):
        sys.exit(f"error: {args.seats} seats requested but only "
                 f"{len(candidates)} choices")

    appearances = Counter(c for _, ch in ballots for c in ch)
    if len(ballots) > 1:
        for c in candidates:
            if appearances[c] == 1:
                warn(f"'{c}' appears on only one ballot (typo?)")

    print("=== Statistics ===")
    print(f"Voters:     {len(ballots)}")
    print(f"Choices:    {len(candidates)} ({', '.join(candidates)})")
    partial = sum(1 for _, ch in ballots if len(ch) < len(candidates))
    print(f"Incomplete ballots: {partial}")
    print(f"Seats:      {args.seats}")

    firsts = Counter(ch[0] for _, ch in ballots if ch)
    print("\n=== First-choice votes ===")
    for c in sorted(candidates, key=lambda c: (-firsts[c], c)):
        print(f"{c:<20} {firsts[c]}")

    d = pairwise(ballots, candidates)
    print("\n=== Head-to-head (row preferred over column) ===")
    print_matrix(candidates, d)

    tb = TieBreaker({a: sum(d[a].values()) for a in candidates}, args.seed)

    if method == "schulze":
        tiers, paths = schulze(d, candidates)
        if args.progress:
            print("\n=== Progress (strongest paths, row to column) ===")
            print_matrix(candidates, paths)
        print("\n=== Result (Schulze method) ===")
        place = 1
        for tier in tiers:
            print(f"{place}. {', '.join(tier)}" + (" (tie)" if len(tier) > 1 else ""))
            place += len(tier)
        winners = []
        for tier in tiers:
            need = args.seats - len(winners)
            if len(tier) <= need:
                winners += tier
            else:
                winners += tb.rank_tied(tier)[:need]
                print(f"Tie for the last seat among {', '.join(tier)}: "
                      f"broken by head-to-head score, then {tb.fallback}")
            if len(winners) == args.seats:
                break
    else:
        label = "instant-runoff" if args.seats == 1 else f"{args.seats}-seat"
        if args.progress:
            print(f"\n=== Progress ({label} rounds) ===")
        winners = stv(ballots, candidates, args.seats, tb, args.progress)
        print("\n=== Result (STV) ===")

    if args.seats == 1:
        print(f"\nWinner: {winners[0]}")
    else:
        print(f"\nElected ({args.seats} seats): {', '.join(winners)}")


if __name__ == "__main__":
    main()
