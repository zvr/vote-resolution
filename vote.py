#!/usr/bin/env python3
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# Copyright (C) 2026, Alexios Zavras
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tally preferential votes from a text file (or stdin).

Each line has the form:  Voter: choice1, choice2, ...   (best to worst)
Blank lines and lines starting with '#' are ignored.
Choices left off a ballot are treated as ranked
below all the choices that appear on it.

Two counting methods are available:
  schulze  a Condorcet method (default).  With several seats,
           the top N of the Schulze ranking are elected.
  stv      single transferable vote with a Droop quota and
           fractional (Gregory) surplus transfers.  With one seat
           this is the instant-runoff method.
           'irv' is accepted as an alias.
"""

import argparse
import contextlib
import io
import pathlib
import random
import re
import sys
from collections import Counter
from fractions import Fraction


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def clean(text):
    """Trim and collapse internal whitespace."""
    return " ".join(text.split())


def parse_votes(lines, source, sep=",", *, case_sensitive=False, lenient=False):
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
    for lineno, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
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
                problem(lineno, f"'{clean(voter)}' lists '{clean(raw)}' more than once", "repeat ignored")
                continue
            seen.add(key(raw))
            choices.append(spelling.setdefault(key(raw), clean(raw)))
        voters.add(key(voter))
        ballots.append((clean(voter), choices))
    return ballots


def pairwise(ballots, candidates):
    """d[a][b] = number of voters who prefer a over b."""
    d = {a: dict.fromkeys(candidates, 0) for a in candidates}
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
    p = {a: dict.fromkeys(candidates, 0) for a in candidates}
    for a in candidates:
        for b in candidates:
            if a != b and d[a][b] > d[b][a]:
                p[a][b] = d[a][b]
    for k in candidates:
        for i in candidates:
            if i == k:
                continue
            for j in candidates:
                if j in {i, k}:
                    continue
                p[i][j] = max(p[i][j], min(p[i][k], p[k][j]))
    wins = {a: sum(1 for b in candidates if a != b and p[a][b] > p[b][a]) for a in candidates}
    tiers = {}
    for a, w in wins.items():
        tiers.setdefault(w, []).append(a)
    return [sorted(tiers[w]) for w in sorted(tiers, reverse=True)], p


class TieBreaker:
    """Deterministic tie-breaking.

    Ties are resolved by earlier rounds (for eliminations),
    then by the head-to-head score (total number of pairwise preferences),
    and finally by a random draw if a seed was given or by alphabetical order otherwise.
    """

    def __init__(self, strength, seed=None):
        self.strength = strength
        self.seed = seed
        # tie-breaking is not security-sensitive; a seedable PRNG is what we want
        self.rng = random.Random(seed) if seed is not None else None  # ruff: ignore[suspicious-non-cryptographic-random-usage]

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


def esc(text):
    """Escape characters that are special in Markdown."""
    return re.sub(r"([\\`*_|<>\[\]#])", r"\\\1", text)


def names(items):
    return ", ".join(esc(i) for i in items)


def heading(level, text):
    print(f"{'#' * level} {text}\n")


def table(headers, rows, align=None):
    """Print a Markdown table; align has one of 'l'/'r' per column."""
    align = align or "l" * len(headers)
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("--:" if a == "r" else ":--" for a in align) + "|")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")
    print()


def stv(ballots, candidates, seats, tb, progress):
    """Single transferable vote; returns the elected candidates in order.

    Each round, every ballot counts for its highest-ranked hopeful choice.
    A choice with more than live/(seats left + 1) votes is elected and
    its surplus is passed on proportionally; if nobody qualifies,
    the weakest choice is eliminated.
    With one seat this is instant-runoff voting.
    """
    weights = [Fraction(1)] * len(ballots)
    hopeful = set(candidates)
    elected = []
    history = []
    rnd = 0
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
        if progress:
            heading(3, f"Round {rnd}")
            print(
                f"{fmt(live)} live votes, {fmt(len(ballots) - live)} exhausted, "
                f"quota {fmt(quota)}, {seats_left} seat(s) left\n"
            )
            table(
                ["Choice", "Votes"],
                [[esc(c), fmt(counts[c])] for c in sorted(counts, key=lambda c: (-counts[c], c))],
                "lr",
            )
        events = []

        winners = sorted((c for c in hopeful if counts[c] > quota), key=lambda c: (-counts[c], c))
        if winners:
            for c in winners:
                ratio = (counts[c] - quota) / counts[c]
                for i, o in enumerate(owners):
                    if o == c:
                        weights[i] *= ratio
                hopeful.remove(c)
                elected.append(c)
                events.append(f"**Elected:** {esc(c)} (surplus {fmt(counts[c] - quota)} transferred)")
        elif len(hopeful) <= seats_left:
            rest = sorted(hopeful, key=lambda c: (-counts[c], c))
            elected.extend(rest)
            events.append(f"**Remaining choices elected:** {names(rest)}")
        else:
            low = min(counts.values())
            tied = [c for c in counts if counts[c] == low]
            out, reason = tb.pick_lowest(tied, history)
            note = f" (tie among {names(sorted(tied))}, broken by {reason})" if len(tied) > 1 else ""
            events.append(f"**Eliminated:** {esc(out)}{note}")
            hopeful.remove(out)
        if progress:
            print("\n".join(f"- {e}" for e in events) + "\n")
        history.append(counts)
    return elected


def print_matrix(candidates, m):
    table(
        [""] + [esc(c) for c in candidates],
        [[f"**{esc(a)}**"] + ["-" if a == b else m[a][b] for b in candidates] for a in candidates],
        "l" + "r" * len(candidates),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Tally preferential votes.")
    parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help="votes file, one 'Voter: a, b, c' per line (default or '-': standard input)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["schulze", "stv", "irv", "all"],
        default="schulze",
        help="counting method (default: schulze; irv is an alias of stv; all runs every method and summarizes)",
    )
    parser.add_argument(
        "-n", "--seats", type=int, default=1, metavar="N", help="number of winners to elect (default: 1)"
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="show the steps of the chosen method (stv: the rounds; schulze: the strongest paths)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="print only the winner(s), one per line (overrides -p)"
    )
    parser.add_argument("-o", "--output", metavar="FILE", help="write the results to FILE instead of standard output")
    parser.add_argument(
        "-s", "--separator", default=",", metavar="C", help="character separating choices on a line (default: ',')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="seed for random tie-breaking (default: break remaining ties alphabetically)",
    )
    parser.add_argument(
        "--case-sensitive", action="store_true", help="treat names that differ only in case as different"
    )
    parser.add_argument("--lenient", action="store_true", help="warn about and skip malformed lines instead of exiting")
    args = parser.parse_args()
    if len(args.separator) != 1 or args.separator == ":":
        parser.error("separator must be a single character other than ':'")
    if args.seats < 1:
        parser.error("seats must be at least 1")
    if args.file == "-" and sys.stdin.isatty():
        parser.error("no input file given and standard input is a terminal")
    return args


def load_ballots(args):
    """Read and validate the ballots; return (ballots, candidates)."""
    options = {"case_sensitive": args.case_sensitive, "lenient": args.lenient}
    if args.file == "-":
        ballots = parse_votes(sys.stdin, "<stdin>", args.separator, **options)
    else:
        try:
            with pathlib.Path(args.file).open(encoding="utf-8") as f:
                ballots = parse_votes(f, args.file, args.separator, **options)
        except OSError as e:
            sys.exit(f"error: {e}")
    if not ballots:
        sys.exit("error: no votes found")

    candidates = sorted({c for _, ch in ballots for c in ch})
    if not candidates:
        sys.exit("error: no choices found")
    if args.seats > len(candidates):
        sys.exit(f"error: {args.seats} seats requested but only {len(candidates)} choices")

    appearances = Counter(c for _, ch in ballots for c in ch)
    if len(ballots) > 1:
        for c in candidates:
            if appearances[c] == 1:
                warn(f"'{c}' appears on only one ballot (typo?)")
    return ballots, candidates


def main():
    args = parse_args()
    method = "stv" if args.method == "irv" else args.method
    ballots, candidates = load_ballots(args)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = report(args, method, ballots, candidates)
    if not args.quiet:
        text = buf.getvalue().rstrip("\n") + "\n"
    elif method == "all":
        text = "".join(f"{m}: {', '.join(w)}\n" for m, w in results.items())
    else:
        text = "".join(f"{w}\n" for w in results[method])
    if args.output:
        try:
            pathlib.Path(args.output).write_text(text, encoding="utf-8")
        except OSError as e:
            sys.exit(f"error: {e}")
    else:
        sys.stdout.write(text)


def report(args, method, ballots, candidates):
    """Print the full report to stdout; return {method: winners}."""
    print("# Statistics\n")
    print(f"- **Voters:** {len(ballots)}")
    print(f"- **Choices:** {len(candidates)} ({names(candidates)})")
    partial = sum(1 for _, ch in ballots if len(ch) < len(candidates))
    print(f"- **Incomplete ballots:** {partial}")
    print(f"- **Seats:** {args.seats}\n")

    firsts = Counter(ch[0] for _, ch in ballots if ch)
    heading(1, "First-choice votes")
    table(["Choice", "Votes"], [[esc(c), firsts[c]] for c in sorted(candidates, key=lambda c: (-firsts[c], c))], "lr")

    d = pairwise(ballots, candidates)
    heading(1, "Head-to-head (row preferred over column)")
    print_matrix(candidates, d)

    results = {}
    for m in ("schulze", "stv") if method == "all" else (method,):
        # a fresh tie-breaker per method keeps seeded draws reproducible
        tb = TieBreaker({a: sum(d[a].values()) for a in candidates}, args.seed)
        if m == "schulze":
            results[m] = run_schulze(args, d, candidates, tb)
        else:
            results[m] = run_stv(args, ballots, candidates, tb)
    if method == "all":
        heading(1, "Summary")
        table(["Method", "Winner(s)"], [[m, names(winners)] for m, winners in results.items()])
    return results


def run_schulze(args, d, candidates, tb):
    tiers, paths = schulze(d, candidates)
    if args.progress:
        heading(2, "Progress (Schulze strongest paths, row to column)")
        print_matrix(candidates, paths)
    heading(2, "Result (Schulze method)")
    place = 1
    for tier in tiers:
        print(f"- **{place}.** {names(tier)}" + (" (tie)" if len(tier) > 1 else ""))
        place += len(tier)
    print()
    winners = []
    for tier in tiers:
        need = args.seats - len(winners)
        if len(tier) <= need:
            winners += tier
        else:
            winners += tb.rank_tied(tier)[:need]
            print(f"Tie for the last seat among {names(tier)}: broken by head-to-head score, then {tb.fallback}.\n")
        if len(winners) == args.seats:
            break
    print_winners(args, winners)
    return winners


def run_stv(args, ballots, candidates, tb):
    label = "instant-runoff" if args.seats == 1 else f"{args.seats}-seat"
    if args.progress:
        heading(2, f"Progress (STV {label} rounds)")
    winners = stv(ballots, candidates, args.seats, tb, args.progress)
    heading(2, "Result (STV)")
    print_winners(args, winners)
    return winners


def print_winners(args, winners):
    if args.seats == 1:
        print(f"**Winner:** {names(winners)}\n")
    else:
        print(f"**Elected ({args.seats} seats):** {names(winners)}\n")


if __name__ == "__main__":
    main()
