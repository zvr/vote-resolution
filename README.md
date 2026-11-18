# vote-resolution
Resolve a voting

## Usage

    python vote.py [options] [FILE]

`FILE` has one ballot per line, `Voter: best, next, ..., worst`; omit it (or
use `-`) to read from standard input. Blank lines and `#` comments are ignored.
Choices missing from a ballot rank below all the listed ones. Names are
case-insensitive and whitespace-trimmed.

| Option | Meaning |
|---|---|
| `-m, --method {schulze,stv,irv}` | Counting method (default `schulze`; `irv` is an alias of `stv`) |
| `-n, --seats N` | Number of winners (default 1). Schulze takes the top N of its ranking; STV uses a Droop quota with fractional surplus transfers |
| `-p, --progress` | Show the steps: STV rounds, or the Schulze strongest paths |
| `-s, --separator C` | Separator between choices (default `,`) |
| `--seed N` | Random tie-breaking with this seed (default: alphabetical) |
| `--case-sensitive` | Do not fold case when comparing names |
| `--lenient` | Warn about and skip malformed lines instead of exiting |

Ties are broken by earlier rounds (eliminations), then by head-to-head score,
then by random draw (with `--seed`) or alphabetical order. Choices that appear
on only one ballot produce a warning on stderr, as they are likely typos.
