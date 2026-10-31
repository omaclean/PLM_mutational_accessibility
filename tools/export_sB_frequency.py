"""Export how *often* each replacement was observed, not merely whether it was.

    python tools/export_sB_frequency.py --source-repo ../ --results results/h3n2_13lineage

``replacement_observed.tsv.gz`` carries a 0/1 flag. That is enough to ask whether a model
separates the replacements that happened from the ones that did not, which is what every
panel of the ranking figure asked until now. It cannot ask the next question: among the
replacements that did happen, does the score pick the ones that went on to sweep?

The frequency that answers it lives only in the exploration archive's
``combined_long_table.csv``, which ``MANIFEST.tsv`` already names as the source of
``replacement_observed.tsv.gz``. This repository is published on its own, so a figure may not
reach back into the archive at run time; the column is therefore derived once, here, and
shipped beside the flag it refines.

``obs_freq`` is the fraction of the lineage's sequences carrying the replacement and ``depth``
is how many sequences that fraction is out of, so a reader can recover the count. Neither
depends on the model, so the table holds one row per replacement rather than one per model.

Four properties are checked rather than assumed, because each is a way the table could be
quietly wrong:

* ``obs_freq`` and ``depth`` do not depend on the model. Storing them once is only sound if
  the archive's two model blocks agree, so that is tested rather than taken on trust.
* ``obs_freq > 0`` exactly where the shipped ``obs_present`` is 1, on every row and in both
  directions. A replacement observed at frequency zero, or unobserved at a positive
  frequency, would mean the two tables describe different runs.
* the two tables cover the same replacements: none added here, none dropped.
* ``obs_freq`` is a frequency in [0, 1] and ``depth`` is not negative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_base import Writer, sha256  # noqa: E402

#: The archive table, relative to the exploration repository root (``--source-repo``).
SOURCE = Path("results/nextstrain_13lin_dense/tables/combined_long_table.csv")
#: The flag table this one refines; read back from ``--results`` and checked against.
OBSERVED = "replacement_observed.tsv.gz"
DEST = "replacement_frequency.tsv.gz"
KEY = ["lineage", "position", "aa"]
#: Columns read from the archive. Everything else there is per-model bookkeeping.
COLUMNS = KEY + ["model", "obs_freq", "obs_present", "depth"]
WHY = ("Fig 3 panel C, SB panel F and the lineage panel: how often each replacement was "
       "observed in each lineage, and the sequence depth behind it")


def model_independent(frame: pd.DataFrame, column: str) -> None:
    """Raise unless every (lineage, position, aa) carries one value of ``column``.

    The table is stored once per replacement rather than once per model. That is only
    correct if the models agree, so it is checked instead of assumed.
    """
    spread = frame.groupby(KEY, sort=False)[column].nunique()
    if (spread > 1).any():
        raise ValueError(
            f"{int((spread > 1).sum())} replacements carry more than one {column!r} across "
            f"models; it is not a property of the sequences alone and cannot be stored once"
        )


def collapse(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per replacement: its observed frequency and the depth behind it."""
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"the archive table is missing {missing}")
    if frame.duplicated(KEY + ["model"]).any():
        raise ValueError("the archive table has duplicate (model, lineage, position, aa) rows")
    for column in ("obs_freq", "depth", "obs_present"):
        model_independent(frame, column)
    out = frame.drop_duplicates(KEY)[KEY + ["obs_freq", "depth"]].copy()
    if not out["obs_freq"].between(0.0, 1.0).all():
        raise ValueError("obs_freq is outside [0, 1]; it is meant to be a frequency, not a count")
    if not (out["depth"] >= 0).all():
        raise ValueError("depth is negative")
    return out.sort_values(KEY).reset_index(drop=True)


def check_against_observed(frequency: pd.DataFrame, observed: pd.DataFrame) -> int:
    """``obs_freq > 0`` must agree with the shipped flag on every row, both ways.

    Returns the number of rows checked, so the caller can say how many rather than claim it.
    """
    if observed.duplicated(KEY).any():
        raise ValueError(f"{OBSERVED} has duplicate (lineage, position, aa) keys")
    both = frequency.merge(observed, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (both["_merge"] != "both").any():
        counts = both["_merge"].value_counts()
        raise ValueError(
            f"the two tables cover different replacements: {int(counts.get('left_only', 0))} only "
            f"in the new frequency table, {int(counts.get('right_only', 0))} only in {OBSERVED}"
        )
    disagree = (both["obs_freq"] > 0) != (both["obs_present"] > 0)
    if disagree.any():
        rows = both.loc[disagree, KEY + ["obs_freq", "obs_present"]].head(5)
        raise ValueError(
            f"{int(disagree.sum())} replacements disagree between obs_freq and the shipped "
            f"obs_present flag; first few:\n{rows.to_string(index=False)}"
        )
    return len(both)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-repo", type=Path, required=True,
                        help="root of the exploration repository holding the archive table")
    parser.add_argument("--results", type=Path, required=True,
                        help="results directory to write into; must already hold " + OBSERVED)
    args = parser.parse_args(argv)

    source = args.source_repo / SOURCE
    if not source.is_file():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 1
    observed_path = args.results / OBSERVED
    if not observed_path.is_file():
        print(f"missing {OBSERVED} in {args.results}; export it first", file=sys.stderr)
        return 1

    frequency = collapse(pd.read_csv(source, usecols=COLUMNS))
    checked = check_against_observed(frequency, pd.read_csv(observed_path, sep="\t"))

    nonzero = frequency.loc[frequency["obs_freq"] > 0]
    print(f"{checked:,} replacements, {len(nonzero):,} of them observed at all; obs_freq > 0 "
          f"agrees with {OBSERVED} on every one")
    print("observed replacements per lineage: "
          + ", ".join(f"{k}={v}" for k, v in nonzero.groupby("lineage").size().items()))

    writer = Writer(args.results)
    writer.add(DEST, frequency, SOURCE.as_posix(), WHY, sha=sha256(source))
    writer.write_manifest()
    print(args.results / DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
