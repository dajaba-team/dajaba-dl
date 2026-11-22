"""Score verified original HI-Small target transactions in a separate process."""
import argparse
import os
from pathlib import Path
from time import perf_counter

from ..io import (
    read_csv,
    read_json,
    require,
    sha256,
    unique_index,
    write_csv,
    write_json,
)
from .graph import form_coverage, graph, key


def score_files(checkpoint, config, transactions, coverage, output, device):
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .inference import TransactionScorer

    records = read_csv(transactions)
    index = unique_index(records, key, "source transaction")
    approved = unique_index(read_csv(coverage), key, "target coverage")
    require(approved, "No target coverage supplied")
    for source, item in approved.items():
        require(source in index and form_coverage(item, index[source]), f"Invalid target coverage for {source!r}")
    start = perf_counter()
    scorer = TransactionScorer(checkpoint, config, device)
    edges, history, targets, counts = graph(records, list(approved), read_json(config))
    scores = scorer.score(edges, history, targets, counts)
    rows = [
        {"source_file": filename, "source_row_number": number, "score": float(score)}
        for (filename, number), score in zip(approved, scores)
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "transaction_scores.csv", rows, ["source_file", "source_row_number", "score"])
    write_json(output / "result.json", {
        "model": scorer.config["model"], "model_sha256": sha256(checkpoint),
        "device": device, "rows": len(rows), "seconds": perf_counter() - start,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "config", "transactions", "coverage", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score_files(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
