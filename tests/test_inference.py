import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dajaba_submission.dl.graph import form_coverage, graph, key
from dajaba_submission.dl.inference import TransactionScorer
from dajaba_submission.dl.model import GINe
from dajaba_submission.io import sha256

CONFIG = Path(__file__).resolve().parents[1] / "config/synthetic.example.json"


def make_row(i, ts, a, b):
    return {"source_file": "HI-Small_Trans.csv", "source_row_number": i, "ts": ts,
            "from_account_key": a, "to_account_key": b, "amount_received": "5.25",
            "currency_received": "US Dollar", "payment_format": "Wire"}


def test_time_bounds_and_counts_use_all_history():
    target = make_row(30, "2022-09-07T12:00", "A", "B")
    records = [make_row(1, "2022-09-06T11:59", "A", "X"),
               make_row(2, "2022-09-06T12:00", "A", "X")]
    records += [make_row(i + 3, f"2022-09-07T11:{i:02d}", "A", "X") for i in range(20)]
    records += [make_row(29, "2022-09-07T12:00", "Y", "B"), target]
    edges, context, targets, counts = graph(records, [key(target)], json.loads(CONFIG.read_text()))
    [idx] = targets
    assert counts[idx].tolist() == [21, 0, 0, 0]
    assert context[idx].shape == (16,)
    assert edges["row"][context[idx]].tolist() == list(range(21, 5, -1))


def test_missing_receiver_coverage_is_rejected():
    target = make_row(1, "2022-09-07T12:00", "A", "B")
    coverage = dict(target, history_start="2022-09-06T12:00", history_end=target["ts"],
                    history_complete="true", sender_complete="true", receiver_complete="false")
    assert not form_coverage(coverage, target)


def test_synthetic_checkpoint_cpu_inference(tmp_path):
    config = json.loads(CONFIG.read_text())
    model = GINe(num_features=1, num_gnn_layers=2, n_classes=2, n_hidden=32,
                 edge_updates=False, edge_dim=config["edge_dim"], final_dropout=0.0)
    for p in model.parameters():
        p.data.zero_()
    model_file = tmp_path / "synthetic.pt"
    torch.save(model.state_dict(), model_file)
    config["model_sha256"] = sha256(model_file)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    target = make_row(2, "2022-09-07T12:00", "A", "B")
    inputs = graph([make_row(1, "2022-09-07T11:59", "C", "B"), target], [key(target)], config)
    scorer = TransactionScorer(model_file, config_path, "cpu")
    np.testing.assert_array_equal(scorer.score(*inputs), [0.5])
    config_path.write_text(json.dumps(dict(config, model_sha256="0" * 64)))
    with pytest.raises(ValueError, match="Unexpected GINe checkpoint"):
        TransactionScorer(model_file, config_path, "cpu")
