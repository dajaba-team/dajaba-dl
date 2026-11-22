"""Frozen counts GINe B and its prepared-graph inference path."""

from pathlib import Path

import numpy as np
import torch

from ..io import read_json, require, sha256
from .model import GINe

BATCH = 2048
CONTEXT_LIMIT = 16
WINDOW = 1440


class Encoder:
    def __init__(self, config):
        self.means = np.asarray(config["means"], dtype=np.float64)
        self.std = np.asarray(config["std"], dtype=np.float64)
        self.known_currency = np.asarray(config["known_currency"], dtype=bool)
        self.known_method = np.asarray(config["known_method"], dtype=bool)
        self.method_column = 4 + len(self.means)
        self.width = self.method_column + len(self.known_method) + 1 + 4
        require(self.width == config["edge_dim"], "GINe encoder width differs")
        require(len(self.means) == len(self.std) == len(self.known_currency)
                and np.isfinite(self.means).all() and np.isfinite(self.std).all()
                and (self.std > 0).all(), "Invalid frozen normalization")

    def encode(self, rows, age):
        features = np.zeros((len(rows), self.width), dtype=np.float32)
        c, m = rows["currency"], rows["payment"]
        require(((c >= 0) & (c < len(self.means))).all() and
                ((m >= 0) & (m < len(self.known_method))).all(), "Category outside the frozen vocabulary")
        require(np.isfinite(rows["amount"]).all(), "Nonfinite received amount")
        known = self.known_currency[c]
        features[:, 0] = age / 1440.0
        features[:, 1] = np.where(known, np.clip((rows["amount"] - self.means[c]) / self.std[c], -5, 5), 0)
        features[:, 2] = ~known
        index = np.arange(len(rows))
        features[index, 3 + np.where(known, c, len(self.means))] = 1
        features[index, self.method_column + np.where(self.known_method[m], m, len(self.known_method))] = 1
        return features


def pack(edges, history, ids, encoder, device, counts):
    refs = np.concatenate((history[ids], ids[:, None]), axis=1)
    require(((refs >= -1) & (refs < len(edges))).all(), "Invalid history index")
    require(not (refs[:, :-1] < -1).any(), "Invalid history padding")
    valid = refs >= 0
    width = refs.shape[1]
    rows = edges[np.maximum(refs, 0)]
    endpoints = np.concatenate((rows["src"], rows["dst"]), axis=1)
    endpoint_valid = np.concatenate((valid, valid), axis=1)
    same = (endpoints[:, :, None] == endpoints[:, None, :]) & endpoint_valid[:, None, :]
    local = same.argmax(axis=2) + (np.arange(len(ids)) * 2 * width)[:, None]
    edge_index = np.stack((local[:, :width][valid], local[:, width:][valid]))
    targets = np.cumsum(valid.sum(axis=1)) - 1
    delta = rows["minute"] - edges["minute"][ids, None]
    present = valid[:, :-1]
    require((((delta[:, :-1] < 0) & (delta[:, :-1] >= -WINDOW)) | ~present).all(),
            "History must be strictly prior and within 24 hours")
    selected = np.where(present, refs[:, :-1], -1)
    require(((selected[:, 1:] < selected[:, :-1]) | ~present[:, 1:]).all(),
            "Duplicate or out-of-order context")
    root = edges[ids]
    incident = ((rows["src"][:, :-1] == root["src"][:, None]) |
                (rows["src"][:, :-1] == root["dst"][:, None]) |
                (rows["dst"][:, :-1] == root["src"][:, None]) |
                (rows["dst"][:, :-1] == root["dst"][:, None]))
    require((incident | ~present).all(), "An unrelated account is in target history")
    attributes = encoder.encode(rows[valid], delta[valid])
    require((counts[ids] >= 0).all(), "Directional counts cannot be negative")
    attributes[targets, -4:] = np.log1p(counts[ids].astype(np.float32))
    return (
        torch.ones((len(ids) * 2 * width, 1), device=device),
        torch.as_tensor(edge_index, device=device),
        torch.as_tensor(attributes, device=device),
        torch.as_tensor(targets, device=device),
    )


class TransactionScorer:
    def __init__(self, checkpoint, config_path, device):
        self.config = read_json(config_path)
        require(sha256(checkpoint) == self.config["model_sha256"], "Unexpected GINe checkpoint")
        self.device = torch.device(device)
        require(self.device.type in {"cpu", "cuda"}, "Use cpu or cuda")
        require(self.device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        if self.device.type == "cpu":
            torch.set_num_threads(2)
        self.encoder = Encoder(self.config)
        self.model = GINe(num_features=1, num_gnn_layers=2, n_classes=2, n_hidden=32,
                          edge_updates=False, edge_dim=self.encoder.width, final_dropout=0.0).to(self.device)
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def score(self, edges, context, targets, counts, batch=BATCH):
        require(context.shape == (len(edges), CONTEXT_LIMIT) and context.dtype.kind == "i",
                "Expected signed 16-neighbor history")
        require(counts.shape == (len(edges), 4) and counts.dtype.kind in "iu",
                "Expected 4 integer directional counts")
        targets = np.asarray(targets)
        require(targets.ndim == 1 and targets.dtype.kind in "iu" and batch > 0 and
                ((targets >= 0) & (targets < len(edges))).all(), "Invalid target indexes")
        require(len(np.unique(edges["row"][targets])) == len(targets), "Duplicate transaction IDs")
        values = np.empty(len(targets), dtype=np.float32)
        for start in range(0, len(targets), batch):
            ids = targets[start:start + batch]
            x, edge_index, attributes, positions = pack(edges, context, ids, self.encoder, self.device, counts)
            values[start:start + len(ids)] = self.model(x, edge_index, attributes)[positions].softmax(
                dim=1
            )[:, 1].cpu().numpy()
        require(np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(), "Nonfinite or invalid GINe scores")
        return values
