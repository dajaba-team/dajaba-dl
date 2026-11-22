"""Build strictly prior per-transaction contexts from an attested retained stream."""

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np

from ..io import require, unique_index

WIDTH = 16
HOURS = 24
EPOCH = datetime(2022, 9, 1)
EDGE = np.dtype([
    ("row", "<i8"), ("minute", "<i8"), ("src", "<i8"), ("dst", "<i8"),
    ("amount", "<f4"), ("currency", "<i2"), ("payment", "<i2"),
])


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is None and result.second == result.microsecond == 0,
            "Source timestamps must be naive and minute aligned")
    return result


def key(row):
    num = int(row["source_row_number"])
    require(num >= 1 and row["source_file"] == "HI-Small_Trans.csv", "Invalid original row key")
    return row["source_file"], num


def form_coverage(item, target):
    t = stamp(target["ts"])
    return (
        key(item) == key(target)
        and item["history_complete"] == item["sender_complete"] == item["receiver_complete"] == "true"
        and stamp(item["history_start"]) == t - timedelta(hours=HOURS)
        and stamp(item["history_end"]) == t
    )


def graph(records, target_keys, config):
    ordered = sorted(unique_index(records, key, "original transaction row").values(),
                     key=lambda r: (stamp(r["ts"]), key(r)[1]))
    raw = unique_index(ordered, key, "original transaction row")
    nodes, offsets = {}, {}
    outgoing, incoming = defaultdict(list), defaultdict(list)
    edges = np.empty(len(ordered), dtype=EDGE)
    for index, row in enumerate(ordered):
        source = key(row)
        offsets[source] = index
        a, b = row["from_account_key"], row["to_account_key"]
        require(a and b, "Missing sender or receiver")
        sender, receiver = (nodes.setdefault(acct, len(nodes)) for acct in (a, b))
        amount = Decimal(row["amount_received"])
        require(amount.is_finite(), "Missing or nonfinite received amount")
        amount = float(amount)
        minute = (stamp(row["ts"]) - EPOCH) // timedelta(minutes=1)
        c, p = row["currency_received"], row["payment_format"]
        require(c in config["currency"] and p in config["payment"], "Category outside the frozen vocabulary")
        edges[index] = (source[1] - 1, minute, sender, receiver,
                        np.sign(amount) * np.log1p(abs(amount)), config["currency"][c], config["payment"][p])
        outgoing[sender].append((minute, index))
        incoming[receiver].append((minute, index))
    counts = np.zeros((len(edges), 4), dtype=np.int32)
    context = np.full((len(edges), WIDTH), -1, dtype=np.int64)
    wanted = []
    for source in target_keys:
        require(source in raw, "Mapped target not present in the DL stream")
        index = offsets[source]
        wanted.append(index)
        minute, a, b = (int(edges[index][f]) for f in ("minute", "src", "dst"))
        candidates = set()
        for column, timeline in enumerate((outgoing[a], incoming[a], outgoing[b], incoming[b])):
            lo = bisect_left(timeline, (minute - HOURS * 60, -1))
            hi = bisect_left(timeline, (minute, -1))
            counts[index, column] = hi - lo
            candidates.update(i for _, i in timeline[max(lo, hi - WIDTH):hi])
        latest = sorted(candidates, reverse=True)[:WIDTH]
        context[index, :len(latest)] = latest
    return edges, context, np.asarray(wanted, dtype=np.int64), counts
