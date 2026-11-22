"""Nearest-batches member with k = 4, 12, 16 neighbours (k = 8 is members.py 'nearest'), for nested ensembling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402


def nearest(k):
    def fp(train_ids, ids):
        table = X.load()[1]
        return X._sorted(X.R.NearestBatches(k=k).fit(table, train_ids).predict(table, ids))
    return fp


if __name__ == "__main__":
    for k in (4, 12, 16):
        X.report(f"nearest{k}", X.lobo(nearest(k)))
