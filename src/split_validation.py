from __future__ import annotations

from hashlib import sha256


def validate_disjoint_samples(splits):
    """Reject identical serialized clean volumes in different splits."""
    seen = {}
    for split_name, samples in splits.items():
        for sample in samples:
            digest = sha256(sample).digest()
            previous = seen.setdefault(digest, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Identical clean volumes occur in {previous!r} and {split_name!r}"
                )
