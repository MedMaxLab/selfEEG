"""Shared helpers for the selfEEG unittest suite."""

import itertools
import random

import torch

__all__ = ["get_device", "make_grid"]


def get_device(probe=None):
    """Return the best available torch device (mps > cuda > cpu).

    If a ``probe`` callable is given it is executed on the selected accelerator;
    should it fail (e.g. an operator is unsupported on that backend) the function
    silently falls back to the cpu. This mirrors the behaviour previously
    duplicated in every ``setUpClass``.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return torch.device("cpu")

    try:
        if probe is None:
            _ = torch.zeros(8, device=device) + 1
        else:
            probe(device)
    except Exception:
        device = torch.device("cpu")
    return device


def make_grid(pars_dict, max_comb=None, seed=0):
    """Build a list of keyword-argument dictionaries from a dict of value lists.

    The full Cartesian product is returned when it contains at most ``max_comb``
    elements (or when ``max_comb`` is ``None``). Otherwise a deterministic subset
    is selected that still exercises **every individual value of every parameter
    at least once**, then topped up with random combinations up to ``max_comb``.
    This keeps the "no crash / no NaN" smoke tests meaningful while avoiding the
    combinatorial explosion of testing the whole product.
    """
    keys = list(pars_dict)
    values = [list(v) for v in pars_dict.values()]
    index_product = [range(len(v)) for v in values]

    if max_comb is None:
        selected = list(itertools.product(*index_product))
        return [dict(zip(keys, [values[k][ci] for k, ci in enumerate(c)])) for c in selected]

    # Materialize index combinations lazily-aware: only build the full list when
    # it is small enough, otherwise sample without enumerating everything.
    total = 1
    for v in values:
        total *= len(v)

    if total <= max_comb:
        selected = list(itertools.product(*index_product))
    else:
        rng = random.Random(seed)
        needed = {(k, vi) for k, v in enumerate(values) for vi in range(len(v))}
        covered = set()
        selected = []
        seen = set()

        # Coverage pass: greedily add random combinations that introduce at least
        # one not-yet-tested parameter value, guaranteeing full single-value cover.
        while needed - covered:
            combo = tuple(rng.randrange(len(v)) for v in values)
            new = {(k, combo[k]) for k in range(len(keys))} - covered
            if new and combo not in seen:
                selected.append(combo)
                seen.add(combo)
                covered |= new

        # Top-up pass: add random unique combinations until reaching max_comb.
        while len(selected) < max_comb:
            combo = tuple(rng.randrange(len(v)) for v in values)
            if combo not in seen:
                selected.append(combo)
                seen.add(combo)

    return [dict(zip(keys, [values[k][ci] for k, ci in enumerate(c)])) for c in selected]
