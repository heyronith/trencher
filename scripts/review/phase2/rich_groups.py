"""Map the 22 rich additive columns to modality groups."""

from __future__ import annotations

from trencher.baselines.feature_spec import rich_additive_feature_names

RICH_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "microstructure": ("mask_micro", "micro_"),
    "ownership": ("mask_ownership", "own_"),
    "graph": ("mask_graph", "graph_"),
    "wallet_prior": ("mask_wallet_prior", "wallet_prior_"),
}


def column_to_group(name: str) -> str:
    if name in ("mask_micro",) or name.startswith("micro_"):
        return "microstructure"
    if name in ("mask_ownership",) or name.startswith("own_"):
        return "ownership"
    if name in ("mask_graph",) or name.startswith("graph_"):
        return "graph"
    if name in ("mask_wallet_prior",) or name.startswith("wallet_prior_"):
        return "wallet_prior"
    raise KeyError(f"unmapped rich column: {name}")


def rich_column_group_map() -> dict[str, str]:
    return {c: column_to_group(c) for c in rich_additive_feature_names()}


def group_column_indices(feature_names: list[str], group: str) -> list[int]:
    rich = set(rich_additive_feature_names())
    out: list[int] = []
    for i, n in enumerate(feature_names):
        if n in rich and column_to_group(n) == group:
            out.append(i)
    return out


def drop_group_indices(feature_names: list[str], group: str) -> list[int]:
    drop = set(group_column_indices(feature_names, group))
    return [i for i in range(len(feature_names)) if i not in drop]


def broad_plus_group_indices(feature_names: list[str], group: str) -> list[int]:
    from trencher.baselines.feature_spec import broad_feature_names

    n_broad = len(broad_feature_names())
    keep = set(range(n_broad)) | set(group_column_indices(feature_names, group))
    return sorted(keep)
