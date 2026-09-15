"""Deterministic candidate identity and RNG derivation.

Candidate identity is deliberately independent from numerical RNG state.  The
full SHA-256 UID is the identity; the 64-bit seed is only a rendering input.
This module contains no scientific selection, support, or radiometry logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable


IDENTITY_SCHEMA = "support-scale-candidate-identity-v1"
RNG_NAMESPACE = "support-scale-candidate-rng-v1"
FLOAT_DECIMALS = 12


def _canonical_value(value: Any) -> Any:
    """Return a JSON-safe value with stable ordering and float formatting."""
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Candidate identity cannot contain NaN or infinity")
        return format(value, f".{FLOAT_DECIMALS}f")
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    raise TypeError(f"Unsupported candidate identity value: {type(value)!r}")


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize identity material as canonical UTF-8 JSON text."""
    return json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def identity_document(
    *,
    experiment_namespace: str,
    cohort_id: str,
    original_host_id: int,
    base_render_id: int,
    appearance_variant_id: str,
    appearance_variant_parameters: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    """Build the prospectively frozen identity document."""
    return {
        "schema": IDENTITY_SCHEMA,
        "experiment_namespace": str(experiment_namespace),
        "cohort_id": str(cohort_id),
        "original_host_id": int(original_host_id),
        "base_render_id": int(base_render_id),
        "appearance_variant_id": str(appearance_variant_id),
        "appearance_variant_parameters": appearance_variant_parameters,
        "candidate_index": int(candidate_index),
    }


def candidate_uid(document: dict[str, Any]) -> tuple[str, str]:
    """Return full SHA-256 candidate UID and its canonical JSON preimage."""
    serialized = canonical_json(document)
    uid = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return uid, serialized


def _seed_for_nonce(uid: str, nonce: int) -> int:
    material = f"{RNG_NAMESPACE}|{uid}|nonce={nonce}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class SeedDerivation:
    rng_seed_64: int
    collision_nonce: int


def derive_unique_rng_seed64(
    uid: str,
    used_seeds: set[int],
    reserved_seeds: Iterable[int] = (),
) -> SeedDerivation:
    """Derive a deterministic 64-bit seed and preclude known collisions.

    The canonical generation order is part of the frozen protocol, so nonce
    resolution is reproducible.  A seed is never used as candidate identity.
    """
    reserved = reserved_seeds if isinstance(reserved_seeds, set) else set(reserved_seeds)
    nonce = 0
    while True:
        seed = _seed_for_nonce(uid, nonce)
        if seed not in used_seeds and seed not in reserved:
            used_seeds.add(seed)
            return SeedDerivation(seed, nonce)
        nonce += 1


def legacy_31bit_seed_preimage(source_label: str, base_candidate_id: int, rank: int) -> str:
    """Reconstruct the superseded seed preimage for failure auditing only."""
    return f"support-diversity-v1|{source_label}|{base_candidate_id}|{rank}"


def legacy_31bit_seed(source_label: str, base_candidate_id: int, rank: int) -> int:
    digest = hashlib.sha256(
        legacy_31bit_seed_preimage(source_label, base_candidate_id, rank).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
