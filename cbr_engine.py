"""
cbr_engine.py — Minimal Case‑Based Reasoning engine (cycle 4R)
Author: Luan — 2025‑05‑19

This module provides a simple but extensible CBR engine to help you
prototype quickly.  It supports:
  • Case storage in memory + optional JSON persistence.
  • Retrieval with a pluggable similarity function (default handles
    numeric and categorical attributes).
  • Naïve reuse/adaptation out‑of‑the‑box, with the option to override
    per‑domain.
  • Revision via an optional validator callable.
  • Automatic retention (learning) after each successful cycle.

Feel free to tweak, extend or replace any part.  Typical extensions:
  ‑ Better similarity metrics (cosine, ontology‑based, etc.)
  ‑ Sophisticated adaptation (rule‑based, ML, etc.)
  ‑ Indexing structures or clustering for faster retrieval.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """A single problem–solution pair."""

    problem: Dict[str, Any]
    solution: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Case":
        return cls(**d)

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def default_similarity(
    a: Dict[str, Any],
    b: Dict[str, Any],
    weights: Dict[str, float] | None = None,
) -> float:
    """A very simple similarity function returning a value in [0, 1].

    * Numeric fields → 1 − (∆ / range).
    * Categorical/other fields → 1 if equal, 0 otherwise.
    """

    if weights is None:
        weights = {k: 1.0 for k in a.keys()}

    tot_weight = 0.0
    score = 0.0

    for key in a.keys():
        w = weights.get(key, 1.0)
        tot_weight += w

        av, bv = a[key], b[key]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            rng = 1.0 + max(abs(av), abs(bv))  # avoid division by zero
            score += w * (1.0 - abs(av - bv) / rng)
        else:
            score += w * (1.0 if av == bv else 0.0)

    return score / tot_weight if tot_weight else 0.0

# ---------------------------------------------------------------------------
# The Engine (4R cycle)
# ---------------------------------------------------------------------------

class CBREngine:
    """A minimal but extensible Case‑Based Reasoning engine."""

    def __init__(
        self,
        similarity: Callable[[Dict[str, Any], Dict[str, Any]], float] = default_similarity,
        persistence_path: str | pathlib.Path | None = None,
    ) -> None:
        self._cases: List[Case] = []
        self.similarity = similarity
        self._path = pathlib.Path(persistence_path) if persistence_path else None

        if self._path and self._path.exists():
            self._load()

    # --------------------------- R1: Retrieve ---------------------------
    def retrieve(self, problem: Dict[str, Any], k: int = 1) -> List[Tuple[Case, float]]:
        """Return the *k* most similar cases and their similarity scores."""
        ranked = sorted(
            ((c, self.similarity(problem, c.problem)) for c in self._cases),
            key=lambda t: t[1],
            reverse=True,
        )
        return ranked[:k]

    # --------------------------- R2: Reuse ------------------------------
    def reuse(self, retrieved: Case, problem: Dict[str, Any]) -> Any:
        """Adapt the retrieved solution to the new problem.

        Default: returns the solution unchanged (copy). Override this
        method for domain‑specific adaptation.
        """
        return retrieved.solution

    # --------------------------- R3: Revise -----------------------------
    def revise(
        self,
        problem: Dict[str, Any],
        proposed_solution: Any,
        validator: Callable[[Dict[str, Any], Any], bool] | None = None,
    ) -> Any:
        """Validate/adjust the solution before accepting it."""
        if validator and not validator(problem, proposed_solution):
            raise ValueError("Proposed solution failed validation")
        return proposed_solution

    # --------------------------- R4: Retain -----------------------------
    def retain(self, problem: Dict[str, Any], solution: Any) -> None:
        """Store the new (problem, solution) pair so the system learns."""
        self._cases.append(Case(problem, solution))
        self._save()

    # ----------------------- Convenience cycle -------------------------
    def cycle(
        self,
        problem: Dict[str, Any],
        k: int = 1,
        validator: Callable[[Dict[str, Any], Any], bool] | None = None,
        learn: bool = True,
    ) -> Tuple[Any, float]:
        """Run the full 4R cycle and return (solution, similarity)."""
        retrieved, sim = self.retrieve(problem, k=k)[0]
        adapted = self.reuse(retrieved, problem)
        final = self.revise(problem, adapted, validator=validator)
        if learn:
            self.retain(problem, final)
        return final, sim

    # ----------------------- Persistence helpers -----------------------
    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf8") as fh:
            json.dump([c.to_dict() for c in self._cases], fh, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        with self._path.open(encoding="utf8") as fh:
            self._cases = [Case.from_dict(d) for d in json.load(fh)]

    # -------------------------- Utility API ----------------------------
    @property
    def cases(self) -> List[Case]:
        return self._cases

    def __len__(self) -> int:
        return len(self._cases)


# ---------------------------------------------------------------------------
# Quick smoke‑test: python cbr_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = CBREngine(persistence_path="data/cases.json")

    # Insert some sample cases only if base is empty
    if not engine.cases:
        engine.retain({"temp": 80, "pressure": 30}, "check valve")
        engine.retain({"temp": 40, "pressure": 10}, "normal operation")

    # New problem
    problem = {"temp": 78, "pressure": 29}
    sol, sim = engine.cycle(problem, validator=None)
    print("Suggested solution:", sol, f"(similarity={sim:.2%})")
