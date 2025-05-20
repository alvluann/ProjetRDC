"""
construction_cbr.py — Domain‑specific engine for recommending construction materials
Extends the generic CBREngine with:
  • custom similarity that mixes categorical and numeric attributes;
  • adaptation that scales material quantities by project area;
  • helper to load cases from a CSV file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, Tuple

from cbr_engine import CBREngine, Case


class ConstructionCBREngine(CBREngine):
    """CBR engine tailored for material‑recommendation problems."""

    # ----- custom similarity ------------------------------------------------
    @staticmethod
    def similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Weighted mix of categorical and numeric similarity (0‑1)."""
        weights = {
            "project_type": 2.0,  # ex.: residential, commercial, industrial
            "climate": 1.5,       # ex.: hot, mild, humid, coastal
            "area": 1.0,          # m²
            "budget": 1.0,        # currency
        }
        total = sum(weights.values())
        score = 0.0

        # categorical exact match (1) or 0
        for cat in ("project_type", "climate"):
            score += weights[cat] * (1.0 if a[cat] == b[cat] else 0.0)

        # numeric — linear decay
        for num in ("area", "budget"):
            rng = 1 + max(a[num], b[num])
            score += weights[num] * (1.0 - abs(a[num] - b[num]) / rng)

        return score / total

    # ------------------------------------------------------------------ reuse
    def reuse(self, retrieved: Case, problem: Dict[str, Any]):
        """Scale material quantities according to project area."""
        original_area = retrieved.problem["area"]
        new_area = problem["area"]
        factor = new_area / original_area if original_area else 1.0

        adapted = {
            "materials": {
                m: round(qty * factor)
                for m, qty in retrieved.solution["materials"].items()
            },
            "supplier": retrieved.solution["supplier"],
        }
        return adapted


# ---------------------------------------------------------------------------
# Utilities to import data
# ---------------------------------------------------------------------------

def load_cases_from_csv(path: str | Path, engine: CBREngine) -> None:
    """Expect CSV with columns: project_type,climate,area,budget,materials_json,supplier"""
    import json

    with open(path, newline="", encoding="utf8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            problem = {
                "project_type": row["project_type"],
                "climate": row["climate"],
                "area": float(row["area"]),
                "budget": float(row["budget"]),
            }
            solution = {
                "materials": json.loads(row["materials_json"]),
                "supplier": row["supplier"],
            }
            engine.retain(problem, solution)


# ---------------------------------------------------------------------------
# Quick demo when run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = ConstructionCBREngine(similarity=ConstructionCBREngine.similarity,
                                   persistence_path="data/construction_cases.json")
    if not len(engine):
        # seed with a single demo case
        engine.retain(
            {
                "project_type": "residential",
                "climate": "hot",
                "area": 100,
                "budget": 20000,
            },
            {
                "materials": {
                    "bloco_ceramico": 1200,
                    "cimento": 80,
                    "areia_m3": 10,
                },
                "supplier": "L&J Materiais",
            },
        )

    # new problem
    new_project = {
        "project_type": "residential",
        "climate": "hot",
        "area": 150,
        "budget": 32000,
    }
    solution, sim = engine.cycle(new_project)
    print(f"Recomendação (sim={sim:.2%}):", solution)
