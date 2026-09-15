#!/usr/bin/env python3
"""Build publication-facing v39 assets without training or inference.

The script creates a version-neutral study-design figure and copies only
audited, already-frozen scientific assets from the immutable v38 output.  The
v39 output therefore remains isolated while the underlying numerical evidence
is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
V38 = ROOT / "outputs" / "ieee_access_manuscript_v38_revision"
OUT = ROOT / "outputs" / "ieee_access_manuscript_v39_revision"
FIG = OUT / "figures"

COL = {
    "navy": "#17365D",
    "blue": "#4472C4",
    "orange": "#ED7D31",
    "red": "#C44E52",
    "light": "#EEF3F8",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_frozen(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "destination": destination.relative_to(ROOT).as_posix(),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def build_figure1() -> list[dict[str, object]]:
    fig, ax = plt.subplots(figsize=(7.16, 3.45), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.25, 4.05, 2.15, 1.25, "Scene-disjoint\nxView", "Training-only screening\nNo validation/test AP"),
        (2.75, 4.05, 2.15, 1.25, "Controlled\nexposures", "RealOnly · Duplicate\nRealCutout · Unity"),
        (5.25, 4.05, 2.15, 1.25, "Head-only\nevidence", "Paired seeds\nxView · HRSC · DIOR"),
        (7.75, 4.05, 2.00, 1.25, "Boundary\ntests", "Small150 ranking\nFaster R-CNN"),
        (2.05, 1.35, 2.55, 1.35, "Scope-specific\ndevelopment", "Duplicate-only inner validation\nSettings selected separately"),
        (5.35, 1.35, 2.55, 1.35, "Scope-interaction\nanalysis", "10 paired seeds · final epoch 20\nHistorical-validation lock first"),
    ]
    for index, (x, y, width, height, title, subtitle) in enumerate(boxes):
        later = index >= 4
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor="#FFF3E6" if later else COL["light"],
            edgecolor=COL["orange"] if later else COL["blue"],
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.70, title, ha="center", va="center", weight="bold", color=COL["navy"], size=7.7, linespacing=0.95)
        ax.text(x + width / 2, y + height * 0.23, subtitle, ha="center", va="center", size=6.8, linespacing=0.95)

    for x1, y1, x2, y2 in [
        (2.40, 4.68, 2.75, 4.68),
        (4.90, 4.68, 5.25, 4.68),
        (7.40, 4.68, 7.75, 4.68),
        (4.60, 2.02, 5.35, 2.02),
    ]:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=COL["navy"]))
    ax.add_patch(FancyArrowPatch((3.84, 4.02), (3.33, 2.72), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=COL["orange"]))
    ax.add_patch(FancyArrowPatch((6.25, 2.70), (6.25, 4.02), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=COL["orange"]))
    ax.text(
        5.0,
        0.55,
        "Evidence boundary: screening association and adaptation-specific efficacy; no causal or detector-general claim",
        ha="center",
        weight="bold",
        color=COL["red"],
        size=8.0,
    )

    records: list[dict[str, object]] = []
    for suffix in ("png", "pdf"):
        path = FIG / f"fig1_study_design_v39.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        records.append({"destination": path.relative_to(ROOT).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    plt.close(fig)
    return records


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    records = build_figure1()

    copies = [
        (V38 / "figures" / "fig2_feature_support_v38.png", FIG / "fig2_feature_support_v39.png"),
        (V38 / "figures" / "fig2_feature_support_v38.pdf", FIG / "fig2_feature_support_v39.pdf"),
        (V38 / "figures" / "fig5_fasterrcnn_hierarchy_v38.png", FIG / "fig5_fasterrcnn_hierarchy_v39.png"),
        (V38 / "figures" / "fig5_fasterrcnn_hierarchy_v38.pdf", FIG / "fig5_fasterrcnn_hierarchy_v39.pdf"),
        (V38 / "figures" / "fig6_compact_qualitative_v38.png", FIG / "fig6_compact_qualitative_v39.png"),
        (V38 / "ten_seed_head_only_results.csv", OUT / "ten_seed_head_only_results.csv"),
        (V38 / "v48_per_seed_interactions.csv", OUT / "scope_per_seed_interactions.csv"),
        (V38 / "xview_annotation_statistics.csv", OUT / "xview_annotation_statistics.csv"),
        (V38 / "scene_cluster_results.csv", OUT / "scene_cluster_results.csv"),
        (V38 / "scene_zero_target_sensitivity.csv", OUT / "scene_zero_target_sensitivity.csv"),
        (V38 / "figure2_feature_support_input_v38.json", OUT / "figure2_feature_support_input.json"),
    ]
    records.extend(copy_frozen(source, destination) for source, destination in copies)
    note = OUT / "figure6_selection_note_v39.md"
    note.write_text(
        "# Figure 6 selection note\n\n"
        "The panel reuses only examples selected under the preserved qualitative-selection rule. "
        "It contains one geometry-matched Unity/RealCutout training pair, the favorable Unity-recovery case, "
        "and the adverse Unity false-negative case. The display threshold is 0.25. The examples were not "
        "reselected for this manuscript and do not estimate an effect size.\n",
        encoding="utf-8",
    )
    records.append({"destination": note.relative_to(ROOT).as_posix(), "size": note.stat().st_size, "sha256": sha256(note)})
    manifest = OUT / "v39_asset_provenance.json"
    manifest.write_text(json.dumps({"training_or_inference_performed": False, "assets": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} v39 assets to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
