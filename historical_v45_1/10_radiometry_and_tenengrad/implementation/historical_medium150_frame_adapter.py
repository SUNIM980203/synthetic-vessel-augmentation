"""Authoritative historical Medium150 input adapter.

This module binds a recovered historical host/source/placement/geometry frame
to the frozen v31 pre-render interface.  It does not render candidates, run
calibration, compute support, select pairs, train detectors, or inspect AP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_MANIFEST = (
    ROOT
    / "AnalysisResults"
    / "expanded_v1"
    / "historical_medium150_provenance_recovery"
    / "historical_medium150_frame_manifest.csv"
)
DEFAULT_SOURCE_MANIFEST = (
    ROOT
    / "AnalysisResults"
    / "expanded_v1"
    / "historical_medium150_deterministic_replay_audit"
    / "historical_medium150_source_cutout_manifest.csv"
)
DEFAULT_ORIENTATION_MATRIX = (
    ROOT
    / "AnalysisResults"
    / "expanded_v1"
    / "historical_medium150_deterministic_replay_audit"
    / "historical_medium150_orientation_recovery_matrix.csv"
)
BASE_HOST_ROOT = ROOT / "PreparedData" / "xview_expanded_512_v1" / "images" / "train"
FRAME_MODE = "historical_medium150"
ORIENTATION_MODE = "NATIVE_SOURCE_RASTER"
ADDITIONAL_ROTATION_DEGREES = 0
RESIZE_INTERPOLATION = "PIL.Image.Resampling.LANCZOS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def decimal_equal(left: str | Decimal, right: str | Decimal) -> bool:
    return Decimal(str(left)) == Decimal(str(right))


@dataclass(frozen=True)
class HistoricalFrameSpec:
    frame_index: int
    host_id: int
    host_path: str
    host_sha256: str
    source_scene: str
    source_cutout_path: str
    source_cutout_sha256: str
    center_x: float
    center_y: float
    bbox_xmin: float
    bbox_ymin: float
    bbox_xmax: float
    bbox_ymax: float
    bbox_width: int
    bbox_height: int
    bbox_scale: float
    aspect_ratio: float
    host_width: int
    host_height: int
    source_raw_width: int
    source_raw_height: int
    source_cropped_width: int
    source_cropped_height: int
    orientation_mode: str = ORIENTATION_MODE
    additional_rotation: int = ADDITIONAL_ROTATION_DEGREES
    frame_mode: str = FRAME_MODE

    def current_generator_binding(self) -> dict[str, Any]:
        """Return the exact v31 pre-render binding without materializing pixels."""

        return {
            "frame_mode": self.frame_mode,
            "host_input": self.host_path,
            "host_sha256": self.host_sha256,
            "source_asset_input": self.source_cutout_path,
            "source_cutout_sha256": self.source_cutout_sha256,
            "source_selection_mode": "AUTHORITATIVE_MANIFEST_OVERRIDE",
            "source_pool_ranking_bypassed": True,
            "source_random_choice_bypassed": True,
            "source_replacement_bypassed": True,
            "center_input": [self.center_x, self.center_y],
            "placement_mode": "AUTHORITATIVE_HISTORICAL_CENTER",
            "placement_search_bypassed": True,
            "bbox_input_xywh": [
                self.bbox_xmin,
                self.bbox_ymin,
                float(self.bbox_width),
                float(self.bbox_height),
            ],
            "bbox_sampling_bypassed": True,
            "requested_width": self.bbox_width,
            "requested_height": self.bbox_height,
            "mapped_width": self.bbox_width,
            "mapped_height": self.bbox_height,
            "resize_interpolation": RESIZE_INTERPOLATION,
            "orientation_mode": self.orientation_mode,
            "additional_rotation_degrees": self.additional_rotation,
            "automatic_rotate90_bypassed": True,
            "aspect_normalization_bypassed": True,
            "appearance_seed_input": "UNCHANGED_FROZEN_CANDIDATE_UID_RNG_STAGE",
            "calibration_stage": "UNCHANGED_AFTER_FUTURE_RENDER",
            "candidate_uid_stage": "UNCHANGED_AFTER_FUTURE_RENDER",
            "support_scoring_stage": "UNCHANGED_AFTER_FINAL_CALIBRATED_JPEG",
            "render_executed": False,
            "stop_boundary": "PRE_RENDER_INPUT_CONSTRUCTION",
        }

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _unique_index(rows: list[dict[str, str]], key: str, label: str) -> dict[int, dict[str, str]]:
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        value = int(row[key])
        if value in output:
            raise ValueError(f"Duplicate {key}={value} in {label}")
        output[value] = row
    return output


def load_historical_frames(
    frame_manifest: Path = DEFAULT_FRAME_MANIFEST,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    orientation_matrix: Path = DEFAULT_ORIENTATION_MATRIX,
) -> list[HistoricalFrameSpec]:
    """Load, join, and strictly validate the 150 authoritative frame rows."""

    frame_rows = read_csv(frame_manifest)
    source_rows = read_csv(source_manifest)
    orientation_rows = read_csv(orientation_matrix)
    if len(frame_rows) != 150 or len(source_rows) != 150 or len(orientation_rows) != 150:
        raise ValueError(
            "Authoritative inputs must contain exactly 150 rows each: "
            f"frame={len(frame_rows)}, source={len(source_rows)}, orientation={len(orientation_rows)}"
        )
    frame_by_index = _unique_index(frame_rows, "frame_index", "frame manifest")
    source_by_index = _unique_index(source_rows, "frame_index", "source manifest")
    orientation_by_index = _unique_index(orientation_rows, "frame_index", "orientation matrix")
    expected = set(range(1, 151))
    if set(frame_by_index) != expected or set(source_by_index) != expected or set(orientation_by_index) != expected:
        raise ValueError("All authoritative manifests must contain frame_index 1..150 exactly")
    if len({int(row["host_id"]) for row in frame_rows}) != 150:
        raise ValueError("Historical frame manifest does not contain 150 unique hosts")

    specs: list[HistoricalFrameSpec] = []
    for frame_index in range(1, 151):
        frame = frame_by_index[frame_index]
        source = source_by_index[frame_index]
        orientation = orientation_by_index[frame_index]
        host_id = int(frame["host_id"])
        if int(source["host_id"]) != host_id:
            raise ValueError(f"Host join mismatch at frame {frame_index}")
        if source["host_sha256"].lower() != frame["host_sha256"].lower():
            raise ValueError(f"Frozen host-hash disagreement at frame {frame_index}")
        if int(orientation["modified_image_id"]) != host_id:
            raise ValueError(f"Orientation-matrix host mismatch at frame {frame_index}")
        if orientation["SOURCE_CUTOUT_ID"] != "D" or orientation["NATIVE_RASTER_ORIENTATION"] != "D":
            raise ValueError(f"Source/native orientation not deterministically recovered at frame {frame_index}")

        host_path = BASE_HOST_ROOT / frame["host_filename"]
        source_path = repo_path(source["source_cutout_path"])
        if not host_path.is_file() or sha256_file(host_path) != frame["host_sha256"].lower():
            raise ValueError(f"FRAME_INPUT_HASH_MISMATCH host at frame {frame_index}: {host_path}")
        if not source_path.is_file() or sha256_file(source_path) != source["source_cutout_sha256"].lower():
            raise ValueError(f"FRAME_INPUT_HASH_MISMATCH source at frame {frame_index}: {source_path}")

        for frame_key, source_key in (
            ("bbox_xmin", "bbox_x"),
            ("bbox_ymin", "bbox_y"),
            ("bbox_width", "bbox_width"),
            ("bbox_height", "bbox_height"),
        ):
            if not decimal_equal(frame[frame_key], source[source_key]):
                raise ValueError(f"BBox provenance mismatch at frame {frame_index}: {frame_key}")

        x = Decimal(frame["bbox_xmin"])
        y = Decimal(frame["bbox_ymin"])
        width_decimal = Decimal(frame["bbox_width"])
        height_decimal = Decimal(frame["bbox_height"])
        if width_decimal != width_decimal.to_integral_value() or height_decimal != height_decimal.to_integral_value():
            raise ValueError(f"Current raster API cannot exactly represent noninteger bbox at frame {frame_index}")
        width = int(width_decimal)
        height = int(height_decimal)
        if Decimal(frame["bbox_xmax"]) != x + width_decimal or Decimal(frame["bbox_ymax"]) != y + height_decimal:
            raise ValueError(f"BBox endpoint inconsistency at frame {frame_index}")
        if Decimal(frame["center_x"]) != x + width_decimal / 2 or Decimal(frame["center_y"]) != y + height_decimal / 2:
            raise ValueError(f"Center/bbox inconsistency at frame {frame_index}")
        scale = math.sqrt(width * height)
        if not math.isclose(scale, float(frame["bbox_scale"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"BBox-scale inconsistency at frame {frame_index}")
        if source["orientation_representation"] != ORIENTATION_MODE:
            raise ValueError(f"Native orientation representation mismatch at frame {frame_index}")
        if source["orientation_status"] != "DETERMINISTICALLY_RECONSTRUCTED":
            raise ValueError(f"Native orientation status mismatch at frame {frame_index}")
        if source["independent_in_plane_rotation"].lower() != "false":
            raise ValueError(f"Unexpected historical in-plane rotation at frame {frame_index}")

        with Image.open(host_path) as host_image:
            host_width, host_height = host_image.size
        with Image.open(source_path) as cutout:
            raw_width, raw_height = cutout.size
            alpha_bbox = cutout.convert("RGBA").getchannel("A").getbbox()
            if alpha_bbox is None:
                raise ValueError(f"Empty source alpha at frame {frame_index}")
            cropped_width = alpha_bbox[2] - alpha_bbox[0]
            cropped_height = alpha_bbox[3] - alpha_bbox[1]
        if (raw_width, raw_height) != (int(source["source_raw_width"]), int(source["source_raw_height"])):
            raise ValueError(f"Source raw dimensions mismatch at frame {frame_index}")
        if (cropped_width, cropped_height) != (
            int(source["source_cropped_width"]),
            int(source["source_cropped_height"]),
        ):
            raise ValueError(f"Source alpha-cropped dimensions mismatch at frame {frame_index}")
        if x < 0 or y < 0 or x + width_decimal > host_width or y + height_decimal > host_height:
            raise ValueError(f"Historical bbox is outside host bounds at frame {frame_index}")

        specs.append(
            HistoricalFrameSpec(
                frame_index=frame_index,
                host_id=host_id,
                host_path=relative_repo_path(host_path),
                host_sha256=frame["host_sha256"].lower(),
                source_scene=frame["source_scene"],
                source_cutout_path=relative_repo_path(source_path),
                source_cutout_sha256=source["source_cutout_sha256"].lower(),
                center_x=float(frame["center_x"]),
                center_y=float(frame["center_y"]),
                bbox_xmin=float(frame["bbox_xmin"]),
                bbox_ymin=float(frame["bbox_ymin"]),
                bbox_xmax=float(frame["bbox_xmax"]),
                bbox_ymax=float(frame["bbox_ymax"]),
                bbox_width=width,
                bbox_height=height,
                bbox_scale=scale,
                aspect_ratio=width / height,
                host_width=host_width,
                host_height=host_height,
                source_raw_width=raw_width,
                source_raw_height=raw_height,
                source_cropped_width=cropped_width,
                source_cropped_height=cropped_height,
            )
        )
    return specs


def load_historical_frame(frame_index: int, **kwargs: Any) -> HistoricalFrameSpec:
    if not 1 <= frame_index <= 150:
        raise ValueError("frame_index must be in 1..150")
    return load_historical_frames(**kwargs)[frame_index - 1]


def materialize_pre_render_cutout(spec: HistoricalFrameSpec) -> Image.Image:
    """Future integration hook; not called by the adapter freeze dry-run."""

    source_path = repo_path(spec.source_cutout_path)
    if sha256_file(source_path) != spec.source_cutout_sha256:
        raise ValueError(f"FRAME_INPUT_HASH_MISMATCH source at frame {spec.frame_index}")
    with Image.open(source_path) as image:
        rgba = image.convert("RGBA")
        alpha_bbox = rgba.getchannel("A").getbbox()
        if alpha_bbox is None:
            raise ValueError(f"Empty source alpha at frame {spec.frame_index}")
        native = rgba.crop(alpha_bbox)
        # Intentionally no transpose, rotate, flip, or orientation normalization.
        return native.resize((spec.bbox_width, spec.bbox_height), Image.Resampling.LANCZOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-mode", choices=[FRAME_MODE], required=True)
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--orientation-matrix", type=Path, default=DEFAULT_ORIENTATION_MATRIX)
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = load_historical_frames(args.frame_manifest, args.source_manifest, args.orientation_matrix)
    summary = {
        "frame_mode": FRAME_MODE,
        "historical_frames_loaded": len(specs),
        "host_hashes_verified": len(specs),
        "source_hashes_verified": len(specs),
        "render_executed": False,
        "candidate_images_created": 0,
        "first_binding": specs[0].current_generator_binding(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
