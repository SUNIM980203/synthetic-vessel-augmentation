#!/usr/bin/env python3
"""Compare real and synthetic vessels in a frozen detector feature space."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import roi_align
from ultralytics import YOLO


FEATURE_LAYERS = (16, 19, 22)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--real-coco", type=Path, required=True)
    parser.add_argument("--real-images", type=Path, required=True)
    parser.add_argument(
        "--domain",
        nargs=3,
        action="append",
        metavar=("NAME", "COCO", "IMAGES"),
        required=True,
        help="Synthetic domain name, COCO JSON, and image directory; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_records(
    coco_path: Path,
    images_dir: Path,
    category_id: int,
    synthetic_only: bool,
) -> list[dict[str, object]]:
    coco = load_json(coco_path)
    images = {int(row["id"]): row for row in coco["images"]}
    records: list[dict[str, object]] = []
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) != category_id:
            continue
        if synthetic_only and str(annotation.get("source", "")) != "unity_cutout":
            continue
        image = images[int(annotation["image_id"])]
        bbox = [float(value) for value in annotation["bbox"]]
        records.append(
            {
                "annotation_id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
                "file_name": str(image["file_name"]),
                "image_path": str(images_dir / str(image["file_name"])),
                "source_scene": str(image.get("source_scene", image["file_name"])),
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
                "bbox": bbox,
                "bbox_area": float(bbox[2] * bbox[3]),
                "bbox_scale": float(math.sqrt(max(bbox[2] * bbox[3], 1.0))),
            }
        )
    return records


def scale_bin(scale: float) -> str:
    if scale < 16.0:
        return "tiny_lt16"
    if scale < 32.0:
        return "small_16_32"
    if scale < 64.0:
        return "medium_32_64"
    return "large_ge64"


def extract_embeddings(
    model: torch.nn.Module,
    records: list[dict[str, object]],
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record["image_path"])].append(index)
    image_paths = sorted(grouped)
    result = np.zeros((len(records), 448), dtype=np.float32)
    captured: dict[int, torch.Tensor] = {}
    handles = [
        model.model[layer].register_forward_hook(
            lambda _module, _inputs, output, layer=layer: captured.__setitem__(layer, output)
        )
        for layer in FEATURE_LAYERS
    ]
    try:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors: list[torch.Tensor] = []
            rois: list[list[float]] = []
            record_indices: list[int] = []
            for batch_index, image_path in enumerate(batch_paths):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Could not decode {image_path}")
                rgb = cv2.cvtColor(cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                tensors.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
                for record_index in grouped[image_path]:
                    record = records[record_index]
                    x, y, width, height = record["bbox"]
                    scale_x = imgsz / float(record["image_width"])
                    scale_y = imgsz / float(record["image_height"])
                    x0 = max(0.0, x * scale_x)
                    y0 = max(0.0, y * scale_y)
                    x1 = min(float(imgsz), (x + width) * scale_x)
                    y1 = min(float(imgsz), (y + height) * scale_y)
                    rois.append([float(batch_index), x0, y0, max(x0 + 1.0, x1), max(y0 + 1.0, y1)])
                    record_indices.append(record_index)
            batch = torch.stack(tensors).to(device=device, dtype=torch.float32).div_(255.0)
            captured.clear()
            with torch.inference_mode():
                model(batch)
            roi_tensor = torch.tensor(rois, device=device, dtype=torch.float32)
            embeddings: list[torch.Tensor] = []
            for layer in FEATURE_LAYERS:
                feature_map = captured[layer]
                pooled = roi_align(
                    feature_map,
                    roi_tensor,
                    output_size=(3, 3),
                    spatial_scale=feature_map.shape[-1] / float(imgsz),
                    sampling_ratio=2,
                    aligned=True,
                ).mean(dim=(2, 3))
                embeddings.append(pooled)
            combined = torch.cat(embeddings, dim=1)
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)
            result[np.asarray(record_indices)] = combined.cpu().numpy().astype(np.float32)
    finally:
        for handle in handles:
            handle.remove()
    return result


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - left @ right.T, 0.0, 2.0)


def nearest_real_metrics(
    synthetic: np.ndarray,
    synthetic_records: list[dict[str, object]],
    real: np.ndarray,
    real_records: list[dict[str, object]],
    rng: np.random.Generator,
) -> dict[str, object]:
    by_bin_real = defaultdict(list)
    by_bin_synthetic = defaultdict(list)
    for index, record in enumerate(real_records):
        by_bin_real[scale_bin(float(record["bbox_scale"]))].append(index)
    for index, record in enumerate(synthetic_records):
        by_bin_synthetic[scale_bin(float(record["bbox_scale"]))].append(index)

    result: dict[str, object] = {}
    all_synthetic_nn: list[float] = []
    all_real_reference: list[float] = []
    all_supported: list[bool] = []
    for bin_name in ("tiny_lt16", "small_16_32", "medium_32_64", "large_ge64"):
        synthetic_indices = np.asarray(by_bin_synthetic.get(bin_name, []), dtype=int)
        real_indices = np.asarray(by_bin_real.get(bin_name, []), dtype=int)
        if not len(synthetic_indices) or len(real_indices) < 2:
            continue
        if len(real_indices) > 1200:
            real_indices = np.sort(rng.choice(real_indices, 1200, replace=False))
        real_bin = real[real_indices]
        synthetic_bin = synthetic[synthetic_indices]
        synthetic_distances = cosine_distance(synthetic_bin, real_bin)
        synthetic_scenes = np.asarray([str(synthetic_records[index]["source_scene"]) for index in synthetic_indices])
        real_scenes = np.asarray([str(real_records[index]["source_scene"]) for index in real_indices])
        synthetic_distances[synthetic_scenes[:, None] == real_scenes[None, :]] = np.inf
        synthetic_nn = np.min(synthetic_distances, axis=1)
        real_distances = cosine_distance(real_bin, real_bin)
        real_distances[real_scenes[:, None] == real_scenes[None, :]] = np.inf
        real_reference = np.min(real_distances, axis=1)
        threshold = float(np.percentile(real_reference, 95))
        supported = synthetic_nn <= threshold
        result[bin_name] = {
            "synthetic_count": int(len(synthetic_indices)),
            "real_reference_count": int(len(real_indices)),
            "synthetic_nearest_real_distance_median": float(np.median(synthetic_nn)),
            "synthetic_nearest_real_distance_p95": float(np.percentile(synthetic_nn, 95)),
            "real_leave_one_out_distance_median": float(np.median(real_reference)),
            "real_leave_one_out_distance_p95": threshold,
            "median_distance_ratio": float(np.median(synthetic_nn) / max(np.median(real_reference), 1e-12)),
            "supported_fraction_at_real_p95": float(np.mean(supported)),
        }
        all_synthetic_nn.extend(synthetic_nn.tolist())
        all_real_reference.extend(real_reference.tolist())
        all_supported.extend(supported.tolist())
    return {
        "overall": {
            "synthetic_nearest_real_distance_median": float(np.median(all_synthetic_nn)),
            "real_leave_one_out_distance_median": float(np.median(all_real_reference)),
            "median_distance_ratio": float(np.median(all_synthetic_nn) / max(np.median(all_real_reference), 1e-12)),
            "supported_fraction_at_scale_specific_real_p95": float(np.mean(all_supported)),
        },
        "by_scale": result,
    }


def matched_real_sample(
    synthetic_records: list[dict[str, object]],
    real_records: list[dict[str, object]],
    rng: np.random.Generator,
) -> np.ndarray:
    by_bin = defaultdict(list)
    for index, record in enumerate(real_records):
        by_bin[scale_bin(float(record["bbox_scale"]))].append(index)
    selected: list[int] = []
    for record in synthetic_records:
        candidates = [
            index
            for index in by_bin[scale_bin(float(record["bbox_scale"]))]
            if str(real_records[index]["source_scene"]) != str(record["source_scene"])
        ]
        if not candidates:
            candidates = [
                index
                for index in range(len(real_records))
                if str(real_records[index]["source_scene"]) != str(record["source_scene"])
            ]
        selected.append(int(rng.choice(candidates)))
    return np.asarray(selected, dtype=int)


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a 2-D distance block without constructing an n*n*d tensor."""
    left_norm = np.sum(left * left, axis=1, dtype=np.float64)[:, None]
    right_norm = np.sum(right * right, axis=1, dtype=np.float64)[None, :]
    squared = left_norm + right_norm - 2.0 * (left @ right.T)
    return np.maximum(squared, 0.0)


def _rbf_kernel_mean(left: np.ndarray, right: np.ndarray, gamma: float, block_size: int = 512) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(left), block_size):
        squared = _squared_distances(left[start : start + block_size], right)
        total += float(np.exp(-gamma * squared).sum(dtype=np.float64))
        count += int(squared.size)
    return total / max(count, 1)


def rbf_mmd_squared(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    combined = np.vstack([left, right])
    bandwidth_sample_limit = 4096
    if len(combined) > bandwidth_sample_limit:
        indices = np.linspace(0, len(combined) - 1, bandwidth_sample_limit, dtype=int)
        bandwidth_sample = combined[indices]
    else:
        bandwidth_sample = combined
    squared = _squared_distances(bandwidth_sample, bandwidth_sample)
    upper = squared[np.triu_indices(len(bandwidth_sample), 1)]
    nonzero = upper[upper > 1e-12]
    bandwidth_squared = float(np.median(nonzero)) if len(nonzero) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth_squared, 1e-12)
    value = float(
        _rbf_kernel_mean(left, left, gamma)
        + _rbf_kernel_mean(right, right, gamma)
        - 2.0 * _rbf_kernel_mean(left, right, gamma)
    )
    return value, math.sqrt(bandwidth_squared)


def effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular ** 2
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    probabilities = probabilities[probabilities > 1e-12]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def diversity_metrics(embeddings: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    if len(embeddings) > 500:
        embeddings = embeddings[np.sort(rng.choice(len(embeddings), 500, replace=False))]
    distances = cosine_distance(embeddings, embeddings)
    values = distances[np.triu_indices(len(embeddings), 1)]
    return {
        "pairwise_cosine_distance_median": float(np.median(values)),
        "pairwise_cosine_distance_p95": float(np.percentile(values, 95)),
        "effective_rank": effective_rank(embeddings),
    }


def plot_pca(path: Path, domains: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, float]:
    samples: list[np.ndarray] = []
    labels: list[str] = []
    for name, embeddings in domains.items():
        limit = min(len(embeddings), 1000 if name == "real" else 500)
        indices = np.arange(len(embeddings)) if limit == len(embeddings) else np.sort(rng.choice(len(embeddings), limit, replace=False))
        samples.append(embeddings[indices])
        labels.extend([name] * limit)
    matrix = np.vstack(samples)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vectors[:2].T
    variance = singular ** 2
    explained = variance[:2] / variance.sum()
    colors = ["#777777", "#d95f02", "#1b9e77", "#7570b3"]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    labels_array = np.asarray(labels)
    for color, name in zip(colors, domains):
        mask = labels_array == name
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=10, alpha=0.38, label=name, color=color, edgecolors="none")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("Frozen Duplicate-backbone vessel ROI embeddings")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"pc1_explained_fraction": float(explained[0]), "pc2_explained_fraction": float(explained[1])}


def render_report(result: dict[str, object]) -> str:
    synthetic_names = [name for name in result["domains"] if name != "real"]
    best_mmd_name = min(synthetic_names, key=lambda name: result["domains"][name]["scale_matched_mmd_squared"])
    best_support_name = max(
        synthetic_names,
        key=lambda name: result["domains"][name]["support"]["overall"]["supported_fraction_at_scale_specific_real_p95"],
    )
    best_support = result["domains"][best_support_name]["support"]["overall"]["supported_fraction_at_scale_specific_real_p95"]
    lines = [
        "# Vessel detector-feature support report",
        "",
        "## Design",
        "",
        "A frozen seed-20260723 Duplicate YOLO26n detector produced P3/P4/P5 ROI-aligned embeddings for real training vessels and each synthetic vessel condition. Metrics use training objects only; no validation or test predictions are involved. Nearest-real support is computed within bbox-scale bins and always excludes candidates from the same acquisition scene.",
        "",
        "## Outcome",
        "",
        f"{best_mmd_name} has the lowest scale-matched MMD, while {best_support_name} has the highest cross-scene support fraction ({best_support:.3f}). A support fraction far below 0.5 means most synthetic vessels remain outside the feature neighborhoods normally occupied by real vessels. Compare effective rank with real to distinguish excessive unsupported variation from simple diversity shortage.",
        "",
        "## Feature support",
        "",
        "| Domain | Objects | NN distance ratio vs real | Supported at real p95 | Scale-matched MMD2 | Effective rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, domain in result["domains"].items():
        if name == "real":
            continue
        overall = domain["support"]["overall"]
        lines.append(
            f"| {name} | {domain['count']} | {overall['median_distance_ratio']:.3f} | {overall['supported_fraction_at_scale_specific_real_p95']:.3f} | {domain['scale_matched_mmd_squared']:.4f} | {domain['diversity']['effective_rank']:.2f} |"
        )
    lines.extend(["", "## Scale-specific support", ""])
    for name, domain in result["domains"].items():
        if name == "real":
            continue
        lines.extend([f"### {name}", "", "| Scale | Synthetic n | NN ratio | Supported fraction |", "|---|---:|---:|---:|"])
        for scale, row in domain["support"]["by_scale"].items():
            lines.append(f"| {scale} | {row['synthetic_count']} | {row['median_distance_ratio']:.3f} | {row['supported_fraction_at_real_p95']:.3f} |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "A nearest-real distance ratio above 1 means synthetic objects lie farther from real examples than real objects lie from other real examples at the same bbox scale. The support fraction uses the real leave-one-out p95 distance as a scale-specific threshold. MMD measures the distribution difference after matching the real sample to the synthetic scale-bin frequencies. Effective rank is an embedding-diversity indicator, not a quality score.",
            "",
            "Use this analysis to distinguish unsupported synthetic appearance from insufficient diversity. It is diagnostic and must not be used to unlock the held-out test split.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = YOLO(str(args.weights)).model.to(device).eval()

    records: dict[str, list[dict[str, object]]] = {
        "real": load_records(args.real_coco, args.real_images, args.category_id, synthetic_only=False)
    }
    for name, coco, images in args.domain:
        if name in records:
            raise ValueError(f"Duplicate domain name: {name}")
        records[name] = load_records(Path(coco), Path(images), args.category_id, synthetic_only=True)

    embeddings: dict[str, np.ndarray] = {}
    for name, domain_records in records.items():
        embeddings[name] = extract_embeddings(model, domain_records, args.imgsz, args.batch_size, device)
        np.savez_compressed(
            args.output / f"{name}_embeddings.npz",
            embeddings=embeddings[name],
            annotation_ids=np.asarray([row["annotation_id"] for row in domain_records]),
            bbox_scales=np.asarray([row["bbox_scale"] for row in domain_records], dtype=np.float32),
        )

    real_embeddings = embeddings["real"]
    real_records = records["real"]
    result: dict[str, object] = {
        "analysis": "frozen_duplicate_backbone_vessel_feature_support_v1",
        "weights": str(args.weights),
        "feature_layers": list(FEATURE_LAYERS),
        "embedding_dimensions": int(real_embeddings.shape[1]),
        "imgsz": args.imgsz,
        "seed": args.seed,
        "mmd_implementation": {
            "estimator": "biased_rbf_mmd_squared",
            "kernel_mean": "exact_blockwise",
            "bandwidth_rule": "median_nonzero_pairwise_distance",
            "bandwidth_deterministic_subset_max": 4096,
        },
        "domains": {
            "real": {
                "count": len(real_records),
                "diversity": diversity_metrics(real_embeddings, rng),
            }
        },
        "test_split_used": False,
    }
    for name in records:
        if name == "real":
            continue
        support = nearest_real_metrics(embeddings[name], records[name], real_embeddings, real_records, rng)
        real_indices = matched_real_sample(records[name], real_records, rng)
        mmd, bandwidth = rbf_mmd_squared(embeddings[name], real_embeddings[real_indices])
        result["domains"][name] = {
            "count": len(records[name]),
            "support": support,
            "scale_matched_mmd_squared": mmd,
            "mmd_bandwidth": bandwidth,
            "diversity": diversity_metrics(embeddings[name], rng),
        }
    result["pca"] = plot_pca(args.output / "feature_pca.png", embeddings, rng)
    with (args.output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    (args.output / "report.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
