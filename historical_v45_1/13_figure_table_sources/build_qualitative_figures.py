from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "figures" / "qualitative_v29"

CONTROL_DIR = ROOT / "PreparedData" / "hybrid_xview_expanded_vessel_v10_control150"
UNITY_DIR = ROOT / "PreparedData" / "hybrid_xview_expanded_vessel_v10_medium150_scaled"
REALCUTOUT_DIR = ROOT / "PreparedData" / "hybrid_xview_expanded_vessel_realcontrol_v12_150"

HRSC_ROOT = ROOT / "datasets" / "HRSC2016_MS" / "raw_v1"
HRSC_PRED_ROOT = ROOT / "runs" / "hrsc_external_v13_prediction_export"
DIOR_ROOT = ROOT / "PreparedData" / "yolo_dior_v14"
DIOR_PRED_ROOT = ROOT / "runs" / "dior_external_v14_evaluation"

SEED = 20260723
SCORE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
CONDITIONS = ("duplicate", "realcutout_medium150", "unity_medium150")

GT_COLOR = "#FFD23F"
TP_COLOR = "#00C2D7"
FP_COLOR = "#F15A8A"
SITE_COLOR = "#FFFFFF"
EDGE_COLOR = "#101820"


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, w, h = map(float, box)
    return x, y, x + w, y + h


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def draw_box(
    ax,
    box: tuple[float, float, float, float],
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.2,
) -> None:
    x1, y1, x2, y2 = box
    for expansion, stroke_color, stroke_width in (
        (0.0, EDGE_COLOR, linewidth + 1.5),
        (0.0, color, linewidth),
    ):
        ax.add_patch(
            Rectangle(
                (x1 - expansion, y1 - expansion),
                x2 - x1 + 2 * expansion,
                y2 - y1 + 2 * expansion,
                fill=False,
                edgecolor=stroke_color,
                linewidth=stroke_width,
                linestyle=linestyle,
                zorder=5,
            )
        )
    if label:
        ax.text(
            x1,
            y1 + 3.0,
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": EDGE_COLOR, "edgecolor": color, "pad": 1.5, "alpha": 0.88},
            zorder=8,
        )


def square_view(
    focal_box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    minimum: float,
    maximum: float,
    multiplier: float,
) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = focal_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = min(maximum, max(minimum, multiplier * max(x2 - x1, y2 - y1)))
    left = min(max(0.0, cx - side / 2.0), max(0.0, width - side))
    top = min(max(0.0, cy - side / 2.0), max(0.0, height - side))
    right = min(float(width), left + side)
    bottom = min(float(height), top + side)
    return left, top, right, bottom


def intersects(box: tuple[float, float, float, float], view: tuple[float, float, float, float]) -> bool:
    return box[2] > view[0] and box[0] < view[2] and box[3] > view[1] and box[1] < view[3]


def add_image(ax, path: Path, view: tuple[float, float, float, float]) -> tuple[int, int]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        size = image.size
        ax.imshow(image)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[3], view[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#777777")
    return size


def save_figure(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def representative_training_records() -> list[dict]:
    unity = load_json(UNITY_DIR / "annotations" / "instances_train.json")
    records = [annotation for annotation in unity["annotations"] if annotation.get("source") == "unity_cutout"]
    records.sort(key=lambda item: math.sqrt(float(item["area"])))
    groups = [records[:50], records[50:100], records[100:]]
    selected = []
    for group_index, group in enumerate(groups):
        scales = sorted(math.sqrt(float(item["area"])) for item in group)
        placements = sorted(float(item["placement_score"]) for item in group)
        target_scale = scales[len(scales) // 2]
        target_placement = placements[len(placements) // 2]
        scale_span = max(scales) - min(scales) or 1.0
        placement_span = max(placements) - min(placements) or 1.0
        chosen = min(
            group,
            key=lambda item: (
                abs(math.sqrt(float(item["area"])) - target_scale) / scale_span
                + abs(float(item["placement_score"]) - target_placement) / placement_span,
                int(item["image_id"]),
            ),
        )
        selected.append(
            {
                "scale_group": ("lower", "middle", "upper")[group_index],
                "annotation_id": int(chosen["id"]),
                "image_id": int(chosen["image_id"]),
                "bbox_xywh": [float(value) for value in chosen["bbox"]],
                "scale_px": math.sqrt(float(chosen["area"])),
                "placement_score": float(chosen["placement_score"]),
            }
        )
    return selected


def build_training_input_figure() -> list[dict]:
    selected = representative_training_records()
    unity_coco = load_json(UNITY_DIR / "annotations" / "instances_train.json")
    real_coco = load_json(REALCUTOUT_DIR / "annotations" / "instances_train.json")
    control_coco = load_json(CONTROL_DIR / "annotations" / "instances_train.json")
    unity_images = {int(item["id"]): item for item in unity_coco["images"]}
    real_images = {int(item["id"]): item for item in real_coco["images"]}
    control_images = {int(item["id"]): item for item in control_coco["images"]}

    fig, axes = plt.subplots(3, 3, figsize=(11.8, 10.0))
    headers = ("Clean host", "Unity Medium150", "RealCutout Medium150")
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, pad=8, fontweight="bold")

    for row, record in enumerate(selected):
        image_id = record["image_id"]
        box = xywh_to_xyxy(record["bbox_xywh"])
        paths = (
            CONTROL_DIR / "images" / "train" / control_images[image_id]["file_name"],
            UNITY_DIR / "images" / "train" / unity_images[image_id]["file_name"],
            REALCUTOUT_DIR / "images" / "train" / real_images[image_id]["file_name"],
        )
        with Image.open(paths[0]) as image:
            view = square_view(box, image.size, minimum=130.0, maximum=250.0, multiplier=2.4)
        record["view_xyxy"] = [round(value, 3) for value in view]
        record["paths"] = [str(path.relative_to(ROOT)) for path in paths]
        for column, path in enumerate(paths):
            ax = axes[row, column]
            add_image(ax, path, view)
            if column == 0:
                draw_box(ax, box, SITE_COLOR, "Insertion site", linestyle="--", linewidth=2.0)
            else:
                draw_box(ax, box, GT_COLOR, f"{record['scale_px']:.1f} px", linewidth=2.2)
        axes[row, 0].set_ylabel(
            f"{record['scale_group'].capitalize()} scale tercile\nimage {image_id}",
            rotation=90,
            labelpad=8,
            fontweight="bold",
        )

    fig.legend(
        handles=[
            Line2D([0], [0], color=SITE_COLOR, linestyle="--", linewidth=2.5, marker="s", markeredgecolor=EDGE_COLOR, markerfacecolor="none", label="Matched insertion site"),
            Line2D([0], [0], color=GT_COLOR, linewidth=3.0, label="Matched insertion box"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.955, bottom=0.065, wspace=0.035, hspace=0.08)
    save_figure(fig, "fig_training_input_examples_v29")
    return selected


def load_prediction_sets(root: Path) -> dict[str, dict[str, list[dict]]]:
    output: dict[str, dict[str, list[dict]]] = {}
    for condition in CONDITIONS:
        rows = load_json(root / f"{condition}_s{SEED}" / "predictions.json")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if int(row.get("category_id", -1)) != 5:
                continue
            grouped[str(row["image_id"])].append(
                {
                    "score": float(row["score"]),
                    "box": xywh_to_xyxy(row["bbox"]),
                    "file_name": row.get("file_name"),
                }
            )
        output[condition] = grouped
    return output


def match_predictions(gt_boxes: list[tuple[float, float, float, float]], predictions: list[dict]) -> dict:
    used: set[int] = set()
    evaluated = []
    for prediction in sorted(
        (item for item in predictions if item["score"] >= SCORE_THRESHOLD),
        key=lambda item: item["score"],
        reverse=True,
    ):
        best_index = None
        best_iou = 0.0
        for index, gt_box in enumerate(gt_boxes):
            if index in used:
                continue
            overlap = box_iou(prediction["box"], gt_box)
            if overlap > best_iou:
                best_index = index
                best_iou = overlap
        status = "fp"
        if best_index is not None and best_iou >= IOU_THRESHOLD:
            used.add(best_index)
            status = "tp"
        evaluated.append(
            {
                "score": prediction["score"],
                "box": prediction["box"],
                "status": status,
                "matched_gt": best_index if status == "tp" else None,
                "iou": best_iou,
            }
        )
    return {"predictions": evaluated, "matched_gt": used, "unmatched_gt": set(range(len(gt_boxes))) - used}


def load_hrsc_ground_truth() -> tuple[dict[str, list[tuple[float, float, float, float]]], dict[str, Path]]:
    ids = HRSC_ROOT.joinpath("ImageSets", "test.txt").read_text(encoding="utf-8").split()
    ground_truth = {}
    images = {}
    for raw_id in ids:
        root = ET.parse(HRSC_ROOT / "Annotations" / f"{raw_id}.xml").getroot()
        boxes = []
        for node in root.findall(".//bndbox"):
            boxes.append(tuple(float(node.find(key).text) for key in ("xmin", "ymin", "xmax", "ymax")))
        image_id = str(int(raw_id))
        ground_truth[image_id] = boxes
        images[image_id] = HRSC_ROOT / "AllImages" / f"{raw_id}.bmp"
    return ground_truth, images


def load_dior_ground_truth() -> tuple[dict[str, list[tuple[float, float, float, float]]], dict[str, Path]]:
    ground_truth = {}
    images = {}
    for image_path in sorted((DIOR_ROOT / "images" / "test").glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = []
        label_path = DIOR_ROOT / "labels" / "test" / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                category, xc, yc, bw, bh = map(float, line.split())
                if int(category) != 4:
                    continue
                boxes.append(
                    (
                        (xc - bw / 2.0) * width,
                        (yc - bh / 2.0) * height,
                        (xc + bw / 2.0) * width,
                        (yc + bh / 2.0) * height,
                    )
                )
        ground_truth[image_path.stem] = boxes
        images[image_path.stem] = image_path
    return ground_truth, images


def natural_key(value: str):
    suffix = value.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else value


def select_external_case(
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    predictions: dict[str, dict[str, list[dict]]],
    rule: str,
) -> tuple[dict, dict[str, dict]]:
    candidates = []
    evaluations_by_image = {}
    for image_id, gt_boxes in ground_truth.items():
        evaluations = {
            condition: match_predictions(gt_boxes, predictions[condition].get(image_id, []))
            for condition in CONDITIONS
        }
        evaluations_by_image[image_id] = evaluations
        duplicate = evaluations["duplicate"]
        realcutout = evaluations["realcutout_medium150"]
        unity = evaluations["unity_medium150"]
        if rule == "unity_recovery":
            focal = sorted(unity["matched_gt"] - duplicate["matched_gt"] - realcutout["matched_gt"])
        elif rule == "unity_failure":
            focal = sorted(duplicate["matched_gt"] & realcutout["matched_gt"] - unity["matched_gt"])
        elif rule == "unity_only_fp_negative":
            focal = []
            duplicate_fp = [item for item in duplicate["predictions"] if item["status"] == "fp"]
            realcutout_fp = [item for item in realcutout["predictions"] if item["status"] == "fp"]
            unity_fp = [item for item in unity["predictions"] if item["status"] == "fp"]
            if not gt_boxes and unity_fp and not duplicate_fp and not realcutout_fp:
                focal = [max(range(len(unity_fp)), key=lambda index: unity_fp[index]["score"])]
        else:
            raise ValueError(f"Unknown selection rule: {rule}")
        if focal:
            candidates.append((image_id, focal[0]))
    candidates.sort(key=lambda item: natural_key(item[0]))
    if not candidates:
        raise RuntimeError(f"No candidates for rule {rule}")
    image_id, focal_index = candidates[len(candidates) // 2]
    if rule == "unity_only_fp_negative":
        unity_fp = [
            item
            for item in evaluations_by_image[image_id]["unity_medium150"]["predictions"]
            if item["status"] == "fp"
        ]
        focal_box = unity_fp[focal_index]["box"]
        focal_score = unity_fp[focal_index]["score"]
        focal_gt = None
    else:
        focal_box = ground_truth[image_id][focal_index]
        focal_score = None
        focal_gt = focal_index
    return (
        {
            "rule": rule,
            "eligible_image_count": len(candidates),
            "selection_index": len(candidates) // 2,
            "image_id": image_id,
            "focal_gt_index": focal_gt,
            "focal_score": focal_score,
            "focal_box_xyxy": [round(value, 3) for value in focal_box],
        },
        evaluations_by_image[image_id],
    )


def draw_detection_panel(
    ax,
    image_path: Path,
    view: tuple[float, float, float, float],
    gt_boxes: list[tuple[float, float, float, float]],
    evaluation: dict | None,
    focal_gt_index: int | None,
) -> None:
    add_image(ax, image_path, view)
    if evaluation is None:
        visible_indices = [focal_gt_index] if focal_gt_index is not None else range(len(gt_boxes))
        for index in visible_indices:
            gt_box = gt_boxes[index]
            if intersects(gt_box, view):
                draw_box(ax, gt_box, GT_COLOR, "Focal GT", linewidth=2.3)
        return
    if focal_gt_index is not None:
        gt_box = gt_boxes[focal_gt_index]
        if focal_gt_index in evaluation["unmatched_gt"]:
            draw_box(ax, gt_box, GT_COLOR, "FN", linestyle="--", linewidth=2.1)
        for prediction in evaluation["predictions"]:
            if prediction["status"] == "tp" and prediction["matched_gt"] == focal_gt_index:
                draw_box(ax, prediction["box"], TP_COLOR, f"TP {prediction['score']:.2f}", linewidth=2.3)
        return
    for gt_index in sorted(evaluation["unmatched_gt"]):
        gt_box = gt_boxes[gt_index]
        if intersects(gt_box, view):
            draw_box(ax, gt_box, GT_COLOR, "FN", linestyle="--", linewidth=2.1)
    for prediction in evaluation["predictions"]:
        if not intersects(prediction["box"], view):
            continue
        if prediction["status"] == "tp":
            draw_box(ax, prediction["box"], TP_COLOR, f"TP {prediction['score']:.2f}", linewidth=2.3)
        else:
            draw_box(ax, prediction["box"], FP_COLOR, f"FP {prediction['score']:.2f}", linewidth=2.3)


def build_external_detection_figure() -> list[dict]:
    hrsc_gt, hrsc_images = load_hrsc_ground_truth()
    dior_gt, dior_images = load_dior_ground_truth()
    hrsc_predictions = load_prediction_sets(HRSC_PRED_ROOT)
    dior_predictions = load_prediction_sets(DIOR_PRED_ROOT)

    specifications = (
        ("HRSC2016-MS", "Unity TP; controls FN", hrsc_gt, hrsc_images, hrsc_predictions, "unity_recovery"),
        ("HRSC2016-MS", "Unity FN; controls TP", hrsc_gt, hrsc_images, hrsc_predictions, "unity_failure"),
        ("DIOR mirror", "Unity TP; controls FN", dior_gt, dior_images, dior_predictions, "unity_recovery"),
        ("DIOR mirror", "Unity-only FP on negative image", dior_gt, dior_images, dior_predictions, "unity_only_fp_negative"),
    )

    selected = []
    rows = []
    for dataset_name, label, ground_truth, images, predictions, rule in specifications:
        record, evaluations = select_external_case(ground_truth, predictions, rule)
        image_id = record["image_id"]
        image_path = images[image_id]
        with Image.open(image_path) as image:
            maximum = min(float(max(image.size)), 500.0)
            view = square_view(tuple(record["focal_box_xyxy"]), image.size, minimum=200.0, maximum=maximum, multiplier=2.5)
        record.update(
            {
                "dataset": dataset_name,
                "row_label": label,
                "image_path": str(image_path.relative_to(ROOT)),
                "view_xyxy": [round(value, 3) for value in view],
                "ground_truth_count": len(ground_truth[image_id]),
                "overlay_scope": "focal target only" if record["focal_gt_index"] is not None else "all displayed false positives",
            }
        )
        record["condition_counts"] = {
            condition: {
                "tp": sum(item["status"] == "tp" for item in evaluations[condition]["predictions"]),
                "fp": sum(item["status"] == "fp" for item in evaluations[condition]["predictions"]),
                "fn": len(evaluations[condition]["unmatched_gt"]),
            }
            for condition in CONDITIONS
        }
        selected.append(record)
        rows.append((record, ground_truth[image_id], image_path, evaluations, view))

    fig, axes = plt.subplots(4, 4, figsize=(13.8, 13.0))
    headers = ("Ground truth", "Duplicate", "RealCutout", "Unity")
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, pad=8, fontweight="bold")

    for row_index, (record, gt_boxes, image_path, evaluations, view) in enumerate(rows):
        focal_gt_index = record["focal_gt_index"]
        draw_detection_panel(axes[row_index, 0], image_path, view, gt_boxes, None, focal_gt_index)
        draw_detection_panel(axes[row_index, 1], image_path, view, gt_boxes, evaluations["duplicate"], focal_gt_index)
        draw_detection_panel(axes[row_index, 2], image_path, view, gt_boxes, evaluations["realcutout_medium150"], focal_gt_index)
        draw_detection_panel(axes[row_index, 3], image_path, view, gt_boxes, evaluations["unity_medium150"], focal_gt_index)
        axes[row_index, 0].set_ylabel(
            f"{record['dataset']}\n{record['row_label']}\n{record['image_id']}",
            rotation=90,
            labelpad=8,
            fontweight="bold",
        )

    fig.legend(
        handles=[
            Line2D([0], [0], color=GT_COLOR, linestyle="--", linewidth=3.0, label="FN: unmatched ground truth"),
            Line2D([0], [0], color=TP_COLOR, linewidth=3.0, label="TP: IoU >= 0.50"),
            Line2D([0], [0], color=FP_COLOR, linewidth=3.0, label="FP"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.003),
    )
    fig.text(
        0.995,
        0.01,
        f"Seed {SEED}; display score >= {SCORE_THRESHOLD:.2f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.96, bottom=0.07, wspace=0.035, hspace=0.08)
    save_figure(fig, "fig_external_detection_examples_v29")
    return selected


def write_support_files(training_records: list[dict], external_records: list[dict]) -> None:
    metadata = {
        "version": "v29",
        "selection_policy": {
            "training_inputs": (
                "Split all 150 accepted Medium150 insertions into three scale-sorted groups of 50. "
                "Within each group, choose the case nearest the group medians of box scale and placement score. "
                "No detector result was used."
            ),
            "external_predictions": (
                "Use the first prespecified optimization seed (20260723), score >= 0.25, and IoU >= 0.50. "
                "For each prespecified qualitative category, sort all eligible image identifiers and select the median identifier. "
                "No AP-contribution or effect-size maximization was used."
            ),
        },
        "training_input_examples": training_records,
        "external_detection_examples": external_records,
    }
    (OUT / "qualitative_selection_metadata_v29.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    captions = """# Qualitative figure captions v29

## Training-input comparison

**Matched training-input examples for the controlled exposure conditions.** Rows represent the lower, middle, and upper scale terciles of the 150 accepted Medium150 insertions. Within each tercile, the displayed example was selected deterministically as the case nearest the tercile medians of bounding-box scale and placement score; detector outcomes were not used. Columns show the clean xView host, the Unity Medium150 insertion, and the geometry-matched RealCutout insertion at the identical host location and box dimensions. Crops are enlarged for visibility; the detector received the complete 512 x 512 training patch.

## External-detection comparison

**Rule-selected qualitative vessel detections on HRSC2016-MS and the fixed DIOR public-mirror split.** Columns show ground truth and predictions from the Duplicate, RealCutout Medium150, and Unity Medium150 models for the first prespecified optimization seed (20260723). Predictions are displayed at score >= 0.25 and matched at IoU >= 0.50. Cyan boxes are true positives, magenta boxes are false positives, and dashed yellow boxes are false negatives. Positive-image rows display only the rule-designated focal vessel and its matched prediction or false-negative box; unrelated objects are omitted from the overlay for legibility. For each prespecified category, all eligible image identifiers were sorted and the median identifier was selected; no image was selected by maximizing AP contribution or the model difference. The panels deliberately include both Unity-favorable recoveries and Unity-unfavorable false-negative/false-positive cases.

## 훈련 입력 비교(한글)

**통제된 노출 조건의 위치·기하 일치 훈련 입력 예시.** 각 행은 승인된 Medium150 삽입 150개의 하위, 중간 및 상위 크기 삼분위군을 나타낸다. 각 삼분위군에서는 검출 결과를 사용하지 않고 바운딩박스 크기와 배치 점수가 해당 군의 중앙값에 가장 가까운 사례를 결정론적으로 선택하였다. 열은 깨끗한 xView 호스트 영상, Unity Medium150 삽입 영상, 그리고 동일한 호스트 위치와 박스 크기를 사용한 RealCutout 삽입 영상을 나타낸다. 가시성을 위해 일부 영역을 확대했으며, 실제 검출기 학습에는 전체 512 x 512 패치가 사용되었다.

## 외부 검출 비교(한글)

**HRSC2016-MS 및 고정 DIOR 공개 미러 분할에서 규칙 기반으로 선택한 선박 검출 예시.** 열은 정답과 첫 번째 사전 지정 최적화 시드(20260723)의 Duplicate, RealCutout Medium150 및 Unity Medium150 예측을 나타낸다. 표시 임계값은 score >= 0.25이며 IoU >= 0.50에서 정답과 매칭하였다. 청록색은 true positive, 자홍색은 false positive, 노란색 점선은 false negative를 나타낸다. 양성 영상 행에서는 가독성을 위해 선택 규칙이 지정한 초점 선박과 이에 대응하는 예측 또는 미검출 박스만 표시하였다. 각 범주에서 적격 영상 식별자를 정렬한 후 중앙 식별자를 선택했으며, AP 기여도나 모델 간 차이를 최대화하여 영상을 고르지 않았다. Unity에 유리한 회복 사례와 Unity에 불리한 미검출·오검출 사례를 모두 포함하였다.
"""
    (OUT / "figure_captions_v29.md").write_text(captions, encoding="utf-8")


def main() -> None:
    configure_plotting()
    OUT.mkdir(parents=True, exist_ok=True)
    training_records = build_training_input_figure()
    external_records = build_external_detection_figure()
    write_support_files(training_records, external_records)
    print(json.dumps({"output": str(OUT), "training": training_records, "external": external_records}, indent=2))


if __name__ == "__main__":
    main()
