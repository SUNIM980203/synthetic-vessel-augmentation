# Scene evaluator parity audit (v38)

## Conclusion

Exact parity with the canonical v46 AP is structurally unavailable without rerunning inference because the retained COCO prediction JSON rounds bounding-box coordinates to 0.001 pixel and confidence scores to five decimal places. The canonical evaluator computed AP before that serialization, from unrounded tensors. The retained JSON is sufficient for the scene-cluster sensitivity but must be labeled as a reconstructed scene-analysis metric.

## Implementation comparison

- The canonical v46 evaluator used Ultralytics 8.4.50 validation at confidence 0.001, NMS IoU 0.7, max detections 300, and computed per-class AP from in-memory predictions before JSON serialization.
- The scene evaluator uses the same 0.50:0.05:0.95 IoU grid, greedy IoU matching order, class-4/category-5 mapping, confidence ranking, 101-point interpolated precision envelope, negative images, and target set.
- Replacing COCO annotation boxes with the prepared normalized YOLO labels produced the same reconstruction, excluding annotation-coordinate conversion as the observed cause.
- With unit image weights, the weighted scene AP routine matches Ultralytics `ap_per_class` reconstruction to an absolute tolerance of 1e-12 for every audited condition and seed. Thus weighted integration is not the residual source.
- The remaining difference arises before AP integration: rounded serialized boxes can change matches near an IoU threshold, and rounded scores can introduce or alter confidence ties. Original unrounded per-prediction tensors were not retained.

## Numerical comparison

| Contrast | Canonical v46 effect | Rounded-JSON reconstruction | Reconstruction − canonical |
|---|---:|---:|---:|
| Unity − Duplicate | +0.007946996 | +0.007943418 | -0.000003578 |
| Unity − RealCutout | +0.002401645 | +0.002475732 | +0.000074087 |

Maximum absolute per-seed condition-level reconstruction discrepancies:

- `unity_medium150`: 0.000295052
- `duplicate`: 0.000452740
- `realcutout_medium150`: 0.000291851

## Required reporting boundary

Canonical v46 estimates remain the efficacy results in the main Results section. Scene resampling uses and is labeled `reconstructed scene-analysis AP`. The scene values must not replace canonical values. The automated unit-weight tolerance test passes; exact canonical parity is marked unavailable rather than fabricated.
