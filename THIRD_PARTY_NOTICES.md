# Third-party software notices

The archive contains selected research Python sources, not vendored dependencies, model weights or a full Unity project.

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics ; AGPL-3.0 / separately negotiated Enterprise terms (no Enterprise grant asserted). Historical evaluation code extends DetectionValidator and uses its box conversion/class mapping conventions; the archive's ExactJSONDetectionValidator serialization retains unrounded Python floating-point values and extra provenance fields. Upstream source: ultralytics/models/yolo/detect/val.py and ultralytics/utils/ops.py. Copyright remains with Ultralytics contributors. The AGPL license text is reproduced in LICENSE, from the installed distribution, without substantive alteration. Historical version requirements belong to the individual protocol; the installed 8.4.104 version is not asserted for every experiment.
- PyTorch/torchvision, NumPy, SciPy, Matplotlib, Pillow, OpenCV, pycocotools and other imported packages are external dependencies. Their source and binaries are not bundled; obtain them from their respective projects with their original licenses. No ownership claim is made over them.

No MIT license or proprietary relicensing is asserted for the Ultralytics-related code. All Python files are provided as source under the root code-license scope. Dependencies and private data needed to execute historical scripts are described as unavailable/not bundled, not as reproduced by the safe verification utilities.

The absence of an individual copyright header is not evidence that third-party rights are absent. These notices record observed dependency and adaptation information, not a comprehensive legal opinion.
