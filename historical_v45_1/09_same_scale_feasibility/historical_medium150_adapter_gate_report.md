# Historical Medium150 adapter gate report

Historical frames loaded: **150 / 150**

Host hashes verified: **150 / 150**

Source cutout hashes verified: **150 / 150**

Centers mapped exactly: **150 / 150**

BBox geometry mapped exactly: **150 / 150**

Native orientation mapped at pre-render boundary: **150 / 150**

Automatic source reselection bypassed: **TRUE**

Automatic source rotation bypassed: **TRUE**

Scientific core changed: **FALSE**

Standard pipeline regression: **PASS**

Appearance-orientation conflict: **TRUE**

## Overall

```text
CASE = CASE_E_APPEARANCE_ORIENTATION_CONFLICT
TECHNICAL_150_ROW_MAPPING_PASS = TRUE
HISTORICAL_FRAME_ADAPTER_READY = FALSE
READY_FOR_MEDIUM150_CANDIDATE_GATE = FALSE
BLOCKER = APPEARANCE_ORIENTATION_CONFLICT
CANDIDATES_GENERATED = 0
DETECTOR_TRAINING_RUNS = 0
AP_VALUES_INSPECTED = 0
EFFICACY_CHECKPOINTS_CREATED = 0
```

The historical input binding itself maps 150/150. The full gate remains closed because the frozen `major_axis_reflection` variants explicitly flip internal vessel structure along the major axis. The adapter did not remove or alter those families. This semantic compatibility question requires explicit author review before any candidate-gate phase.
