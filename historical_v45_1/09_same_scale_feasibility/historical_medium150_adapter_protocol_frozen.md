# Frozen historical Medium150 adapter protocol

- State: `FROZEN_ADAPTER_AND_DRY_RUN_ONLY`
- Frame mode: `historical_medium150`
- Adapter SHA-256: `d4aaaf1808033b85c70fd997d8076191a4c92eb25643dd57f318a86e1fa8d57c`
- Authoritative joined manifest SHA-256: `313a0b960fe0672f0cc9c917ba01dc69b2f008ce9c064171d65421b51bdaba31`
- Frozen 150-frame manifest SHA-256: `1173c772b56adbbc1cdda9943bfa5c27f5de83b73b58babc2d67eef07792764f`
- Technical input mapping: **150/150 PASS**
- Scientific core changed: **FALSE**
- Standard pipeline regression: **PASS**
- Appearance-orientation conflict: **TRUE** (`major_axis_reflection` variants)

The adapter freezes historical host, source raster, native orientation, center, and exact bbox geometry. Existing appearance, calibration, support, identity, integrity, and S3-selection logic remains byte-unchanged. Candidate generation remains locked pending explicit author resolution of the appearance-orientation conflict.
