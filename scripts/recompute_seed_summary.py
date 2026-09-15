"""CPU-only recomputation from public contrasts; does not recompute AP."""
from pathlib import Path
from collections import defaultdict
import csv
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
groups = 0
for model in ('n', 's', 'm'):
    paths = list((ROOT/f'factorial/{model}/statistics').glob('*per_seed_contrasts.csv'))
    assert len(paths) == 1
    grouped = defaultdict(list)
    with paths[0].open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            a, b = float(r['did_1e-4']), float(r['did_2e-4'])
            for lr, value in [('1e-4', a), ('2e-4', b)]:
                assert math.isclose(value, float(r['delta_head_'+lr])-float(r['delta_full_'+lr]), abs_tol=1e-12)
            assert math.isclose(float(r['marginal_did']), (a+b)/2, abs_tol=1e-12)
            assert math.isclose(float(r['three_way']), b-a, abs_tol=1e-12)
            grouped[(r['dataset'], r['metric_family'], r['metric'])].append((int(r['seed']), float(r['marginal_did'])))
    for key, pairs in sorted(grouped.items()):
        assert sorted(s for s, _ in pairs) == list(range(20260723,20260733))
        vals = [v for _,v in pairs]
        avg = statistics.mean(vals)
        half = 2.2621571628540993*statistics.stdev(vals)/math.sqrt(10)
        print(f'YOLO26{model},{",".join(key)},mean={avg:.9f},CI=[{avg-half:.9f},{avg+half:.9f}]')
        groups += 1
assert groups == 54, groups
print('PASS: 54 model/domain/family/metric groups; 540 seed rows; contrast identities valid')
