# SynLess experiment report

CLIP/CUB pilot; canonical-caption retrieval on the recorded reporting subset.

Loss deltas are treatment minus a matched native continuation; negative is helpful. R@1 deltas are fractions.

| Starting state / arm | Seeds | Mean loss delta | Seed SD | T→I R@1 delta | I→T R@1 delta |
|---|---:|---:|---:|---:|---:|
| 100:bids | 3 | -0.00141933 | 0.000844341 | +0.00195312 | +0 |
| 100:bids_no_normalization | 3 | -0.00128212 | 0.00090826 | +0.000651042 | +0 |
| 100:bottom_less | 3 | +0.000391019 | 0.000784027 | +0 | +0 |
| 100:hardest_real | 3 | +0.000437712 | 0.000186071 | +0.000651042 | +0 |
| 100:less_head_mean | 3 | -0.000433492 | 9.86777e-05 | +0.00325521 | +0 |
| 100:less_head_task_max | 3 | -0.00048391 | 0.000413859 | +0.00195312 | +0.000651042 |
| 100:native | 3 | +0 | 0 | +0 | +0 |
| 100:normalized_instance_max | 3 | -0.00119861 | 0.000206253 | +0.00260417 | -0.000651042 |
| 100:ot | 3 | -0.000860157 | 0.000302071 | +0 | +0 |
| 100:random_0 | 3 | -9.23177e-05 | 0.000366439 | +0 | +0.000651042 |
| 100:random_1 | 3 | -0.000268802 | 7.82597e-05 | +0 | +0.000651042 |
| 100:random_2 | 3 | -0.000232021 | 0.000912461 | +0.000651042 | +0 |
| 100:top_less | 3 | -4.8702e-06 | 0.000627447 | +0.00195312 | +0.000651042 |
| 100:top_marginal | 3 | -0.000513208 | 0.000928833 | +0.00130208 | +0 |
| 100:uniform | 3 | -0.000469419 | 0.00023282 | +0 | +0 |
| 500:bids | 3 | -0.000282767 | 0.000322676 | +0 | +0 |
| 500:bids_no_normalization | 3 | -0.000404092 | 0.000742354 | -0.000651042 | +0.000651042 |
| 500:bottom_less | 3 | -0.00158757 | 0.00134788 | -0.000651042 | -0.000651042 |
| 500:hardest_real | 3 | +0.000433209 | 0.000217898 | -0.000651042 | +0 |
| 500:less_head_mean | 3 | +0.000576906 | 0.000462529 | -0.00195312 | -0.000651042 |
| 500:less_head_task_max | 3 | +0.00125865 | 0.00045256 | -0.00195312 | -0.000651042 |
| 500:native | 3 | +0 | 0 | +0 | +0 |
| 500:normalized_instance_max | 3 | -0.000557592 | 0.000290707 | -0.00195312 | +0 |
| 500:ot | 3 | -0.000497895 | 0.000340608 | +0 | +0.000651042 |
| 500:random_0 | 3 | -0.000119212 | 0.000122816 | -0.00195312 | +0.000651042 |
| 500:random_1 | 3 | -0.000186428 | 0.000462128 | -0.00260417 | +0 |
| 500:random_2 | 3 | -0.000276806 | 0.000708463 | -0.00130208 | +0 |
| 500:top_less | 3 | +0.00116703 | 0.00100321 | +0 | -0.000651042 |
| 500:top_marginal | 3 | +0.00137772 | 0.000661367 | -0.000651042 | -0.000651042 |
| 500:uniform | 3 | -0.000178359 | 0.000130225 | -0.000651042 | +0 |

## Which constructions were selected?

Mean per-seed enrichment relative to candidate-pool prevalence; 1 means no preference.

| State / selector | OT | Uniform | Hardest real |
|---|---:|---:|---:|
| 100:top_less | 1.156 | 1.188 | 0.656 |
| 100:less_head_task_max | 1.125 | 1.094 | 0.781 |
| 100:bids | 1.500 | 1.438 | 0.062 |
| 100:top_marginal | 1.375 | 1.125 | 0.500 |
| 500:top_less | 1.156 | 1.562 | 0.281 |
| 500:less_head_task_max | 1.000 | 1.094 | 0.906 |
| 500:bids | 1.438 | 1.469 | 0.094 |
| 500:top_marginal | 1.156 | 1.438 | 0.406 |

Inspect each stage's selection_summary.json for construction enrichment and score distributions.
BIDS and less_head_* use the same exact projection-head attribution matrix; top_less uses all trainable parameters.
Training selectors have equal per-context capacities. Unconstrained BIDS/LESS selections are retained as diagnostics.
Random repeats share a training seed and are not independent seed replications.
Selection validation is disjoint from reporting. Prior SCSS use of CUB means the reporting set is not historically untouched.
A high rank is a prediction, not proof of improvement. This pilot does not establish full-dataset retrieval gains.
