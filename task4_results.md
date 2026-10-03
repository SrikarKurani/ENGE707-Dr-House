# Task 4 results

Selected config (3-fold CV on training data): `{'learning_iterations': 10000, 'N': 1000, 'p_spec': 0.2}`

| Model | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| Task 2: raw data, leakage kept | 0.9248 | 0.9045 | 0.8559 | 0.9585 |
| Task 2: raw data, leakage removed | 0.8781 | 0.7917 | 0.7225 | 0.8896 |
| Original eLCS, Task 3 data | 0.8171 | 0.6665 | 0.5007 | 0.7575 |
| Improved eLCS, Task 3 data | 0.8792 | 0.8021 | 0.7337 | 0.9140 |
