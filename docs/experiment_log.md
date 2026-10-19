# Experiment log

Every run worth keeping, newest last (design §4): date, run name and id, config hash, device/dtype,
headline numbers with their interval, plus wall time, GPU-hours and API cost in the notes. Reruns
are grouped by `(config_hash, device, dtype)` (design §7.5).

| date | run_name | run_id | config_hash | device/dtype | headline | notes |
|---|---|---|---|---|---|---|
| 2026-10-18 | smoke_qwen_hellaswag | 20261019T030343Z-42a02e07 | 42a02e07f127 | mps/float32 | acc_norm 0.52 [0.42, 0.61] | Qwen2.5-0.5B @ 060db64, HellaSwag validation @ 218ec52, n=100, zero_shot log-likelihood; acc 0.44 [0.34, 0.54]; wall time 19.5 s (run.json started→finished; 20.1 s for `make smoke`) on Apple M5; code 419b272 (clean); 0 GPU-hours, $0 API |
