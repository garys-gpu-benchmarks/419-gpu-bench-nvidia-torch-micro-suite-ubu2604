# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Runs local scripts/gemm.py and scripts/conv2d.py (PyTorch CUDA), not a TorchBench package and not ROCm. device_id, dtype (fp16), M/N/K, input_shape / image size, batch_size, warmup_iters, num_iterations, and seed come from yaml. kernel_type selects which script; the collector always runs both and writes gemm, conv2d, and summary rows Sweep dimensions: device_id, kernel_type, dtype, input_shape, M, N, K, batch_size.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| device_id | `--device-id` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| kernel_type | `--kernel-type` | smoke=all, baseline=all, extended=all | all | From Parameter list; see Execution Description With Parameters. |
| dtype | `--dtype` | smoke=fp16, baseline=fp16, extended=fp16 | fp16 | From Parameter list; see Execution Description With Parameters. |
| input_shape | `--input-shape` | smoke=1,3,32,32, baseline=1,3,224,224, extended=8,3,224,224 | 1,3,224,224 | From Parameter list; see Execution Description With Parameters. |
| M | `--m` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| N | `--n` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| K | `--k` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| batch_size | `--batch-size` | smoke=1, baseline=8, extended=8 | 8 | From Parameter list; see Execution Description With Parameters. |
| prompt_source | `--prompt-source` | smoke=synthetic, baseline=synthetic, extended=synthetic | synthetic | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=2, baseline=10, extended=20 | 10 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=5, baseline=501000, extended=1390000 | 501000 | From Parameter list; see Execution Description With Parameters. |
| seed | `--seed` | smoke=42, baseline=42, extended=42 | 42 | From Parameter list; see Execution Description With Parameters. |

### Derived quantities

Matrix extents come from the Parameters table. Example: use the tested value of `device_id` from Profile Parameter Values. Do not invent a leading dimension that the spec does not state.

## Invocation

```bash
Run scripts/gemm.py and scripts/conv2d.py via collect_torch_micro.py. Official TorchBench runners are not executed
```

## Raw Output Format

raw_results.csv, raw_results.jsonl, and gemm plus conv2d text in raw_output.txt

check,device_id,dtype,kernel_type,status,gemm_time_usec,gemm_tflops,gemm_bandwidth_gb_s,conv_time_usec,conv_tflops,conv_bandwidth_gb_s
summary,0,fp16,all,ok,15,20,600,25,12,400

## Metrics

- **#1: GEMM execution time, us** — stored as `gemm_time_usec`.
- **#2: GEMM throughput, TFLOPS** — stored as `gemm_tflops`.
- **#3: GEMM minimum traffic, GB/s** — stored as `gemm_bandwidth_gb_s`.
- **#4: Conv2d execution time, us** — stored as `conv_time_usec`.
- **#5: Conv2d throughput, TFLOPS** — stored as `conv_tflops`.
- **#6: Conv2d minimum traffic, GB/s** — stored as `conv_bandwidth_gb_s`.

## Framework

Runs local scripts/gemm.py and scripts/conv2d.py (PyTorch CUDA), not a TorchBench package and not ROCm. device_id, dtype (fp16), M/N/K, input_shape / image size, batch_size, warmup_iters, num_iterations, and seed come from yaml. kernel_type selects which script; the collector always runs both and writes gemm, conv2d, and summary rows

## Installation and Execution Summary

Run local scripts/gemm.py for GEMM (C = A x B) and scripts/conv2d.py for Conv2d on yaml M/N/K and input_shape, parse RESULT time/TFLOPS/bandwidth, and write gemm, conv2d, and summary CSV rows, to measure PyTorch CUDA microkernel performance. This is not TorchBench run.py

## Platform Portability

- **AMD (primary):** ```bash
Run scripts/gemm.py and scripts/conv2d.py via collect_torch_micro.py. Official TorchBench runners are not executed
```
- **NVIDIA:** Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

raw_results.csv, raw_results.jsonl, and gemm plus conv2d text in raw_output.txt

check,device_id,dtype,kernel_type,status,gemm_time_usec,gemm_tflops,gemm_bandwidth_gb_s,conv_time_usec,conv_tflops,conv_bandwidth_gb_s
summary,0,fp16,all,ok,15,20,600,25,12,400

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Runs local scripts/gemm.py and scripts/conv2d.py (PyTorch CUDA), not a TorchBench package and not ROCm. device_id, dtype (fp16), M/N/K, input_shape / image size, batch_size, warmup_iters, num_iterations, and seed come from yaml. kernel_type selects which script; the collector always runs both and writes gemm, conv2d, and summary rows
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Runs local scripts/gemm.py and scripts/conv2d.py (PyTorch CUDA), not a TorchBench package and not ROCm. device_id, dtype (fp16), M/N/K, input_shape / image size, batch_size, warmup_iters, num_iterations, and seed come from yaml. kernel_type selects which script; the collector always runs both and writes gemm, conv2d, and summary rows

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
