# Results: 419 PyTorch Microkernel Suite

**Main goal:** Measure PyTorch GEMM and Conv2d microkernel time, TFLOPS, and bandwidth  
**Tools:** Run the PyTorch microkernel suite through scripts/gemm.py and scripts/conv2d.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_146 · recorded 2026-11-18). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 NVL (95830 MiB) · AMD EPYC 9374F 32-Core Processor · Ubuntu 26.04.1 LTS · kernel 6.8.0-138-generic · driver 595.112.07
- **Run:** 2026-11-18T11:01:52Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:02 · benchmark 00:05 · whole run 01:16
- **Source:** run 419_20261118_110152
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype fp16 --input-shape 1,3,32,32 --M 64 --N 64 --K 64 --warmup-iters 2 --num-iterations 5 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | GEMM execution time, us (gemm_time_usec) | 13.190199 |
| 2 | GEMM throughput, TFLOPS (gemm_tflops) | 0.039748 |
| 3 | GEMM minimum traffic, GB/s (gemm_bandwidth_gb_s) | 1.863202 |
| 4 | Conv2d execution time, us (conv_time_usec) | 12.506201 |
| 5 | Conv2d throughput, TFLOPS (conv_tflops) | 0.035372 |
| 6 | Conv2d minimum traffic, GB/s (conv_bandwidth_gb_s) | 1.835889 |
<!-- /profile:smoke -->
