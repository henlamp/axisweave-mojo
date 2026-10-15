# Validation

Native macOS ARM64 and Linux x86-64 checks cover Mojo 1.1.0 and
1.2.0.dev2026100905. Prebuilt packages currently target macOS ARM64.

Checks compare einops/NumPy results, shapes, aliases, reductions and lifetime rejection.
Executable examples and installed consumers are checked separately.
Exact source identities, compiler versions and reports: [CI artifacts](https://github.com/henlamp/axisweave-mojo/actions/workflows/checks.yml).

`uv run python tools/check_gpu.py --require-api metal` requires real Apple GPU
execution and compares Float32 against CPU and independently generated NumPy/einops
fixtures, including strided/empty tensors, aliases and queued lifetimes. The sensor
pipeline checks device-resident composition; Float64 CPU precision and Float64 GPU
rejection are tested separately. `--compile-only` checks both index widths of all seven kernels for
NVIDIA sm_80/sm_90/sm_120 and records that no GPU executed. Normal CI uses this mode.
The manual GPU workflow requires a registered `metal` or `cuda` self-hosted runner.

`tests/gpu-verification.json` records current input hashes and hardware identity;
`tests/gpu-benchmark.json` separates warmed GPU execution from transfers and CPU
execution. Timings have no pass threshold and imply no general speedup.
