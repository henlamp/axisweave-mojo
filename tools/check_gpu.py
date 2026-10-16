"""Require real GPU execution, independent NumPy fixtures and NVIDIA compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import warnings
from pathlib import Path

import einops
import numpy as np
from _support import (
    BUILD,
    FIXTURES,
    MOJO,
    ROOT,
    run,
    source_hash,
    verification_manifest,
    write_report,
)
from check_properties import corpus, encode


def transform_cases():
    cases = corpus(seed=20261015, count=384)
    rng = np.random.default_rng(20261015)
    for case in cases:
        if case[0] != "valid":
            continue
        storage = rng.normal(size=len(case[8])).astype(np.float32)
        if case[2] == "prod":
            storage = np.float32(1) + storage * np.float32(0.03)
        tensor = np.ndarray(
            tuple(case[4]),
            dtype=np.float32,
            buffer=storage,
            offset=case[7] * 4,
            strides=tuple(s * 4 for s in case[6]),
        )
        if case[1] == "reduce":
            expected = einops.reduce(tensor, case[3], case[2], **case[5])
        else:
            expected = getattr(einops, case[1])(tensor, case[3], **case[5])
        case[8] = storage
        case[10] = expected.reshape(-1)
    for size in (0, 1, 127, 128, 129, 255, 256, 257, 1025):
        source = rng.normal(size=size).astype(np.float32)
        cases.append(
            [
                "valid",
                "rearrange",
                "sum",
                "a -> a",
                [size],
                {},
                [1],
                0,
                source,
                [size],
                source.copy(),
                f"tail-{size}",
            ]
        )
    for kind in ("sum", "mean", "prod"):
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = getattr(np, kind)(np.empty((3, 0), dtype=np.float32), axis=1)
        cases.append(
            [
                "valid",
                "reduce",
                kind,
                "a b -> a",
                [3, 0],
                {},
                [0, 1],
                0,
                np.empty(0, dtype=np.float32),
                [3],
                expected,
                f"empty-{kind}",
            ]
        )
    for kind in ("sum", "mean", "prod"):
        expected = [float("nan") if kind == "mean" else 1 if kind == "prod" else 0] * 4
        cases.append(
            [
                "valid",
                "reduce",
                kind,
                "a b -> a",
                [4, 0],
                {},
                [1073741824, 1],
                0,
                [],
                [4],
                expected,
                f"empty-large-stride-{kind}",
            ]
        )
    for kind in ("min", "max"):
        cases.append(
            [
                "invalid",
                "reduce",
                kind,
                "a b -> a",
                [3, 0],
                {},
                [0, 1],
                0,
                [],
                [3],
                [0, 0, 0],
                f"empty-{kind}",
            ]
        )
    return cases


def contraction_cases():
    rng = np.random.default_rng(20261016)
    definitions = [
        ([[2, 3], [3, 5]], "row k, k col -> row col"),
        ([[3], [5]], "row, col -> row col"),
        ([[], [2, 5]], ", row col -> row col"),
        ([[1, 3], [2, 3]], "batch k, batch k -> batch"),
        ([[2, 0], [0, 3]], "row k, k col -> row col"),
        ([[0, 3], [3, 4]], "row k, k col -> row col"),
        ([[2, 3], [2, 3]], "row k, row k -> k row"),
        ([[2, 3, 4], [1, 4, 5]], "... row k, ... k col -> ... row col"),
        ([[2, 3, 4], [4, 5]], "... row k, k col -> ... row col"),
        ([[2, 1, 3, 4], [5, 4, 2]], "... row k, ... k col -> ... row col"),
        ([[2, 3, 4, 5], [4, 5, 7]], "a b k l, k l c -> a b c"),
        ([[3, 2, 1, 4], [2, 1, 4, 5]], "a k l m, k l m b -> b a"),
        ([[2, 0, 3], [0, 3, 5]], "a k l, k l b -> a b"),
        ([[2, 1, 5], [1, 3, 5]], "a k l, k b l -> a b"),
        ([[2, 3], [3]], "a k, k ->"),
        ([[3, 3], [3]], "k k, k ->"),
        ([[2, 3, 3], [3]], "batch k k, k -> batch"),
        ([[], []], ", ->"),
    ]
    for _ in range(48):
        b, m, k, n = rng.integers(0, 9, size=4).tolist()
        definitions.append(([[b, m, k], [b, k, n]], "b m k, b k n -> b n m"))
    lines = []
    for shapes, pattern in definitions:
        inputs = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
        wanted = einops.einsum(*inputs, pattern)
        fields = [pattern]
        fields.extend(",".join(map(str, shape)) for shape in shapes)
        fields.extend(
            ",".join(format(float(v), ".17g") for v in x.flat) for x in inputs
        )
        fields.append(",".join(map(str, wanted.shape)))
        fields.append(",".join(format(float(v), ".17g") for v in wanted.flat))
        lines.append("|".join(fields))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--require-api", choices=("metal", "cuda"))
    parser.add_argument("--package-dir", help="Use installed/precompiled modules")
    args = parser.parse_args()
    if args.compile_only and args.require_api:
        parser.error("--require-api needs hardware execution")
    verified = verification_manifest()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    include = args.package_dir or ROOT / "src"
    compiler = run([MOJO, "--version"], "gpu-version")
    nvidia = {}
    for arch in ("sm_80", "sm_86", "sm_90", "sm_120"):
        output = BUILD / f"gpu-{arch}.s"
        run(
            [
                MOJO,
                "build",
                "--Werror",
                "--emit",
                "asm",
                "--target-accelerator",
                arch,
                "-I",
                include,
                "-I",
                ROOT / "tests",
                ROOT / "tests/gpu_references.mojo",
                "-o",
                output,
            ],
            f"gpu-compile-{arch}",
            timeout=600,
        )
        assembly = output.read_text()
        if f".target {arch}" not in assembly or assembly.count(".entry ") != 14:
            raise RuntimeError(
                f"Missing device PTX for all fourteen kernel variants: {arch}"
            )
        nvidia[arch] = {
            "ptx_kernels": 14,
            "status": "compiled",
            "hardware_execution": False,
            "assembly_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    report = {
        "mojo": compiler,
        "platform": platform.platform(),
        "verified_inputs": verified,
        "source_sha256": source_hash("axisweave"),
        "nvidia": nvidia,
        "numpy": np.__version__,
        "einops": einops.__version__,
        "hardware_execution": False,
    }
    if not args.compile_only:
        cases = transform_cases()
        transforms = FIXTURES / "gpu-transforms.txt"
        transforms.write_text("\n".join(encode(case) for case in cases) + "\n")
        contractions = contraction_cases()
        tensors = FIXTURES / "gpu-contractions.txt"
        tensors.write_text("\n".join(contractions) + "\n")
        executable = BUILD / "gpu-references"
        run(
            [
                MOJO,
                "build",
                "--Werror",
                "-I",
                include,
                "-I",
                ROOT / "tests",
                ROOT / "tests/gpu_references.mojo",
                "-o",
                executable,
            ],
            "gpu-references-build",
            timeout=600,
        )
        output = run(
            [executable, transforms, tensors], "gpu-references-run", timeout=600
        )
        print(output, flush=True)
        if (
            args.require_api
            and f"API: {args.require_api}" not in output.splitlines()[0]
        ):
            raise RuntimeError("The requested GPU backend was not executed")
        if "API: cuda" in output.splitlines()[0]:
            report["cuda_devices"] = run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ],
                "gpu-cuda-device",
            ).splitlines()
        report.update(
            {
                "hardware_execution": True,
                "device": output.splitlines()[0],
                "valid_transforms": sum(case[0] == "valid" for case in cases),
                "rejections": sum(case[0] == "invalid" for case in cases),
                "contractions": len(contractions),
                "fixtures_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (transforms, tensors)
                },
            }
        )
        application = run(
            [
                MOJO,
                "run",
                "--Werror",
                "-I",
                include,
                ROOT / "examples/gpu_sensor_features.mojo",
                "--values",
            ],
            "gpu-sensor-application",
            timeout=600,
        )
        image = (
            (np.arange(4 * 16 * 16 * 3) % 97 - 48).astype(np.float32) / 32
        ).reshape(4, 16, 16, 3)
        weights = ((np.arange(12 * 8) % 11 - 5).astype(np.float32) / 16).reshape(12, 8)
        patches = einops.rearrange(
            image,
            "batch (row ph) (col pw) channel -> (batch row col) (ph pw channel)",
            ph=2,
            pw=2,
        )
        wanted = np.einsum("ij,jk->ik", patches, weights, optimize=False).mean(axis=1)
        if not np.isfinite(wanted).all():
            raise RuntimeError("Non-finite independent sensor reference")
        np.testing.assert_allclose(
            json.loads(application)["values"], wanted, rtol=3e-5, atol=3e-5
        )
        report["sensor_pipeline"] = {
            "status": "passed",
            "numpy_outputs": len(wanted),
            "device_stages": 3,
        }
        for fixture in ("gpu_contracts",):
            print(
                run(
                    [
                        MOJO,
                        "run",
                        "--Werror",
                        "-I",
                        include,
                        "-I",
                        ROOT / "tests",
                        ROOT / "tests" / f"{fixture}.mojo",
                    ],
                    fixture,
                    timeout=600,
                ),
                flush=True,
            )
    if not args.compile_only:
        timings = json.loads(
            run(
                [
                    MOJO,
                    "run",
                    "--Werror",
                    "-I",
                    include,
                    ROOT / "tests/gpu_benchmark.mojo",
                ],
                "gpu-benchmark",
                timeout=600,
            )
        )
        timings.update(
            {
                "device": report["device"],
                "source_sha256": report["source_sha256"],
                "verified_inputs": verified,
                "allocation_and_compilation_timed": False,
                "status": "passed",
            }
        )
        timings["median_seconds"] = {
            key: statistics.median(timings[key])
            for key in (
                "cpu_seconds",
                "gpu_resident_seconds",
                "gpu_with_transfers_seconds",
            )
        }
        write_report(ROOT / "tests/gpu-benchmark.json", timings)
        print("Transpose median seconds:", timings["median_seconds"], flush=True)
    rejection = subprocess.run(
        [
            str(MOJO),
            "build",
            "--Werror",
            "-I",
            str(include),
            str(ROOT / "tests/reject_gpu_float64.mojo"),
            "-o",
            str(BUILD / "must-not-build-gpu"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    diagnostic = rejection.stdout + rejection.stderr
    (BUILD / "reject_gpu_float64.log").write_text(diagnostic)
    if (
        rejection.returncode == 0
        or "DeviceBuffer[.float64]" not in diagnostic
        or "DeviceBuffer[.float32]" not in diagnostic
    ):
        raise RuntimeError(
            "Float64 device input was not rejected for the expected reason"
        )
    report["float64_device_input"] = "rejected by type checker"
    lifetime = subprocess.run(
        [
            str(MOJO),
            "build",
            "--Werror",
            "-I",
            str(include),
            str(ROOT / "tests/reject_gpu_shape_lifetime.mojo"),
            "-o",
            str(BUILD / "must-not-build-gpu-shape"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    diagnostic = lifetime.stdout + lifetime.stderr
    (BUILD / "reject_gpu_shape_lifetime.log").write_text(diagnostic)
    if lifetime.returncode == 0 or "origin" not in diagnostic:
        raise RuntimeError("GPU shape metadata escaped its owning plan")
    report["shape_lifetime"] = "rejected by type checker"
    if args.package_dir:
        module = Path(args.package_dir) / "axisweave.mojoc"
        report["compiled_module_sha256"] = hashlib.sha256(
            module.read_bytes()
        ).hexdigest()
    if verified != verification_manifest():
        raise RuntimeError("GPU inputs changed during verification")
    report["status"] = "passed"
    destination = ROOT / "tests/gpu-verification.json"
    write_report(destination, report)
    print(f"GPU report: {destination}")


if __name__ == "__main__":
    main()
