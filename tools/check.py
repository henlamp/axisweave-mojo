"""Generate independent NumPy/einops fixtures, compile and run native Mojo.

Python is a development oracle only. The generated executable uses no Python.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import einops
import numpy as np
from _support import source_hash

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".build"
FIXTURES = BUILD / "fixtures"
PACKAGES = BUILD / "packages"
MOJO = Path(sys.executable).parent / "mojo"
RNG = np.random.default_rng(20261008)


def literal(value):
    value = float(value)
    if math.isnan(value):
        return "(Float64(0) / Float64(0))"
    if math.isinf(value):
        return "(Float64(%s) / Float64(0))" % ("1" if value > 0 else "-1")
    return repr(value)


def mojo_list(values, element_type="Float64"):
    return (
        "["
        + ", ".join(
            str(int(v)) if element_type == "Int" else literal(v) for v in values
        )
        + "]"
    )


def main():
    BUILD.mkdir(exist_ok=True)
    FIXTURES.mkdir(exist_ok=True)
    PACKAGES.mkdir(exist_ok=True)
    cases, invalid = [], []

    def add(
        operation,
        shape,
        pattern,
        lengths=None,
        reduction="sum",
        dtype=np.float64,
        values=None,
        strides=None,
        offset=0,
    ):
        shape = tuple(shape)
        if values is None:
            values = RNG.normal(0.4, 0.3, size=math.prod(shape)).astype(dtype)
        storage = np.asarray(values, dtype=dtype).reshape(-1)
        if strides is None:
            array = storage.reshape(shape)
        else:
            array = np.ndarray(
                shape,
                dtype=dtype,
                buffer=storage,
                offset=offset * storage.itemsize,
                strides=tuple(s * storage.itemsize for s in strides),
            )
        lengths = dict(lengths or {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if operation == "reduce":
                expected = einops.reduce(array, pattern, reduction, **lengths)
            else:
                expected = getattr(einops, operation)(array, pattern, **lengths)
        cases.append(
            (
                operation,
                shape,
                pattern,
                lengths,
                reduction,
                dtype,
                storage,
                np.asarray(expected),
                strides,
                offset,
            )
        )

    for permutation in itertools.permutations("abcd"):
        add("rearrange", [2, 3, 2, 2], "a b c d -> " + " ".join(permutation))
    for dtype in (np.float64, np.float32):
        for rank in range(1, 5):
            names = list("abcd"[:rank])
            for _ in range(5):
                sizes = {name: int(RNG.integers(1, 4)) for name in names}
                shuffled = list(RNG.permutation(names))
                shape = [sizes[n] for n in names]
                add(
                    "rearrange",
                    shape,
                    " ".join(names) + " -> (" + " ".join(shuffled) + ")",
                    dtype=dtype,
                )
                add(
                    "rearrange",
                    [math.prod(shape)],
                    "(" + " ".join(names) + ") -> " + " ".join(shuffled),
                    {n: sizes[n] for n in names[:-1]},
                    dtype=dtype,
                )
        for reduction in ("sum", "mean", "min", "max", "prod"):
            for pattern in (
                "b h w c -> b c",
                "b h w c -> c b",
                "b h w c ->",
                "b h w c -> (b h) c",
                "b (h ph) (w pw) c -> b h w c",
            ):
                add(
                    "reduce",
                    [2, 4, 4, 2],
                    pattern,
                    {"ph": 2, "pw": 2} if "ph" in pattern else None,
                    reduction,
                    dtype,
                )
        for n in (0, 1, 3):
            for pattern in ("b c -> b c k", "b c -> k b c", "b c -> (b k) c"):
                add("repeat", [2, 3], pattern, {"k": n}, dtype=dtype)

    for operation, shape, pattern, lengths in [
        ("rearrange", [2, 3, 4], "b ... -> ... b", {}),
        ("rearrange", [2, 3, 4, 2], "b ... c -> b (...) c", {}),
        ("rearrange", [2, 3, 4], "... -> (...) ", {}),
        ("rearrange", [3], "b ... -> ... b", {}),
        ("rearrange", [], "... -> (...) ", {}),
        ("rearrange", [], "-> 1", {}),
        ("rearrange", [1, 3, 1], "1 b 1 -> b", {}),
        ("rearrange", [3], "b -> () b 1", {}),
        ("repeat", [2, 3, 4], "b ... -> b ... k", {"k": 2}),
        ("repeat", [2, 3], "... c -> k ... c", {"k": 3}),
        ("repeat", [], "-> b c", {"b": 2, "c": 3}),
        ("reduce", [2, 3, 4], "b ... -> b", {}),
        ("reduce", [2, 3, 4], "... ->", {}),
        ("rearrange", [2, 0, 3], "a b c -> c a b", {}),
        ("rearrange", [0], "(a b) -> a b", {"a": 2}),
        ("reduce", [2, 0], "a b -> a", {}),
    ]:
        add(operation, shape, pattern, lengths)
    add("reduce", [2, 0], "a b -> a", reduction="mean")
    add("reduce", [2, 0], "a b -> a", reduction="prod")
    for reduction in ("sum", "mean", "min", "max", "prod"):
        add(
            "reduce",
            [2, 3],
            "a b -> a",
            reduction=reduction,
            values=[1, np.nan, 3, 4, 5, 6],
        )
    add("reduce", [2, 2], "a b -> a", reduction="min", values=[0.0, -0.0, -0.0, 0.0])
    add("reduce", [2, 2], "a b -> a", reduction="max", values=[0.0, -0.0, -0.0, 0.0])
    add("rearrange", [3, 2], "a b -> b a", strides=[1, 3], values=np.arange(6))
    add(
        "rearrange",
        [2, 3],
        "a b -> (b a)",
        strides=[6, 2],
        offset=1,
        values=np.arange(12),
    )
    add("repeat", [2, 3], "a b -> b k a", {"k": 2}, strides=[0, 1], values=np.arange(3))
    add(
        "reduce",
        [3, 2],
        "a b -> a",
        reduction="sum",
        strides=[1, 3],
        values=np.arange(6),
    )
    add("rearrange", [2, 3], "a b -> b a", dtype=np.int32, values=np.arange(6))
    add("repeat", [2, 3], "a b -> a k b", {"k": 2}, dtype=np.int32, values=np.arange(6))

    for operation, shape, pattern, lengths in [
        ("rearrange", [2, 3], "a a -> a", {}),
        ("rearrange", [2, 3], "a b -> a a b", {}),
        ("rearrange", [2, 3], "a b -> a", {}),
        ("rearrange", [2, 3], "a b -> a b c", {"c": 2}),
        ("rearrange", [6], "(a b) -> a b", {}),
        ("rearrange", [7], "(a b) -> a b", {"b": 2}),
        ("rearrange", [2, 3], "a -> a", {}),
        ("rearrange", [2, 3], "a b a b", {}),
        ("rearrange", [2, 3], "a b -> a b -> a b", {}),
        ("rearrange", [2, 3], "a (b -> a b", {}),
        ("rearrange", [2, 3], "a b) -> a b", {}),
        ("rearrange", [6], "((a b)) -> a b", {"b": 2}),
        ("rearrange", [2, 3], "a b -> ... a b", {}),
        ("rearrange", [2, 3], "... ... -> ...", {}),
        ("rearrange", [2, 3], "a (...) -> a ...", {}),
        ("rearrange", [2, 3], "a b -> a b", {"z": 1}),
        ("rearrange", [2, 3], "a b -> a b", {"a": 4}),
        ("repeat", [2, 3], "a b -> a b c", {}),
        ("repeat", [2, 3], "a b -> a c", {"c": 2}),
        ("reduce", [2, 3], "a b -> a c", {"c": 2}),
        ("rearrange", [2, 3], "_a b -> b _a", {}),
    ]:
        array = np.zeros(shape)
        try:
            if operation == "reduce":
                einops.reduce(array, pattern, "sum", **lengths)
            else:
                getattr(einops, operation)(array, pattern, **lengths)
        except (einops.EinopsError, ValueError):
            invalid.append((operation, shape, pattern, lengths))
        else:
            raise AssertionError(f"Invalid fixture accepted by einops: {pattern}")

    for operation, shape, pattern, reduction in [
        ("reduce", [4, 6], "(h 2) (w 3) -> h w", "mean"),
        ("reduce", [2, 4, 6], "b (h 2) (w 3) -> b h w", "sum"),
        ("reduce", [2, 0, 6], "b (h 2) (w 3) -> b h w", "mean"),
        ("repeat", [2, 3], "h w -> h w 2", "sum"),
        ("repeat", [2, 3], "h w -> (h 2) (w 3)", "sum"),
        ("repeat", [], "-> 2 3", "sum"),
    ]:
        add(operation, shape, pattern, reduction=reduction)

    source = [
        "from std.collections import Dict, List, Span",
        "from std.testing import assert_equal, assert_true",
        "from axisweave import plan, StaticRecipe",
        "from reference_checks import check, check_transform, check_wrappers",
    ]
    for number, (
        operation,
        shape,
        pattern,
        lengths,
        reduction,
        dtype,
        storage,
        expected,
        strides,
        offset,
    ) in enumerate(cases):
        source.append(f"\n\ndef case_{number}() raises:")
        scalar = {np.float64: "Float64", np.float32: "Float32", np.int32: "Int32"}[
            dtype
        ]
        label = f"case {number}: {operation} {pattern} ({np.dtype(dtype)}; {reduction})"
        # NumPy's min/max signed-zero ties differ across CPU implementations.
        # Compare their numerical values here; native contracts separately
        # require deterministic negative-minimum / positive-maximum tie rules.
        zero_sign = (
            "False" if operation == "reduce" and reduction in {"min", "max"} else "True"
        )
        source.extend(
            [
                f"    var data{number}: List[{scalar}] = {mojo_list(storage, 'Int' if dtype == np.int32 else scalar)}",
                f"    var shape{number}: List[Int] = {mojo_list(shape, 'Int')}",
                f"    var hints{number} = Dict[String, Int]()",
            ]
        )
        source.extend(
            f"    hints{number}[{json.dumps(k)}] = {v}" for k, v in lengths.items()
        )
        extra = (
            ""
            if strides is None
            else f", strides={mojo_list(strides, 'Int')}, offset={offset}"
        )
        source.extend(
            [
                f"    var wanted_shape{number}: List[Int] = {mojo_list(expected.shape, 'Int')}",
                f"    var wanted{number}: List[Float64] = {mojo_list(expected.reshape(-1))}",
                f"    check_transform(Span(data{number}), Span(shape{number}), {json.dumps(pattern)}, .{operation}, hints{number}, Span(wanted_shape{number}), Span(wanted{number}), .{reduction}, {json.dumps(label)}, {zero_sign}{extra})",
            ]
        )
        if strides is None:
            dimensions = ", ".join(str(d) for d in shape)
            parameters = json.dumps(pattern) + (", " + dimensions if dimensions else "")
            axis_spec = " ".join(f"{key}={value}" for key, value in lengths.items())
            parameters += f", operation=.{operation}, axes={json.dumps(axis_spec)}"
            source.extend(
                [
                    f"    var static_recipe = StaticRecipe[{parameters}]()",
                    "    var static_shape = static_recipe.output_shape()",
                    f"    assert_equal(len(static_shape), len(wanted_shape{number}))",
                    f"    for i in range(len(static_shape)): assert_equal(static_shape[i], wanted_shape{number}[i])",
                    f"    var static_result = static_recipe.execute[.{reduction}](Span(data{number}))",
                    f"    check(Span(static_result), Span(wanted{number}), {json.dumps(label + ' static')}, {zero_sign})",
                    f"    var static_into = List[{scalar}](length=static_recipe.output_size, fill=0)",
                    f"    static_recipe.execute_into[.{reduction}](Span(data{number}), Span(static_into))",
                    f"    check(Span(static_into), Span(wanted{number}), {json.dumps(label + ' static into')}, {zero_sign})",
                ]
            )
            if operation != "reduce":
                source.extend(
                    [
                        f"    var static_view = static_recipe.view(Span(data{number}))",
                        "    var static_materialized = static_view.materialize()",
                        f"    check(Span(static_materialized), Span(wanted{number}), {json.dumps(label + ' static view')})",
                    ]
                )
    source.append("\n\ndef check_invalid() raises:")
    for number, (operation, shape, pattern, lengths) in enumerate(invalid):
        source.extend(
            [
                f"    var bad_shape{number}: List[Int] = {mojo_list(shape, 'Int')}",
                f"    var bad_hints{number} = Dict[String, Int]()",
            ]
        )
        source.extend(
            f"    bad_hints{number}[{json.dumps(k)}] = {v}" for k, v in lengths.items()
        )
        source.extend(
            [
                f"    var rejected{number} = False",
                "    try:",
                f"        var bad = plan({json.dumps(pattern)}, Span(bad_shape{number}), .{operation}, bad_hints{number})",
                "    except:",
                f"        rejected{number} = True",
                f"    assert_true(rejected{number}, {json.dumps('invalid ' + pattern)})",
            ]
        )
    source.append("    check_wrappers()")
    source.append(
        f'    print("PASS: {len(cases)} dynamic and {sum(case[8] is None for case in cases)} static independent einops cases, {len(invalid)} invalid patterns, buffer validation, public wrappers")'
    )
    source.append("\n\ndef main() raises:")
    source.extend(f"    case_{number}()" for number in range(len(cases)))
    source.append("    check_invalid()")
    fixture = FIXTURES / "parity.mojo"
    fixture.write_text("\n".join(source) + "\n")
    executable = BUILD / "parity"
    print(
        f"Oracle: einops {einops.__version__}, NumPy {np.__version__}; {len(cases)} cases",
        flush=True,
    )
    subprocess.run(
        [
            str(MOJO),
            "build",
            "--Werror",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "tests"),
            str(fixture),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
    subprocess.run(
        [
            str(MOJO),
            "run",
            "--Werror",
            "-I",
            str(ROOT / "src"),
            str(ROOT / "tests/contracts.mojo"),
        ],
        check=True,
    )
    rejected = subprocess.run(
        [
            str(MOJO),
            "build",
            "--Werror",
            "-I",
            str(ROOT / "src"),
            str(ROOT / "tests/reject_dangling_view.mojo"),
            "-o",
            str(BUILD / "reject-dangling"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        rejected.returncode == 0
        or "origin_of(local)" not in rejected.stderr
        or "cannot implicitly convert" not in rejected.stderr
    ):
        raise AssertionError(
            "The compiler did not reject the escaped local view for the expected origin mismatch:\n"
            + rejected.stderr
        )
    print(
        "PASS: compiler rejects a view escaping its source buffer lifetime", flush=True
    )
    for name, reason in [
        ("reject_static_pattern.mojo", "Invalid static einops recipe"),
        (
            "reject_static_reduction_view.mojo",
            "A reduction cannot be a static tensor view",
        ),
        ("reject_static_dangling_view.mojo", "origin_of(source)"),
    ]:
        rejection = subprocess.run(
            [
                str(MOJO),
                "build",
                "--Werror",
                "-I",
                str(ROOT / "src"),
                str(ROOT / "tests" / name),
                "-o",
                str(BUILD / "rejected-test"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if rejection.returncode == 0 or reason not in rejection.stderr:
            raise AssertionError(f"Wrong result/reason for {name}:\n{rejection.stderr}")
    print(
        "PASS: invalid static patterns, reduction views and escaped origins are rejected",
        flush=True,
    )

    subprocess.run(
        [
            str(MOJO),
            "precompile",
            "--Werror",
            str(ROOT / "src/axisweave"),
            "-o",
            str(PACKAGES / "axisweave.mojoc"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(MOJO),
            "run",
            "--Werror",
            "-I",
            str(PACKAGES),
            str(ROOT / "examples/patches.mojo"),
        ],
        check=True,
    )
    for name in (
        "static_patches.mojo",
        "research_scenarios.mojo",
        "sensor_tensors.mojo",
    ):
        subprocess.run(
            [
                str(MOJO),
                "run",
                "--Werror",
                "-I",
                str(PACKAGES),
                str(ROOT / "examples" / name),
            ],
            check=True,
        )
    result = {
        "einops": einops.__version__,
        "numpy": np.__version__,
        "reference_cases": len(cases),
        "static_reference_cases": sum(case[8] is None for case in cases),
        "invalid_patterns": len(invalid),
        "native_contracts": "passed",
        "lifetime_rejection": "passed",
        "compiled_package_import": "passed",
        "static_compile_rejections": "passed",
        "source_identity": source_hash("axisweave"),
        "fixture_directory": "isolated from precompiled packages",
    }
    (ROOT / "tests/verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
