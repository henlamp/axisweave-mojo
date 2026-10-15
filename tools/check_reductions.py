"""Independent logical, fixed-width integer and custom row reduction references."""

from __future__ import annotations

import itertools
import json
import warnings

import einops
import numpy as np
from _support import FIXTURES, MOJO, ROOT, reject, run, source_hash, write_report


def literal(values, scalar):
    if scalar in {"Bool", "Scalar[.bool]"}:
        return "[" + ",".join("True" if x else "False" for x in values) + "]"
    if scalar in {"Int32", "Int64", "UInt32", "UInt64"}:
        return "[" + ",".join(str(int(x)) for x in values) + "]"
    return (
        "["
        + ",".join(
            f"({scalar}(0)/{scalar}(0))" if np.isnan(x) else repr(float(x))
            for x in values
        )
        + "]"
    )


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261010)
    source = [
        "from std.collections import List, Span",
        "from std.testing import assert_equal",
        "from axisweave import reduce_custom, reduce_custom_into, execute, execute_into, plan, StaticRecipe",
        "from reduction_references import variance, nanmean, rms, check",
    ]
    cases = 0
    counts = {"bool": 0, "integer": 0, "custom": 0}

    def case(lines):
        nonlocal cases
        source.append(f"\ndef case_{cases}() raises:")
        source.extend("    " + line for line in lines)
        cases += 1

    patterns = [
        ([2, 3], "row col -> row"),
        ([2, 3, 4], "batch row col -> col batch"),
        ([2, 3, 4], "batch row col ->"),
        ([2, 6], "batch (row col) -> col batch"),
        ([2, 0], "row col -> row"),
        ([0, 3], "row col -> col"),
        ([0, 3], "row col -> row"),
        ([], " ->"),
        ([3], "col -> col"),
    ]
    for shape, pattern in patterns:
        hints = {"col": 2} if "(row col)" in pattern else {}
        axes = " ".join(f"{key}={value}" for key, value in hints.items())
        for kind, scalar in itertools.product(
            ("any", "all"), ("Bool", "Scalar[.bool]")
        ):
            values = rng.integers(0, 2, size=shape).astype(np.bool_)
            expected = einops.reduce(values, pattern, kind, **hints)
            lines = [
                f"var shape: List[Int] = {shape}",
                f"var data: List[{scalar}] = {literal(values.reshape(-1), scalar)}",
                f"var wanted: List[{scalar}] = {literal(expected.reshape(-1), scalar)}",
                f"var recipe = plan({json.dumps(pattern)}, Span(shape), .reduce, {json.dumps(hints)})",
                f"assert_equal(execute(recipe,Span(data),.{kind}),wanted)",
                f"var target = List[{scalar}](length=len(wanted),fill=False)",
                f"execute_into(recipe,Span(data),Span(target),.{kind})",
                "assert_equal(target,wanted)",
                f"var static = StaticRecipe[{json.dumps(pattern)}{''.join(', ' + str(dim) for dim in shape)}, operation=.reduce,axes={json.dumps(axes)}]()",
                f"assert_equal(static.execute[.{kind}](Span(data)),wanted)",
                f"static.execute_into[.{kind}](Span(data),Span(target))",
                "assert_equal(target,wanted)",
            ]
            case(lines)
            counts["bool"] += 1
        for dtype, reduction in itertools.product(
            (np.int32, np.int64, np.uint32, np.uint64), ("sum", "prod", "min", "max")
        ):
            scalar = {
                np.int32: "Int32",
                np.int64: "Int64",
                np.uint32: "UInt32",
                np.uint64: "UInt64",
            }[dtype]
            if reduction in {"min", "max"} and 0 in shape:
                continue
            values = rng.integers(0, 6, size=shape).astype(dtype)
            if dtype in (np.int32, np.int64):
                values -= np.asarray(2, dtype=dtype)
            if values.size:
                values.reshape(-1)[0] = np.iinfo(dtype).max
            op = getattr(np, reduction)

            def reference(tensor, axes, op=op, reduction=reduction):
                return (
                    op(tensor, axis=axes, dtype=tensor.dtype)
                    if reduction in {"sum", "prod"}
                    else op(tensor, axis=axes)
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = einops.reduce(values, pattern, reference, **hints)
            case(
                [
                    f"var shape: List[Int] = {shape}",
                    f"var data: List[{scalar}] = {literal(values.reshape(-1), scalar)}",
                    f"var wanted: List[{scalar}] = {literal(expected.reshape(-1), scalar)}",
                    f"var recipe = plan({json.dumps(pattern)},Span(shape),.reduce,{json.dumps(hints)})",
                    f"assert_equal(execute(recipe,Span(data),.{reduction}),wanted)",
                    f"var target = List[{scalar}](length=len(wanted),fill=0)",
                    f"execute_into(recipe,Span(data),Span(target),.{reduction})",
                    "assert_equal(target,wanted)",
                    f"var static = StaticRecipe[{json.dumps(pattern)}{''.join(', ' + str(dim) for dim in shape)}, operation=.reduce,axes={json.dumps(axes)}]()",
                    f"assert_equal(static.execute[.{reduction}](Span(data)),wanted)",
                ]
            )
            counts["integer"] += 1
        for dtype, name in itertools.product(
            (np.float32, np.float64), ("variance", "nanmean", "rms")
        ):
            scalar = "Float32" if dtype == np.float32 else "Float64"
            values = rng.normal(size=shape).astype(dtype)
            if name == "nanmean" and values.size:
                values.reshape(-1)[0] = np.nan

            def reference(tensor, axes, name=name):
                tensor = tensor.astype(np.float64)
                if name == "variance":
                    return np.var(tensor, axis=axes)
                if name == "nanmean":
                    return np.nanmean(tensor, axis=axes)
                return np.sqrt(np.mean(tensor * tensor, axis=axes))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = einops.reduce(values, pattern, reference, **hints)
            case(
                [
                    f"var shape: List[Int] = {shape}",
                    f"var data: List[{scalar}] = {literal(values.reshape(-1), scalar)}",
                    f"var wanted: List[Float64] = {literal(expected.reshape(-1), 'Float64')}",
                    f"var recipe = plan({json.dumps(pattern)},Span(shape),.reduce,{json.dumps(hints)})",
                    f"var actual = reduce_custom(recipe,Span(data),{name})",
                    "check(Span(actual),Span(wanted))",
                    f"var target = List[{scalar}](length=len(wanted),fill=0)",
                    f"reduce_custom_into(recipe,Span(data),Span(target),{name})",
                    "check(Span(target),Span(wanted))",
                ]
            )
            counts["custom"] += 1
    source.append("\ndef main() raises:")
    source.extend(f"    case_{i}()" for i in range(cases))
    source.append(
        f'    print("PASS: {counts["bool"]} logical, {counts["integer"]} fixed-width integer and {counts["custom"]} custom reduction references")'
    )
    fixture = FIXTURES / "reductions.mojo"
    fixture.write_text("\n".join(source) + "\n")
    print(
        run(
            [
                MOJO,
                "run",
                "--Werror",
                "-I",
                ROOT / "src",
                "-I",
                ROOT / "tests",
                fixture,
            ],
            "reductions",
            timeout=300,
        )
    )
    print(
        run(
            [
                MOJO,
                "run",
                "--Werror",
                "-I",
                ROOT / "src",
                ROOT / "tests/reduction_contracts.mojo",
            ],
            "reduction-contracts",
        )
    )
    reject("reject_reduction_values_lifetime.mojo", "cannot implicitly convert")
    reject("reject_static_integer_mean.mojo", "Unsupported static reduction")
    write_report(
        ROOT / "tests/reductions-verification.json",
        {
            "cases": counts,
            "einops": einops.__version__,
            "numpy": np.__version__,
            "source_sha256": source_hash("axisweave"),
            "integer_policy": "same DType; sum/prod wrap modulo the input bit width",
        },
    )


if __name__ == "__main__":
    main()
