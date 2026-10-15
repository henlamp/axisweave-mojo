"""Independent pack/unpack, named-contraction and shape-parser references."""

from __future__ import annotations

import itertools
import json
import statistics

import einops
import numpy as np
from _support import FIXTURES, MOJO, ROOT, reject, run, source_hash, write_report
from check import mojo_list


def borrowed(scalar, count):
    return (
        f"var operands: Array[ImmSpan[{scalar}, origin_of(inputs[0])], {count}] = ["
        + ",".join(
            f"ImmSpan[{scalar}, origin_of(inputs[0])](inputs[{i}])"
            for i in range(count)
        )
        + "]"
    )


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261009)
    source = [
        "from std.collections import Array, ImmSpan, List, Span",
        "from std.testing import assert_equal",
        "from axisweave import parse_shape, mix_plan",
        "from reference_checks import check",
        "from tensor_references import check_pack, check_contraction",
    ]
    count = 0
    packing = 0
    inferred = 0
    contractions = 0
    parsed = 0
    mixing = 0

    def function(lines):
        nonlocal count
        source.append(f"\ndef case_{count}() raises:")
        source.extend("    " + line for line in lines)
        count += 1

    for dtype, before, after, trial in itertools.product(
        (np.float32, np.float64), range(3), range(3), range(3)
    ):
        prefix = [2] * before
        suffix = [3] * after
        middles = [[], [trial], [2, trial + 1]]
        shapes = [prefix + middle + suffix for middle in middles]
        tensors = [rng.normal(size=shape).astype(dtype) for shape in shapes]
        pattern = " ".join(
            [*(f"p{i}" for i in range(before)), "*", *(f"s{i}" for i in range(after))]
        )
        values, ps = einops.pack(tensors, pattern)
        reference = einops.unpack(values, ps, pattern)
        scalar = "Float32" if dtype == np.float32 else "Float64"
        function(
            [
                f"var shapes: List[List[Int]] = {json.dumps(shapes)}",
                f"var inputs: List[List[{scalar}]] = ["
                + ",".join(mojo_list(array.reshape(-1), scalar) for array in tensors)
                + "]",
                f"var wanted: List[Float64] = {mojo_list(values.reshape(-1))}",
                f"var wanted_shape: List[Int] = {list(values.shape)}",
                f"var wanted_ps: List[List[Int]] = {json.dumps([list(dims) for dims in ps])}",
                "var expected: List[List[Float64]] = ["
                + ",".join(mojo_list(array.reshape(-1)) for array in reference)
                + "]",
                borrowed(scalar, len(tensors)),
                f"check_pack[{'.float32' if dtype == np.float32 else '.float64'}, {len(tensors)}](shapes, inputs, {json.dumps(pattern)}, wanted, wanted_shape, wanted_ps, expected, Span(operands))",
            ]
        )
        candidate = next((i for i, dims in enumerate(ps) if len(dims) > 0), None)
        if candidate is not None:
            hints = [list(dims) for dims in ps]
            hints[candidate][0] = -1
            expected_inferred = einops.unpack(values, hints, pattern)
            assert all(
                np.array_equal(actual, expected)
                for actual, expected in zip(expected_inferred, tensors, strict=True)
            )
            inferred += 1
        packing += 1

    definitions = [
        ([[2, 3], [3, 5]], "row feature, feature column -> row column"),
        ([[2, 3, 4], [4, 5]], "batch row feature, feature column -> batch row column"),
        ([[3, 4], [4, 2], [2, 5]], "a b, b c, c d -> a d"),
        ([[3, 3]], "diagonal diagonal -> diagonal"),
        ([[3, 3]], "diagonal diagonal ->"),
        ([[0, 0]], "diagonal diagonal ->"),
        ([[4]], "feature ->"),
        ([[3], [5]], "row, column -> row column"),
        ([[], [2, 5]], ", row column -> row column"),
        ([[1, 3], [2, 3]], "batch feature, batch feature -> batch"),
        ([[2, 0], [0, 3]], "row feature, feature column -> row column"),
        ([[0, 3], [3, 4]], "row feature, feature column -> row column"),
        ([[2, 3], [2, 3]], "row feature, row feature -> feature row"),
        ([[3, 4]], "row column -> column row"),
        (
            [[2, 3, 4], [1, 4, 5]],
            "... row feature, ... feature column -> ... row column",
        ),
        ([[2, 3, 4], [4, 5]], "... row feature, feature column -> ... row column"),
        (
            [[2, 1, 3, 4], [5, 4, 2]],
            "... row feature, ... feature column -> ... row column",
        ),
        ([[2, 3, 3]], "... diagonal diagonal -> ... diagonal"),
        ([[2, 3, 4, 5], [4, 5, 7]], "a b k l, k l c -> a b c"),
        ([[3, 2, 1, 4], [2, 1, 4, 5]], "a k l m, k l m b -> b a"),
        ([[2, 3, 5], [3, 5], [2, 3]], "a k l, k l, a k -> a"),
        ([[3, 3, 2, 2]], "a a b b ->"),
        ([[2, 0, 3], [0, 3, 5]], "a k l, k l b -> a b"),
        ([[2, 1, 5], [1, 3, 5]], "a k l, k b l -> a b"),
        ([[2, 3, 5]], "a b c -> c a b"),
        ([[2, 3], [3]], "a k, k ->"),
    ]
    for dtype, (shapes, pattern) in itertools.product(
        (np.float32, np.float64), definitions
    ):
        tensors = [rng.normal(size=shape).astype(dtype) for shape in shapes]
        expected = einops.einsum(*tensors, pattern)
        scalar = "Float32" if dtype == np.float32 else "Float64"
        function(
            [
                f"var shapes: List[List[Int]] = {json.dumps(shapes)}",
                f"var inputs: List[List[{scalar}]] = ["
                + ",".join(mojo_list(array.reshape(-1), scalar) for array in tensors)
                + "]",
                f"var wanted: List[Float64] = {mojo_list(np.asarray(expected).reshape(-1))}",
                f"var shape: List[Int] = {list(np.asarray(expected).shape)}",
                borrowed(scalar, len(tensors)),
                f"check_contraction[{'.float32' if dtype == np.float32 else '.float64'}, {len(tensors)}](shapes, inputs, {json.dumps(pattern)}, wanted, shape, Span(operands))",
            ]
        )
        contractions += 1
        if len(tensors) == 2:
            for with_bias in (False, True):
                named = pattern.split("->")[1].split()
                bias_axes = named[-1] if named and named[-1] != "..." else ""
                bias_shape = [np.asarray(expected).shape[-1]] if bias_axes else []
                bias = rng.normal(size=bias_shape).astype(dtype)
                # Broadcast the selected final named axis through NumPy itself.
                wanted = (
                    np.asarray(expected) + bias if with_bias else np.asarray(expected)
                )
                lines = [
                    f"var input_shape: List[Int] = {shapes[0]}",
                    f"var weight_shape: List[Int] = {shapes[1]}",
                    f"var input: List[{scalar}] = {mojo_list(tensors[0].reshape(-1), scalar)}",
                    f"var weights: List[{scalar}] = {mojo_list(tensors[1].reshape(-1), scalar)}",
                ]
                call = f"mix_plan(Span(input_shape), Span(weight_shape), {json.dumps(pattern)}"
                arguments = "Span(input), Span(weights)"
                if with_bias:
                    lines += [
                        f"var bias_shape: List[Int] = {bias_shape}",
                        f"var bias: List[{scalar}] = {mojo_list(bias.reshape(-1), scalar)}",
                    ]
                    call += f", bias_shape=Span(bias_shape), bias_axes={json.dumps(bias_axes)}"
                    arguments += ", bias=Span(bias)"
                lines += [
                    f"var recipe = {call})",
                    f"var actual = recipe.execute({arguments})",
                    f"var wanted: List[Float64] = {mojo_list(wanted.reshape(-1))}",
                    "check(Span(actual), Span(wanted))",
                    f"var shape: List[Int] = {list(wanted.shape)}",
                    "assert_equal(List(recipe.output_shape()), shape)",
                    f"var target = List[{scalar}](length=len(wanted), fill=0)",
                    "recipe.execute_into(Span(input), Span(weights), Span(target)"
                    + (", bias=Span(bias)" if with_bias else "")
                    + ")",
                    "check(Span(target), Span(wanted))",
                ]
                function(lines)
                mixing += 1

    for shape, pattern in [
        ([2, 3, 5, 7], "batch _ height width"),
        ([2, 3, 5, 7], "batch ... width"),
        ([3], "... width"),
        ([], "..."),
        ([], ""),
        ([1, 2, 3], "1 height width"),
        ([2, 3, 4], "batch 3 feature"),
        ([0, 3], "batch feature"),
        ([2, 1], "batch ... 1"),
    ]:
        expected = einops.parse_shape(np.zeros(shape), pattern)
        lines = [
            f"var shape: List[Int] = {shape}",
            f"var actual = parse_shape(Span(shape), {json.dumps(pattern)})",
            f"assert_equal(len(actual), {len(expected)})",
        ]
        lines.extend(
            f"assert_equal(actual[{json.dumps(key)}], {value})"
            for key, value in expected.items()
        )
        function(lines)
        parsed += 1
    source.append("\ndef main() raises:")
    source.extend(f"    case_{i}()" for i in range(count))
    source.append(
        f'    print("PASS: {packing} packing, {inferred} inferred unpacking, {contractions} contraction, {mixing} mixing and {parsed} shape references")'
    )
    fixture = FIXTURES / "tensor_features.mojo"
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
            "tensor-features",
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
                ROOT / "tests/tensor_features_contracts.mojo",
            ],
            "tensor-features-contracts",
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
                ROOT / "tests/zero_extents.mojo",
            ],
            "zero-extents",
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
                ROOT / "tests/source_buffers.mojo",
            ],
            "source-buffers",
        )
    )
    reject("reject_operand_mutation.mojo", "must be mutable")
    reject("reject_operand_lifetime.mojo", "cannot implicitly convert")
    reject("reject_unpack_output_lifetime.mojo", "cannot implicitly convert")
    reject("reject_view_shape_lifetime.mojo", "cannot implicitly convert")
    reject("reject_pack_shape_lifetime.mojo", "cannot implicitly convert")
    reject("reject_borrowed_shape_mutation.mojo", "must be mutable")
    reject("reject_unpack_lifetime.mojo", "cannot implicitly convert")
    reject("reject_recipe_shape_lifetime.mojo", "cannot implicitly convert")
    reject("reject_recipe_shape_mutation.mojo", "must be mutable")
    benchmark = json.loads(
        run(
            [
                MOJO,
                "run",
                "--Werror",
                "-I",
                ROOT / "src",
                ROOT / "tests/buffer_benchmarks.mojo",
            ],
            "buffer-benchmarks",
        )
    )
    expected_checksum = 0.0
    for _ in range(5):
        for iteration in range(100):
            first = np.float32(0.125) + np.float32(iteration % 7) * np.float32(0.001)
            row = np.full(56, np.float32(0.125))
            row[0] = first
            value = np.float32(0)
            for item in row:
                value += item * np.float32(0.25)
            expected_checksum += float(value + first)
    for key in ("owned_checksum", "borrowed_checksum"):
        np.testing.assert_allclose(benchmark[key], expected_checksum, rtol=1e-12)
    for key in ("owned_seconds", "borrowed_seconds"):
        if len(benchmark[key]) != 5 or any(
            not np.isfinite(t) or t <= 0 for t in benchmark[key]
        ):
            raise AssertionError("Invalid native benchmark trials")
        benchmark[key.replace("seconds", "median")] = statistics.median(benchmark[key])
    benchmark["source_sha256"] = source_hash("axisweave")
    benchmark["scope"] = (
        "Five alternating trials of 100 prepared Float32 contractions (32x56 times 56x24) and flat packs, with changing inputs and matching independent NumPy checksums. Both paths reuse caller output buffers. No speed threshold; background load uncontrolled. Adapter sizes describe span metadata, not total operation allocations."
    )
    write_report(ROOT / "tests/buffer-benchmarks.json", benchmark)
    write_report(
        ROOT / "tests/tensor-features-verification.json",
        {
            "packing_cases": packing,
            "inferred_unpacking_cases": inferred,
            "contraction_cases": contractions,
            "shape_cases": parsed,
            "mixing_cases": mixing,
            "borrowed_packing_cases": packing,
            "borrowed_contraction_cases": contractions,
            "direct_unpacking_cases": packing,
            "einops": einops.__version__,
            "numpy": np.__version__,
            "source_sha256": source_hash("axisweave"),
        },
    )


if __name__ == "__main__":
    main()
