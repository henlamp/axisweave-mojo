"""Exact einops references for element-preserving buffers and borrowed views."""

from __future__ import annotations

import itertools
import json

import einops
import numpy as np
from _support import FIXTURES, MOJO, ROOT, reject, run, source_hash, write_report
from check_reductions import literal


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261010)
    source = [
        """from std.collections import Array, ImmSpan, List, Span
from std.testing import assert_equal
from axisweave import packing_plan, unpacking_plan, pack, unpack, plan, view, rearrange_view, repeat_view, StaticRecipe
"""
    ]
    count = 0
    packing = 0
    views = 0

    def case(lines):
        nonlocal count
        source.append(f"\ndef case_{count}() raises:")
        source.extend("    " + line for line in lines)
        count += 1

    def values(shape, scalar):
        result = rng.integers(0, 2, size=shape).astype(
            np.uint64 if scalar == "UInt64" else np.bool_
        )
        if scalar == "UInt64" and result.size:
            result.reshape(-1)[0] = np.iinfo(np.uint64).max
        return result

    for scalar, before, after, trial in itertools.product(
        ("Bool", "Scalar[.bool]", "UInt64"), range(3), range(3), range(3)
    ):
        shapes = [[2] * before + mid + [3] * after for mid in ([], [trial], [2, trial])]
        tensors = [values(shape, scalar) for shape in shapes]
        pattern = " ".join(
            [*(f"p{i}" for i in range(before)), "*", *(f"s{i}" for i in range(after))]
        )
        packed, dimensions = einops.pack(tensors, pattern)
        restored = einops.unpack(packed, dimensions, pattern)
        lines = [
            f"var shapes: List[List[Int]] = {shapes}",
            f"var inputs: List[List[{scalar}]] = ["
            + ",".join(literal(t.reshape(-1), scalar) for t in tensors)
            + "]",
            f"var wanted: List[{scalar}] = {literal(packed.reshape(-1), scalar)}",
            f"var recipe = packing_plan(Span(shapes), {json.dumps(pattern)})",
            "assert_equal(recipe.pack(Span(inputs)), wanted)",
            f"var target = List[{scalar}](length=len(wanted), fill={scalar}())",
            "recipe.pack_into(Span(inputs), Span(target))",
            "assert_equal(target, wanted)",
            f"var operands: Array[ImmSpan[{scalar}, origin_of(inputs[0])], 3] = ["
            + ",".join(
                f"ImmSpan[{scalar}, origin_of(inputs[0])](inputs[{i}])"
                for i in range(3)
            )
            + "]",
            "assert_equal(recipe.pack(Span(operands)), wanted)",
            "recipe.pack_into(Span(operands), Span(target))",
            "assert_equal(target, wanted)",
            f"var owned_joint = pack(ImmSpan(inputs), Span(shapes), {json.dumps(pattern)})",
            "assert_equal(owned_joint[0], wanted)",
            f"var borrowed_joint = pack(Span(operands), Span(shapes), {json.dumps(pattern)})",
            "assert_equal(borrowed_joint[0], wanted)",
            "assert_equal(unpack(recipe, Span(target)), inputs)",
            "var borrowed = recipe.unpack_views(Span(target))",
            f"var hints: List[List[Int]] = {json.dumps(dimensions)}",
            "hints[1][0] = -1",
            f"var packed_shape: List[Int] = {list(packed.shape)}",
            f"var inferred = unpacking_plan(Span(packed_shape), Span(hints), {json.dumps(pattern)})",
            "assert_equal(inferred.unpack(Span(target)), inputs)",
            f"var split = List[{scalar}](length=len(wanted), fill={scalar}())",
            f"var destinations = Array[Span[mut=True, {scalar}, origin_of(split)], 3]()",
        ]
        offset = 0
        for i, t in enumerate(restored):
            lines.extend(
                [
                    f"destinations[{i}] = Span(split)[{offset}:{offset + t.size}]",
                    f"assert_equal(borrowed[{i}].materialize(), inputs[{i}])",
                    f"assert_equal(List(borrowed[{i}].shape()), shapes[{i}])",
                    f"var direct{i} = List[{scalar}](length={t.size}, fill={scalar}())",
                    f"borrowed[{i}].materialize_into(Span(direct{i}))",
                    f"assert_equal(direct{i}, inputs[{i}])",
                ]
            )
            offset += t.size
        lines.append("recipe.unpack_into(Span(target), Span(destinations))")
        offset = 0
        for i, t in enumerate(restored):
            lines.append(
                f"assert_equal(List(Span(split)[{offset}:{offset + t.size}].as_imm()), inputs[{i}])"
            )
            offset += t.size
        case(lines)
        packing += 1

    patterns = [
        ([2, 3], "row col -> col row", "rearrange", {}),
        ([2, 6], "row (a b) -> b row a", "rearrange", {"b": 2}),
        ([], "->", "rearrange", {}),
        ([0, 3], "row col -> col row", "rearrange", {}),
        ([2], "row -> copies row", "repeat", {"copies": 3}),
        ([2], "row -> row copies", "repeat", {"copies": 0}),
    ]
    for scalar, (shape, pattern, operation, hints) in itertools.product(
        ("Bool", "Scalar[.bool]", "UInt64"), patterns
    ):
        tensor = values(shape, scalar)
        wanted = getattr(einops, operation)(tensor, pattern, **hints)
        axes = " ".join(f"{k}={v}" for k, v in hints.items())
        case(
            [
                f"var shape: List[Int] = {shape}",
                f"var data: List[{scalar}] = {literal(tensor.reshape(-1), scalar)}",
                f"var wanted: List[{scalar}] = {literal(wanted.reshape(-1), scalar)}",
                f"var recipe = plan({json.dumps(pattern)}, Span(shape), .{operation}, {hints})",
                "var prepared = view(recipe, Span(data))",
                f"var convenient = {operation}_view(Span(data), Span(shape), {json.dumps(pattern)}, {hints})",
                "assert_equal(prepared.materialize(), wanted)",
                "assert_equal(convenient.materialize(), wanted)",
                f"var static = StaticRecipe[{json.dumps(pattern)}{''.join(', ' + str(d) for d in shape)}, operation=.{operation}, axes={json.dumps(axes)}]()",
                "var borrowed = static.view(Span(data))",
                "assert_equal(borrowed.materialize(), wanted)",
                f"var target = List[{scalar}](length=len(wanted), fill={scalar}())",
                "prepared.materialize_into(Span(target))",
                "assert_equal(target, wanted)",
                "borrowed.materialize_into(Span(target))",
                "assert_equal(target, wanted)",
                f"var output_shape: List[Int] = {list(wanted.shape)}",
                "assert_equal(List(prepared.shape()), output_shape)",
                "assert_equal(List(borrowed.shape()), output_shape)",
                "for i in range(len(wanted)):",
                "    assert_equal(prepared[i], wanted[i])",
                "    assert_equal(borrowed[i], wanted[i])",
            ]
        )
        views += 1
    source.append("\ndef main() raises:")
    source.extend(f"    case_{i}()" for i in range(count))
    source.append(
        f'    print("PASS: {packing} exact buffer and {views} view references")'
    )
    fixture = FIXTURES / "element_buffers.mojo"
    fixture.write_text("\n".join(source))
    print(
        run(
            [MOJO, "run", "--Werror", "-I", ROOT / "src", fixture],
            "element-buffers",
            timeout=600,
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
                ROOT / "tests/buffer_contracts.mojo",
            ],
            "buffer-contracts",
        )
    )
    reject("reject_bool_view_lifetime.mojo", "cannot implicitly convert")
    reject("reject_bool_unpack_lifetime.mojo", "cannot implicitly convert")
    reject("reject_static_bool_view_lifetime.mojo", "cannot implicitly convert")
    write_report(
        ROOT / "tests/element-buffers-verification.json",
        {
            "packing_cases": packing,
            "view_cases": views,
            "element_types": ["Bool", "Scalar[.bool]", "UInt64"],
            "einops": einops.__version__,
            "numpy": np.__version__,
            "source_sha256": source_hash("axisweave"),
        },
    )


if __name__ == "__main__":
    main()
