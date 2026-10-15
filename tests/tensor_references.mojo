# SPDX-License-Identifier: MIT
"""Native checks driven by independently computed einops fixtures."""
from std.collections import Array, ImmSpan, List, Span
from std.testing import assert_equal
from axisweave import (
    packing_plan,
    unpacking_plan,
    pack,
    unpack,
    einsum_plan,
    einsum,
)
from reference_checks import check


def check_pack[
    origin: Origin[mut=False], //, dtype: DType, count: Int
](
    shapes: List[List[Int]],
    inputs: List[List[Scalar[dtype]]],
    pattern: String,
    wanted: List[Float64],
    wanted_shape: List[Int],
    wanted_ps: List[List[Int]],
    expected: List[List[Float64]],
    operands: ImmSpan[ImmSpan[Scalar[dtype], origin], _],
) raises:
    var recipe = packing_plan(Span(shapes), pattern)
    var packed = recipe.pack(Span(inputs))
    check(Span(packed), Span(wanted))
    assert_equal(List(recipe.output_shape()), wanted_shape)
    assert_equal(recipe.packed_shapes(), wanted_ps)
    var target = List[Scalar[dtype]](length=len(packed), fill=0)
    recipe.pack_into(Span(inputs), Span(target))
    check(Span(target), Span(wanted))
    var restored = unpack(recipe, Span(packed))
    var borrowed = recipe.unpack_views(Span(packed))
    var joint = pack(Span(inputs), Span(shapes), pattern)
    check(Span(joint[0]), Span(wanted))
    var borrowed_pack = recipe.pack(operands)
    check(Span(borrowed_pack), Span(wanted))
    recipe.pack_into(operands, Span(target))
    check(Span(target), Span(wanted))
    var borrowed_joint = pack(operands, Span(shapes), pattern)
    check(Span(borrowed_joint[0]), Span(wanted))
    var split = List[Scalar[dtype]](length=len(wanted), fill=0)
    var outputs = Array[
        Span[mut=True, Scalar[dtype], origin_of(split)], count
    ]()
    var offset = 0
    for i in range(count):
        outputs[i] = Span(split)[offset : offset + len(expected[i])]
        offset += len(expected[i])
    recipe.unpack_into(Span(packed), Span(outputs))
    offset = 0
    for i in range(count):
        check(Span(restored[i]), Span(expected[i]))
        check(
            Span(split).as_imm()[offset : offset + len(expected[i])],
            Span(expected[i]),
        )
        var copied = borrowed[i].materialize()
        check(Span(copied), Span(expected[i]))
        assert_equal(List(recipe.input_shape(i)), shapes[i])
        offset += len(expected[i])
    var hints = wanted_ps.copy()
    var candidate = -1
    for i in range(count):
        if len(hints[i]) > 0:
            candidate = i
            break
    if candidate >= 0:
        hints[candidate][0] = -1
        var recovered = unpacking_plan(Span(wanted_shape), Span(hints), pattern)
        var recovered_values = recovered.unpack(Span(packed))
        assert_equal(recovered.packed_shapes(), wanted_ps)
        for i in range(count):
            check(Span(recovered_values[i]), Span(expected[i]))


def check_contraction[
    origin: Origin[mut=False], //, dtype: DType, count: Int
](
    shapes: List[List[Int]],
    inputs: List[List[Scalar[dtype]]],
    pattern: String,
    wanted: List[Float64],
    shape: List[Int],
    operands: ImmSpan[ImmSpan[Scalar[dtype], origin], _],
) raises:
    var recipe = einsum_plan(Span(shapes), pattern)
    var actual = recipe.execute(Span(inputs))
    check(Span(actual), Span(wanted))
    assert_equal(List(recipe.output_shape()), shape)
    var target = List[Scalar[dtype]](length=len(wanted), fill=0)
    recipe.execute_into(Span(inputs), Span(target))
    check(Span(target), Span(wanted))
    var direct = einsum(Span(inputs), Span(shapes), pattern)
    check(Span(direct), Span(wanted))
    var borrowed_result = recipe.execute(operands)
    check(Span(borrowed_result), Span(wanted))
    recipe.execute_into(operands, Span(target))
    check(Span(target), Span(wanted))
    var borrowed_direct = einsum(operands, Span(shapes), pattern)
    check(Span(borrowed_direct), Span(wanted))
