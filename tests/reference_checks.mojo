# SPDX-License-Identifier: MIT
from std.collections import Dict, List, Span
from std.testing import assert_true, assert_equal
from axisweave import (
    execute,
    execute_into,
    plan,
    rearrange,
    reduce,
    view,
    Operation,
    Reduction,
)


def check[
    dtype: DType, //
](
    actual: Span[Scalar[dtype], _],
    expected: Span[Float64, _],
    label: String = "reference",
    check_zero_sign: Bool = True,
) raises:
    assert_equal(len(actual), len(expected), label)
    for i in range(len(actual)):
        var value = Float64(actual[i])
        var wanted = expected[i]
        if wanted != wanted:
            assert_true(value != value, label)
        elif wanted > 1e308 or wanted < -1e308:
            assert_equal(value, wanted, label)
        else:
            var tolerance = Float64(
                1e-10 + 1e-10 * (wanted if wanted >= 0 else -wanted)
            )
            comptime if dtype == DType.float32:
                tolerance = 2e-5 + 2e-5 * (wanted if wanted >= 0 else -wanted)
            assert_true(
                value >= wanted - tolerance and value <= wanted + tolerance,
                label,
            )
            if check_zero_sign and value == 0 and wanted == 0:
                assert_equal(Float64(1) / value, Float64(1) / wanted, label)


def check_transform[
    dtype: DType, //
](
    data: Span[Scalar[dtype], _],
    shape: Span[Int, _],
    pattern: String,
    operation: Operation,
    hints: Dict[String, Int],
    wanted_shape: Span[Int, _],
    wanted: Span[Float64, _],
    reduction: Reduction,
    label: String,
    check_zero_sign: Bool = True,
    *,
    strides: List[Int] = List[Int](),
    offset: Int = 0,
) raises:
    var recipe = plan(
        pattern, shape, operation, hints, strides=strides, offset=offset
    )
    assert_equal(List(recipe.output_shape()), List(wanted_shape), label)
    var actual = execute(recipe, data, reduction)
    check(Span(actual), wanted, label, check_zero_sign)
    var target = List[Scalar[dtype]](length=recipe.output_size(), fill=0)
    execute_into(recipe, data, Span(target), reduction)
    check(Span(target), wanted, String(label, " into"), check_zero_sign)
    if operation != .reduce:
        var borrowed = view(recipe, data)
        assert_equal(List(borrowed.shape()), List(wanted_shape))
        var materialized = borrowed.materialize()
        check(Span(materialized), wanted, String(label, " view"))


def check_wrappers() raises:
    var shape: List[Int] = [2, 3]
    var recipe = plan("a b -> b a", Span(shape))
    var short: List[Float64] = [1, 2]
    var output = List[Float64](length=6, fill=99)
    var rejected = False
    try:
        execute_into(recipe, Span(short), Span(output))
    except:
        rejected = True
    assert_true(rejected, "short source")
    assert_equal(output[0], Float64(99), "no write before validation")
    var data: List[Float64] = [0, 1, 2, 3, 4, 5]
    var expected: List[Float64] = [0, 3, 1, 4, 2, 5]
    var reordered = rearrange(Span(data), Span(shape), "a b -> b a")
    check(Span(reordered), Span(expected), "rearrange wrapper")
    var reduced = reduce(Span(data), Span(shape), "a b -> a", .sum)
    var wanted: List[Float64] = [3, 12]
    check(Span(reduced), Span(wanted), "reduce wrapper")
