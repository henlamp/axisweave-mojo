"""Callback errors, origins, strided rows, empty identities and invalid combinations."""
from std.collections import List, Span
from std.origin import MutUntrackedOrigin
from std.testing import assert_equal, assert_true
from axisweave import (
    ReductionValues,
    plan,
    execute,
    execute_into,
    reduce,
    reduce_custom,
    reduce_custom_into,
    StaticRecipe,
)


def ordered[
    d: DType, s: Origin[mut=False], p: Origin[mut=False]
](values: ReductionValues[d, s, p]) raises -> Scalar[d]:
    var total = Scalar[d](0)
    for i in range(len(values)):
        total += Scalar[d](i + 1) * values[i]
    return total


def invalid_index[
    d: DType, s: Origin[mut=False], p: Origin[mut=False]
](values: ReductionValues[d, s, p]) raises -> Scalar[d]:
    return values[len(values)]


def main() raises:
    var shape: List[Int] = [2, 3]
    var source: List[Float64] = [77, 1, 2, 3, 77, 77, 4, 5, 6]
    var strides: List[Int] = [5, 1]
    var recipe = plan(
        "row col -> row", Span(shape), .reduce, strides=strides, offset=1
    )
    var wanted: List[Float64] = [14, 32]
    assert_equal(reduce_custom(recipe, Span(source), ordered), wanted)
    var dense: List[Float64] = [1, 2, 3, 4, 5, 6]
    assert_equal(
        reduce(Span(dense), Span(shape), "row col -> row", reducer=ordered),
        wanted,
    )
    var row = ReductionValues(Span(source), recipe, 1)
    assert_equal(len(row), 3)
    assert_equal(row[0], Float64(4))
    assert_equal(row[2], Float64(6))
    var bad_row = False
    try:
        var invalid = ReductionValues(Span(source), recipe, 2)
    except:
        bad_row = True
    assert_true(bad_row)
    var calls = 0

    def counted[
        d: DType, s: Origin[mut=False], p: Origin[mut=False]
    ](values: ReductionValues[d, s, p]) raises {mut calls} -> Scalar[d]:
        calls += 1
        return ordered(values)

    var output = List[Float64](length=2, fill=-7)
    reduce_custom_into(recipe, Span(source), Span(output), counted)
    assert_equal(output, wanted)
    assert_equal(calls, 2)
    var wrong = List[Float64](length=1, fill=-7)
    var rejected = False
    try:
        reduce_custom_into(recipe, Span(source), Span(wrong), counted)
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(calls, 2)
    assert_equal(wrong[0], Float64(-7))
    var alias = Span[Float64, MutUntrackedOrigin](
        unsafe_ptr=source.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        length=2,
    )
    rejected = False
    try:
        reduce_custom_into(recipe, Span(source), alias, counted)
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(calls, 2)
    var rows_seen = 0

    def fail_later[
        d: DType, s: Origin[mut=False], p: Origin[mut=False]
    ](values: ReductionValues[d, s, p]) raises {mut rows_seen} -> Scalar[d]:
        rows_seen += 1
        if rows_seen == 2:
            raise Error("callback failure")
        return ordered(values)

    output = List[Float64](length=2, fill=-7)
    rejected = False
    try:
        reduce_custom_into(recipe, Span(source), Span(output), fail_later)
    except e:
        rejected = String(e) == "callback failure"
    assert_true(rejected)
    assert_equal(output[0], Float64(14))
    assert_equal(output[1], Float64(-7))
    rejected = False
    try:
        _ = reduce_custom(recipe, Span(source), invalid_index)
    except e:
        rejected = "outside the row" in String(e)
    assert_true(rejected)
    var empty: List[Bool] = []
    var empty_shape: List[Int] = [2, 0]
    var empty_plan = plan("row col -> row", Span(empty_shape), .reduce)
    var false_rows: List[Bool] = [False, False]
    var true_rows: List[Bool] = [True, True]
    assert_equal(execute(empty_plan, Span(empty), .any), false_rows)
    assert_equal(execute(empty_plan, Span(empty), .all), true_rows)
    var logical_output: List[Bool] = [True, True]
    rejected = False
    try:
        execute_into(empty_plan, Span(empty), Span(logical_output), .sum)
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(logical_output, true_rows)
    var integer: List[Int64] = [1, 2, 3, 4, 5, 6]
    var integer_output: List[Int64] = [17, 17]
    var integer_plan = plan("row col -> row", Span(shape), .reduce)
    rejected = False
    try:
        execute_into(integer_plan, Span(integer), Span(integer_output), .mean)
    except:
        rejected = True
    assert_true(rejected)
    var untouched: List[Int64] = [17, 17]
    assert_equal(integer_output, untouched)
    var flags: List[Bool] = [True, False, True, False, True, False]
    var transpose = plan("row col -> col row", Span(shape))
    var flags_wanted: List[Bool] = [True, False, False, True, True, False]
    assert_equal(execute(transpose, Span(flags)), flags_wanted)
    var static = StaticRecipe["row col -> col row", 2, 3]()
    assert_equal(static.execute(Span(flags)), flags_wanted)
    var identity = plan("row col -> col row", Span(shape), .reduce)
    _ = reduce_custom(identity, Span(dense), counted)
    assert_equal(calls, 2)
    var zero_output_shape: List[Int] = [0, 3]
    var zero_output = plan("row col -> row", Span(zero_output_shape), .reduce)
    var no_data: List[Float64] = []
    _ = reduce_custom(zero_output, Span(no_data), counted)
    assert_equal(calls, 2)
    print(
        "PASS: strided/captured callbacks, validation before invocation,"
        " partial callback failures, empty truth identities, native Bool copies"
        " and unsupported integer means"
    )
