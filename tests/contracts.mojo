"""Contract checks without importing or running Python."""

from std.collections import Dict, ImmSpan, List, Span
from std.origin import ImmUntrackedOrigin, MutUntrackedOrigin
from std.testing import assert_true, assert_equal
from axisweave import (
    repeat,
    rearrange_view,
    view,
    plan,
    Operation,
    Reduction,
    StaticRecipe,
    execute_into,
    execute,
)


def reject_plan(
    pattern: String,
    shape: List[Int],
    strides: List[Int] = List[Int](),
    offset: Int = 0,
) raises:
    var rejected = False
    try:
        var recipe = plan(pattern, Span(shape), strides=strides, offset=offset)
    except:
        rejected = True
    assert_true(rejected, pattern)


def check_zero_ties[dtype: DType]() raises:
    var values: List[Scalar[dtype]] = [0, -0.0, -0.0, 0, 0, 0, -0.0, -0.0]
    var shape: List[Int] = [4, 2]
    var recipe = plan("a b -> a", Span(shape), .reduce)
    var minima = execute(recipe, Span(values), .min)
    var maxima = execute(recipe, Span(values), .max)
    var static_recipe = StaticRecipe["a b -> a", 4, 2, operation=.reduce]()
    var static_minima = static_recipe.execute[.min](Span(values))
    var static_maxima = static_recipe.execute[.max](Span(values))
    for i in range(4):
        assert_equal(minima[i], Scalar[dtype](0))
        assert_equal(maxima[i], Scalar[dtype](0))
        assert_equal(static_minima[i], Scalar[dtype](0))
        assert_equal(static_maxima[i], Scalar[dtype](0))
        assert_equal(Float64(1) / Float64(minima[i]) < 0, i != 2)
        assert_equal(Float64(1) / Float64(maxima[i]) < 0, i == 3)
        assert_equal(Float64(1) / Float64(static_minima[i]) < 0, i != 2)
        assert_equal(Float64(1) / Float64(static_maxima[i]) < 0, i == 3)


def check_external_aliases() raises:
    var values: List[Float64] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    var original = values.copy()
    var shape: List[Int] = [2, 3]
    var recipe = plan("a b -> b a", Span(shape))
    var static_recipe = StaticRecipe["a b -> b a", 2, 3]()
    # Model an FFI buffer with valid storage but no tracked allocation origin.
    # Keep values alive through the final comparison; every range stays in it.
    for offset in [0, 1, 5]:
        for reversed in [False, True]:
            var source_offset = offset if reversed else 0
            var target_offset = 0 if reversed else offset
            var source = ImmSpan[Float64, ImmUntrackedOrigin](
                unsafe_ptr=values.unsafe_ptr()
                .unsafe_offset(source_offset)
                .as_imm()
                .unsafe_origin_cast[ImmUntrackedOrigin](),
                length=6,
            )
            var target = Span[Float64, MutUntrackedOrigin](
                unsafe_ptr=values.unsafe_ptr()
                .unsafe_offset(target_offset)
                .unsafe_origin_cast[MutUntrackedOrigin](),
                length=6,
            )
            for static in [False, True]:
                var rejected = False
                try:
                    if static:
                        static_recipe.execute_into(source, target)
                    else:
                        execute_into(recipe, source, target)
                except e:
                    rejected = "overlap" in String(e)
                assert_true(rejected)
                assert_equal(values, original)
    var source = ImmSpan[Float64, ImmUntrackedOrigin](
        unsafe_ptr=values.unsafe_ptr()
        .as_imm()
        .unsafe_origin_cast[ImmUntrackedOrigin](),
        length=6,
    )
    var adjacent = Span[Float64, MutUntrackedOrigin](
        unsafe_ptr=values.unsafe_ptr()
        .unsafe_offset(6)
        .unsafe_origin_cast[MutUntrackedOrigin](),
        length=6,
    )
    execute_into(recipe, source, adjacent)
    var expected: List[Float64] = [0, 3, 1, 4, 2, 5]
    for i in range(6):
        assert_equal(values[i], original[i])
        assert_equal(values[6 + i], expected[i])


def check_allocation_order() raises:
    var shape: List[Int] = [1]
    var source: List[Float64] = [1]
    var recipe = plan("a -> a", Span(shape))
    recipe._data.output_size = -1
    var rejected = False
    try:
        var result = execute(recipe, Span(source))
    except e:
        rejected = "extents disagree" in String(e)
    assert_true(rejected, "bad metadata must fail before allocating")
    var hints = Dict[String, Int]()
    hints["copies"] = 1000000
    var repeated = plan("a -> a copies", Span(shape), .repeat, hints)
    var empty = List[Float64]()
    rejected = False
    try:
        var result = execute(repeated, Span(empty))
    except e:
        rejected = "Source buffer" in String(e)
    assert_true(rejected)
    var static_repeat = StaticRecipe[
        "a -> a copies", 1, operation=.repeat, axes="copies=1000000"
    ]()
    rejected = False
    try:
        var result = static_repeat.execute(Span(empty))
    except e:
        rejected = "Source buffer" in String(e)
    assert_true(rejected)


def check_empty_static_reduction[reduction: Reduction]() raises:
    var recipe = StaticRecipe[
        "a b -> a", 9223372036854775807, 0, operation=.reduce
    ]()
    var empty = List[Float64]()
    var rejected = False
    try:
        var result = recipe.execute[reduction](Span(empty))
    except e:
        rejected = "empty reduction" in String(e)
    assert_true(rejected, "invalid reduction must fail before allocating")


def main() raises:
    check_external_aliases()
    check_allocation_order()
    check_empty_static_reduction[.min]()
    check_empty_static_reduction[.max]()
    check_zero_ties[.float32]()
    check_zero_ties[.float64]()
    reject_plan("a -> a", [-1])
    reject_plan("a b -> a b", [9223372036854775807, 2])
    reject_plan("a b -> b a", [2, 3], [-3, 1])
    reject_plan("a b -> b a", [2, 3], [1])
    reject_plan("a -> a", [2], [9223372036854775807])
    reject_plan("a -> a", [2], offset=-1)
    reject_plan("a -> a", [2], offset=9223372036854775807)
    reject_plan("a -> (a 2)", [2])
    var shape: List[Int] = [2, 3]
    var source: List[Float64] = [0, 1, 2, 3, 4, 5]
    var recipe = plan("a b -> b a", Span(shape))
    var short = List[Float64](length=5, fill=99)
    var rejected = False
    try:
        execute_into(recipe, Span(source), Span(short))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(short[0], Float64(99))
    var borrowed = rearrange_view(Span(source), Span(shape), "a b -> b a")
    assert_equal(len(borrowed), 6)
    assert_equal(borrowed[1], Float64(3))
    for index in [-1, 6]:
        rejected = False
        try:
            var value = borrowed[index]
        except:
            rejected = True
        assert_true(rejected)
    var hints = Dict[String, Int]()
    hints["k"] = 2
    var repeated = repeat(Span(source), Span(shape), "a b -> a b k", hints)
    assert_equal(len(repeated), 12)
    assert_equal(repeated[0], Float64(0))
    assert_equal(repeated[1], Float64(0))
    assert_equal(repeated[10], Float64(5))
    assert_equal(repeated[11], Float64(5))
    var reduced = plan("a b -> a", Span(shape), .reduce)
    rejected = False
    try:
        var bad_view = view(reduced, Span(source))
    except:
        rejected = True
    assert_true(rejected)
    var malformed = recipe.copy()
    malformed._data.output_axes[0] = 999
    rejected = False
    try:
        var invalid_view = view(malformed, Span(source))
    except:
        rejected = True
    assert_true(rejected, "malformed recipes rejected by view construction")
    var output = List[Float64](length=2, fill=99)
    rejected = False
    try:
        execute_into(reduced, Span(source), Span(output), Reduction(99))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(output[0], Float64(99))
    var empty_shape: List[Int] = [2, 0]
    var empty = List[Float64]()
    var empty_recipe = plan("a b -> a", Span(empty_shape), .reduce)
    for reduction in [Reduction.min, Reduction.max]:
        rejected = False
        try:
            execute_into(empty_recipe, Span(empty), Span(output), reduction)
        except:
            rejected = True
        assert_true(rejected)
    var bad_metadata = recipe.copy()
    bad_metadata._data.source_strides[0] = 1000
    var destination = List[Float64](length=6, fill=99)
    rejected = False
    try:
        execute_into(bad_metadata, Span(source), Span(destination))
    except:
        rejected = True
    assert_true(rejected, "modified strides rejected before unchecked access")
    assert_equal(destination[0], Float64(99))
    bad_metadata = recipe.copy()
    bad_metadata._data.output_axes[1] = bad_metadata._data.output_axes[0]
    rejected = False
    try:
        execute_into(bad_metadata, Span(source), Span(destination))
    except:
        rejected = True
    assert_true(rejected, "duplicate axes rejected")
    var tail_data: List[Float64] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    var tail_shape: List[Int] = [11]
    var tail_recipe = plan("a -> a", Span(tail_shape), .rearrange)
    var padded = List[Float64](length=13, fill=-999)
    var target = Span(padded)[1:12]
    execute_into(tail_recipe, Span(tail_data), target)
    assert_equal(padded[0], Float64(-999))
    assert_equal(padded[12], Float64(-999))
    for i in range(11):
        assert_equal(padded[i + 1], tail_data[i])
    var static_tail = StaticRecipe["a -> a", 11]()
    static_tail.execute_into(Span(tail_data), target)
    assert_equal(padded[0], Float64(-999))
    assert_equal(padded[12], Float64(-999))
    var typed_sum = plan("a b -> a", Span(shape), .reduce)
    execute_into(typed_sum, Span(source), Span(output), .sum)
    assert_equal(output[0], Float64(3))
    assert_equal(output[1], Float64(12))
    print("PASS: native boundary, storage, reduction and wrapper contracts")
