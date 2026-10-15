"""Metadata lifetimes, high-rank axes, input rejection and external aliases."""
from std.collections import ImmSpan, List, Span
from std.origin import MutUntrackedOrigin
from std.testing import assert_equal, assert_true
from axisweave import (
    plan,
    execute,
    packing_plan,
    unpacking_plan,
    einsum_plan,
    parse_shape,
)


def main() raises:
    var dims = List[Int](length=66, fill=1)
    var pattern = String()
    for i in range(66):
        pattern += String("a", i, " ")
    pattern += String("-> ", pattern)
    var recipe = plan(pattern, Span(dims))
    var values: List[Float64] = [7]
    assert_equal(execute(recipe, Span(values)), values)
    recipe._data.output_axes[65] = 64
    var rejected = False
    try:
        var result = execute(recipe, Span(values))
    except e:
        rejected = "duplicate" in String(e)
    assert_true(rejected)

    var shapes: List[List[Int]] = [[2, 3], [2, 2, 3]]
    var inputs: List[List[Int64]] = [
        [
            9007199254740993,
            9007199254740995,
            9007199254740997,
            9007199254740999,
            9007199254741001,
            9007199254741003,
        ],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    var packed_recipe = packing_plan(Span(shapes), "b * c")
    var packed = packed_recipe.pack(Span(inputs))
    var restored = packed_recipe.unpack(Span(packed))
    assert_equal(restored, inputs)
    var copied_shapes = packed_recipe.packed_shapes()
    copied_shapes[1][0] = 99
    assert_equal(List(packed_recipe.input_shape(1)), shapes[1])

    var empty_shapes: List[List[Int]] = [[0, 3], [0, 2, 3]]
    var empty_inputs: List[List[Float64]] = [[], []]
    var empty_recipe = packing_plan(Span(empty_shapes), "b * c")
    var zero = empty_recipe.pack(Span(empty_inputs))
    assert_equal(len(zero), 0)
    assert_equal(len(empty_recipe.unpack(Span(zero))), 2)
    var empty_views = empty_recipe.unpack_views(Span(zero))
    for part in range(len(empty_views)):
        assert_equal(len(empty_views[part]), 0)
        assert_equal(List(empty_views[part].shape()), empty_shapes[part])
        assert_equal(len(empty_views[part].materialize()), 0)
    var wide_shapes: List[List[Int]] = [[0, 9223372036854775806, 2], [0, 1, 2]]
    var wide_recipe = packing_plan(Span(wide_shapes), "b * c")
    var wide_views = wide_recipe.unpack_views(Span(zero))
    for part in range(len(wide_views)):
        assert_equal(List(wide_views[part].shape()), wide_shapes[part])
        assert_equal(len(wide_views[part].materialize()), 0)
    var mixed_shapes: List[List[Int]] = [[1], [0, 9223372036854775807, 2]]
    var mixed_recipe = packing_plan(Span(mixed_shapes), "*")
    var singleton: List[Float64] = [42]
    var mixed_views = mixed_recipe.unpack_views(Span(singleton))
    assert_equal(mixed_views[0][0], Float64(42))
    assert_equal(List(mixed_views[1].shape()), mixed_shapes[1])
    assert_equal(len(mixed_views[1].materialize()), 0)

    var floating: List[List[Float64]] = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    var matrices: List[List[Int]] = [[2, 3], [3, 2]]
    var contraction = einsum_plan(Span(matrices), "row dim, dim col -> row col")
    var zero_shapes: List[List[Int]] = [[2, 0], [0, 3]]
    var zero_operands: List[List[Float64]] = [[], []]
    var zero_contraction = einsum_plan(Span(zero_shapes), "a k, k b -> a b")
    var zero_target = List[Float64](length=6, fill=99)
    zero_contraction.execute_into(Span(zero_operands), Span(zero_target))
    for value in zero_target:
        assert_equal(value, Float64(0))
    var original = floating[0].copy()
    var target = Span[Float64, MutUntrackedOrigin](
        unsafe_ptr=floating[0]
        .unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        length=4,
    )
    rejected = False
    try:
        contraction.execute_into(Span(floating), target)
    except e:
        rejected = "overlap" in String(e)
    assert_true(rejected)
    assert_equal(floating[0], original)

    var input_shapes: List[List[Int]] = [[6], [6]]
    var flat_recipe = packing_plan(Span(input_shapes)[:1], "*")
    var alias = Span[Float64, MutUntrackedOrigin](
        unsafe_ptr=floating[0]
        .unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        length=6,
    )
    rejected = False
    try:
        flat_recipe.pack_into(Span(floating)[:1], alias)
    except e:
        rejected = "overlap" in String(e)
    assert_true(rejected)
    assert_equal(floating[0], original)

    var damaged = contraction.copy()
    damaged._strides[0][0] = 1000
    var destination = List[Float64](length=4, fill=99)
    rejected = False
    try:
        damaged.execute_into(Span(floating), Span(destination))
    except e:
        rejected = "storage" in String(e)
    assert_true(rejected)
    assert_equal(destination[0], Float64(99))
    var invalid_shapes: List[List[Int]] = [[2, 3], [4, 2]]
    rejected = False
    try:
        var bad = einsum_plan(
            Span(invalid_shapes), "row dim, dim col -> row col"
        )
    except e:
        rejected = "dimensions disagree" in String(e)
    assert_true(rejected)
    for bad_pattern in [
        String("b c"),
        String("b * * c"),
        String("b b *"),
        String("... *"),
    ]:
        rejected = False
        try:
            var bad = packing_plan(Span(shapes), bad_pattern)
        except:
            rejected = True
        assert_true(rejected)
    var packed_shape: List[Int] = [2, 3, 3]
    var bad_hints: List[List[Int]] = [[-1], [-1]]
    rejected = False
    try:
        var bad = unpacking_plan(Span(packed_shape), Span(bad_hints), "b * c")
    except:
        rejected = True
    assert_true(rejected)
    var simple_shape: List[Int] = [2, 3]
    for bad in [String("a a"), String("... ..."), String("a"), String("1 a")]:
        rejected = False
        try:
            var parsed = parse_shape(Span(simple_shape), bad)
        except:
            rejected = True
        assert_true(rejected)
    print(
        "PASS: immutable metadata, high axes, exact integer packing, empty"
        " buffers, alias rejection and malformed inputs"
    )
