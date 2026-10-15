"""Zero extents, bounded dimensions and empty compiled recipes."""
from std.collections import Dict, List, Span
from std.testing import assert_equal, assert_true
from axisweave import (
    StaticRecipe,
    plan,
    execute,
    packing_plan,
    unpacking_plan,
    einsum_plan,
    view,
)


def main() raises:
    var empty = List[Float64]()
    var permutations: List[List[Int]] = [
        [9223372036854775807, 2, 0],
        [9223372036854775807, 0, 2],
        [2, 9223372036854775807, 0],
        [2, 0, 9223372036854775807],
        [0, 9223372036854775807, 2],
        [0, 2, 9223372036854775807],
    ]
    for shape in permutations:
        var recipe = plan("a b c -> c b a", Span(shape))
        assert_equal(recipe.input_size(), 0)
        assert_equal(recipe.output_size(), 0)
        assert_equal(len(execute(recipe, Span(empty))), 0)
        var borrowed = view(recipe, Span(empty))
        assert_equal(len(borrowed.materialize()), 0)
        var reduced = plan("a b c ->", Span(shape), .reduce)
        var sum = execute(reduced, Span(empty), .sum)
        assert_equal(sum[0], Float64(0))
        var shapes: List[List[Int]] = [shape.copy()]
        var sources: List[List[Float64]] = [[]]
        var contraction = einsum_plan(Span(shapes), "a b c ->")
        assert_equal(contraction.execute(Span(sources))[0], Float64(0))
    var static = StaticRecipe["a b c -> c b a", 9223372036854775807, 2, 0]()
    assert_equal(len(static.execute(Span(empty))), 0)
    var static_sum = StaticRecipe[
        "a b c ->", 9223372036854775807, 2, 0, operation=.reduce
    ]()
    assert_equal(static_sum.execute[.sum](Span(empty))[0], Float64(0))

    var flat: List[Int] = [0]
    var hints = Dict[String, Int]()
    hints["a"] = 9223372036854775807
    hints["b"] = 2
    var split = plan("(a b c) -> a b c", Span(flat), axes_lengths=hints)
    var wanted: List[Int] = [9223372036854775807, 2, 0]
    assert_equal(List(split.output_shape()), wanted)
    var compose = plan("a b c -> (a b c)", Span(wanted))
    assert_equal(List(compose.output_shape()), flat)

    var shapes: List[List[Int]] = [
        [9223372036854775807, 9223372036854775807, 1, 0],
        [9223372036854775807, 9223372036854775807, 2, 0],
    ]
    var inputs: List[List[Float64]] = [[], []]
    var packed = packing_plan(Span(shapes), "a b * c")
    assert_equal(len(packed.pack(Span(inputs))), 0)
    var borrowed = packed.unpack_views(Span(empty))
    for i in range(2):
        assert_equal(List(borrowed[i].shape()), shapes[i])
        assert_equal(len(borrowed[i].materialize()), 0)
    var packed_parts = packed.packed_shapes()
    var recovered = unpacking_plan(
        packed.output_shape(), Span(packed_parts), "a b * c"
    )
    assert_equal(len(recovered.unpack(Span(empty))), 2)

    var zero_axis: List[Int] = [0]
    var known_parts: List[List[Int]] = [[9223372036854775807, 2, 0]]
    var known_unpack = unpacking_plan(Span(zero_axis), Span(known_parts), "*")
    assert_equal(known_unpack.packed_shapes(), known_parts)
    var inferred_parts: List[List[Int]] = [[9223372036854775807, 2, -1]]
    var inferred_unpack = unpacking_plan(
        Span(zero_axis), Span(inferred_parts), "*"
    )
    assert_equal(inferred_unpack.packed_shapes(), known_parts)

    var bad_shapes: List[List[Int]] = [[0, -1, 2], [9223372036854775807, 2]]
    for shape in bad_shapes:
        var rejected = False
        try:
            var bad = plan("... -> ...", Span(shape))
        except:
            rejected = True
        assert_true(rejected)
    var rejected = False
    try:
        var bad = plan("a b c -> (a b) c", Span(wanted))
    except:
        rejected = True
    assert_true(rejected)
    hints["a"] = 0
    rejected = False
    try:
        var ambiguous = plan("(a b c) -> a b c", Span(flat), axes_lengths=hints)
    except:
        rejected = True
    assert_true(rejected)
    print(
        "PASS: zero products are order-independent; large empty layouts work"
        " and invalid extents remain rejected"
    )
