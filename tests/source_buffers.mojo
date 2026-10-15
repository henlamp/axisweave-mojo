"""Borrowed operand lifetimes, overlapping buffers and validation before writes."""
from std.collections import Array, ImmSpan, List, Span
from std.origin import MutUntrackedOrigin
from std.testing import assert_equal, assert_true
from axisweave import packing_plan, einsum_plan, mix_plan, rearrange_view


def main() raises:
    var exact: List[Int64] = [
        9007199254740993,
        9223372036854775807,
        -9007199254740993,
    ]
    var integer_buffers: Array[ImmSpan[Int64, origin_of(exact)], 1] = [
        Span(exact).as_imm()
    ]
    var integer_shapes: List[List[Int]] = [[3]]
    assert_equal(
        packing_plan(Span(integer_shapes), "*").pack(Span(integer_buffers)),
        exact,
    )
    var storage: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    var inputs: Array[ImmSpan[Float32, origin_of(storage)], 2] = [
        Span(storage).as_imm()[:6],
        Span(storage).as_imm()[6:],
    ]
    var shapes: List[List[Int]] = [[2, 3], [2, 3]]
    var recipe = packing_plan(Span(shapes), "batch *")
    var wanted: List[Float32] = [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]
    assert_equal(recipe.pack(Span(inputs)), wanted)
    var out = List[Float32](length=12, fill=-1)
    recipe.pack_into(Span(inputs), Span(out))
    assert_equal(out, wanted)
    var split = List[Float32](length=12, fill=0)
    var targets = Array[Span[mut=True, Float32, origin_of(split)], 2]()
    targets[0] = Span(split)[:6]
    targets[1] = Span(split)[6:]
    recipe.unpack_into(Span(out), Span(targets))
    assert_equal(split, storage)
    var bad = List[Float32](length=11, fill=19)
    var bad_targets = Array[Span[mut=True, Float32, origin_of(bad)], 2]()
    bad_targets[0] = Span(bad)[:6]
    bad_targets[1] = Span(bad)[6:]
    var rejected = False
    try:
        recipe.unpack_into(Span(out), Span(bad_targets))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(bad, List[Float32](length=11, fill=19))
    var collision = Array[Span[mut=True, Float32, origin_of(bad)], 2]()
    collision[0] = Span(bad)[:6]
    collision[1] = Span(bad)[5:]
    rejected = False
    try:
        recipe.unpack_into(Span(out), Span(collision))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(bad, List[Float32](length=11, fill=19))

    var contract_shapes: List[List[Int]] = [[2, 3], [3, 2]]
    var contraction = einsum_plan(
        Span(contract_shapes), "row dim, dim column -> row column"
    )
    var result = contraction.execute(Span(inputs))
    var product: List[Float32] = [58, 64, 139, 154]
    assert_equal(result, product)
    # Two pointer spans deliberately describe overlapping foreign storage.
    var aliased = Span[mut=True, Float32, MutUntrackedOrigin](
        unsafe_ptr=storage.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ](),
        length=4,
    )
    rejected = False
    try:
        contraction.execute_into(Span(inputs), aliased)
    except:
        rejected = True
    assert_true(rejected)
    var unchanged: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert_equal(storage, unchanged)
    var invalid: Array[ImmSpan[Float32, origin_of(storage)], 2] = [
        Span(storage).as_imm()[:6],
        Span(storage).as_imm()[7:],
    ]
    var untouched = List[Float32](length=4, fill=23)
    rejected = False
    try:
        contraction.execute_into(Span(invalid), Span(untouched))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(untouched, List[Float32](length=4, fill=23))
    var view = rearrange_view(
        Span(storage).as_imm()[:6], recipe.input_shape(0), "row dim -> dim row"
    )
    var transposed = List[Float32](length=6, fill=0)
    view.materialize_into(Span(transposed))
    var transpose_wanted: List[Float32] = [1, 4, 2, 5, 3, 6]
    assert_equal(transposed, transpose_wanted)

    var bias: List[Float32] = [10, 20]
    var bias_shape: List[Int] = [2]
    var mixed = mix_plan(
        Span(contract_shapes[0]),
        Span(contract_shapes[1]),
        "row dim, dim column -> row column",
        bias_shape=Span(bias_shape),
        bias_axes="column",
    )
    var biased = mixed.execute(inputs[0], inputs[1], bias=Span(bias))
    var biased_wanted: List[Float32] = [68, 84, 149, 174]
    assert_equal(biased, biased_wanted)
    rejected = False
    try:
        mixed.execute_into(
            inputs[0], inputs[1], Span(untouched), bias=Span(bias).as_imm()[:1]
        )
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(untouched, List[Float32](length=4, fill=23))
    var bias_alias = Span[mut=True, Float32, MutUntrackedOrigin](
        unsafe_ptr=bias.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        length=2,
    )
    var scalar_shape: List[Int] = []
    var scalar: List[Float32] = [1]
    var vector_mix = mix_plan(
        Span(bias_shape),
        Span(scalar_shape),
        "column, -> column",
        bias_shape=Span(bias_shape),
        bias_axes="column",
    )
    rejected = False
    try:
        vector_mix.execute_into(
            Span(bias), Span(scalar), bias_alias, bias=Span(bias)
        )
    except:
        rejected = True
    assert_true(rejected)
    var bias_wanted: List[Float32] = [10, 20]
    assert_equal(bias, bias_wanted)
    var matrix_bias_shape: List[Int] = [2, 2]
    var matrix_bias: List[Float32] = [10, 20, 30, 40]
    var permuted_bias = mix_plan(
        Span(contract_shapes[0]),
        Span(contract_shapes[1]),
        "row dim, dim column -> row column",
        bias_shape=Span(matrix_bias_shape),
        bias_axes="column row",
    )
    var permuted_wanted: List[Float32] = [68, 94, 159, 194]
    assert_equal(
        permuted_bias.execute(inputs[0], inputs[1], bias=Span(matrix_bias)),
        permuted_wanted,
    )
    var broadcast_scalar = mix_plan(
        Span(contract_shapes[0]),
        Span(contract_shapes[1]),
        "row dim, dim column -> row column",
        bias_shape=Span(scalar_shape),
        bias_axes="",
    )
    var scalar_bias: List[Float32] = [5]
    var scalar_wanted: List[Float32] = [63, 69, 144, 159]
    assert_equal(
        broadcast_scalar.execute(inputs[0], inputs[1], bias=Span(scalar_bias)),
        scalar_wanted,
    )
    mixed._bias.value()._data.operation = .rearrange
    rejected = False
    try:
        mixed.execute_into(
            inputs[0], inputs[1], Span(untouched), bias=Span(bias)
        )
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(untouched, List[Float32](length=4, fill=23))
    rejected = False
    try:
        var invalid_bias = mix_plan(
            Span(contract_shapes[0]),
            Span(contract_shapes[1]),
            "row dim, dim column -> row column",
            bias_shape=Span(bias_shape),
            bias_axes="missing",
        )
    except:
        rejected = True
    assert_true(rejected)
    print(
        "PASS: borrowed operands, direct unpacking, view output, named mixing"
        " and unchanged destinations on rejection"
    )
