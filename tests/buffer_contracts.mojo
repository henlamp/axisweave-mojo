"""Element preservation and validation before any caller-buffer write."""
from std.collections import Array, ImmSpan, List, Span
from std.origin import MutUntrackedOrigin
from std.testing import assert_equal, assert_true
from axisweave import packing_plan, rearrange_view, StaticRecipe


@fieldwise_init
struct Sample(Defaultable, TrivialRegisterPassable):
    var identifier: UInt64
    var valid: Bool

    def __init__(out self):
        self.identifier = 0
        self.valid = False


struct Unit(Defaultable, TrivialRegisterPassable):
    def __init__(out self):
        pass


def scalar_views[dtype: DType]() raises:
    var data = List[Scalar[dtype]](length=34, fill=0)
    for i in range(len(data)):
        data[i] = Scalar[dtype](i)
    var shape: List[Int] = [2, 17]
    var borrowed = rearrange_view(Span(data), Span(shape), "row col -> col row")
    var fixed = StaticRecipe["row col -> col row", 2, 17]().view(Span(data))
    var dynamic_output = borrowed.materialize()
    var static_output = fixed.materialize()
    for col in range(17):
        for row in range(2):
            var wanted = data[row * 17 + col]
            assert_equal(dynamic_output[col * 2 + row], wanted)
            assert_equal(static_output[col * 2 + row], wanted)


def main() raises:
    scalar_views[.float16]()
    scalar_views[.bfloat16]()
    scalar_views[.int8]()
    scalar_views[.uint8]()
    scalar_views[.int16]()
    scalar_views[.uint16]()
    var shapes: List[List[Int]] = [[2], [2, 2]]
    var inputs: List[List[Bool]] = [[True, False], [False, True, True, False]]
    var recipe = packing_plan(Span(shapes), "row *")
    var target = List[Bool](length=6, fill=True)
    var sentinel = target.copy()
    var rejected = False
    var short: List[List[Bool]] = [[True, False], [True]]
    try:
        recipe.pack_into(Span(short), Span(target))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(target, sentinel)

    var packed = recipe.pack(Span(inputs))
    var split = List[Bool](length=6, fill=True)
    var before = split.copy()
    var destinations = Array[Span[mut=True, Bool, origin_of(split)], 2]()
    destinations[0] = Span(split)[:2]
    destinations[1] = Span(split)[2:5]
    rejected = False
    try:
        recipe.unpack_into(Span(packed), Span(destinations))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(split, before)
    destinations[1] = Span(split)[:4]
    rejected = False
    try:
        recipe.unpack_into(Span(packed), Span(destinations))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(split, before)

    var shape: List[Int] = [2, 3]
    var dynamic = rearrange_view(
        Span(packed), Span(shape), "row col -> col row"
    )
    var static = StaticRecipe["row col -> col row", 2, 3]().view(Span(packed))
    var alias = Span[Bool, MutUntrackedOrigin](
        unsafe_ptr=packed.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        length=len(packed),
    )
    var original = packed.copy()
    rejected = False
    try:
        dynamic.materialize_into(alias)
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(packed, original)
    rejected = False
    try:
        static.materialize_into(alias)
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(packed, original)
    var wrong = List[Bool](length=5, fill=True)
    rejected = False
    try:
        static.materialize_into(Span(wrong))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(wrong, List[Bool](length=5, fill=True))
    rejected = False
    try:
        var invalid = dynamic[len(dynamic)]
    except:
        rejected = True
    assert_true(rejected)
    rejected = False
    try:
        var invalid = static[-1]
    except:
        rejected = True
    assert_true(rejected)

    dynamic._recipe._data.source_strides[0] = 9223372036854775807
    var unchanged = List[Bool](length=6, fill=True)
    rejected = False
    try:
        dynamic.materialize_into(Span(unchanged))
    except:
        rejected = True
    assert_true(rejected)
    assert_equal(unchanged, List[Bool](length=6, fill=True))

    var records: List[Sample] = [
        Sample(18446744073709551615, True),
        Sample(7, False),
    ]
    var lengths: List[Int] = [2]
    var reordered = rearrange_view(Span(records), Span(lengths), "row -> row")
    var copied = reordered.materialize()
    assert_equal(copied[0].identifier, records[0].identifier)
    assert_equal(copied[1].valid, False)
    var record_shapes: List[List[Int]] = [[2]]
    var borrowed = Array[ImmSpan[Sample, origin_of(records)], 1]()
    borrowed[0] = Span(records)
    var record_plan = packing_plan(Span(record_shapes), "*")
    var record_pack = record_plan.pack(Span(borrowed))
    assert_equal(record_pack[0].identifier, records[0].identifier)
    assert_equal(record_pack[1].valid, False)
    var unit_buffers = List[List[Unit]]()
    unit_buffers.append(List[Unit](length=2, fill=Unit()))
    assert_equal(len(record_plan.pack(Span(unit_buffers))), 2)
    print(
        "PASS: Bool alias/extent/index rejection before writes, exact native"
        " records and zero-size elements"
    )
