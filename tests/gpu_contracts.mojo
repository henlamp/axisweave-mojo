# SPDX-License-Identifier: MIT
"""Device boundaries, aliases, lifetime ordering and retained Float64 CPU precision."""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import Dict, List, Span
from std.testing import assert_equal, assert_true
from axisweave import plan, execute, einsum_plan, Reduction
from axisweave.gpu import DeviceTransform, DeviceEinsum
from gpu_references import upload, check, check_value


def pending(ctx: DeviceContext) raises -> DeviceBuffer[.float32]:
    var shape: List[Int] = [2, 3]
    var source: List[Float32] = [1, 2, 3, 4, 5, 6]
    var input = upload(ctx, Span(source))
    var first = DeviceTransform(ctx, plan("a b -> b a", Span(shape)))
    var intermediate = first.execute(input)
    shape = [3, 2]
    var second = DeviceTransform(ctx, plan("a b -> b", Span(shape), .reduce))
    return second.execute(intermediate)


def main() raises:
    var ctx = DeviceContext()
    var other = DeviceContext()
    var shape: List[Int] = [2, 3]
    var recipe = plan("a b -> b a", Span(shape))
    var transform = DeviceTransform(ctx, recipe)
    var source: List[Float32] = [1, 2, 3, 4, 5, 6]
    var sentinel = List[Float32](length=12, fill=987654)
    var input = upload(ctx, Span(source))
    var short = ctx.enqueue_create_buffer[.float32](5)
    var foreign = other.enqueue_create_buffer[.float32](6)
    var parent = upload(ctx, Span(sentinel))
    var first = parent.create_sub_buffer[.float32](0, 6)
    var overlap = parent.create_sub_buffer[.float32](4, 6)
    var output = parent.create_sub_buffer[.float32](6, 6)
    for attempt in range(10):
        var rejected = False
        try:
            if attempt == 0:
                transform.execute_into(short, output)
            elif attempt == 1:
                transform.execute_into(input, short)
            elif attempt == 2:
                transform.execute_into(foreign, output)
            elif attempt == 3:
                transform.execute_into(input, foreign)
            elif attempt == 4:
                transform.execute_into(input, input)
            elif attempt == 5:
                transform.execute_into(first, overlap)
            elif attempt == 6:
                var cpu = DeviceContext(api="cpu")
                _ = DeviceTransform(cpu, recipe)
            elif attempt == 7:
                _ = DeviceTransform(
                    ctx, plan("a b -> a", Span(shape), .reduce), .any
                )
            elif attempt == 8:
                var dims = List[Int]()
                var lengths = Dict[String, Int]()
                lengths["copies"] = 9223372036854775807
                _ = DeviceTransform(
                    ctx, plan(" -> copies", Span(dims), .repeat, lengths)
                )
            else:
                var malformed = recipe.copy()
                malformed._data.output_size = 7
                _ = DeviceTransform(ctx, malformed)
        except:
            rejected = True
        assert_true(rejected, String("Missing transform rejection ", attempt))
        check(parent, Span(sentinel), "Rejected operation wrote output")
        check(input, Span(source), "Rejected alias modified input")
    # Disjoint sub-buffers of the same allocation are valid.
    ctx.enqueue_copy(src_buf=input, dst_buf=first)
    var copy = transform.copy()
    copy.execute_into(first, output)
    var transpose: List[Float32] = [1, 4, 2, 5, 3, 6]
    check(output, Span(transpose), "Disjoint sub-buffers")
    # Exercise wide indexing with a legal large stride on a singleton axis.
    var wide_shape: List[Int] = [1, 6]
    var wide_strides: List[Int] = [1099511627776, 1]
    var wide = DeviceTransform(
        ctx,
        plan(
            "unit element -> element unit",
            Span(wide_shape),
            strides=wide_strides,
        ),
    )
    assert_true(not wide._narrow)
    var wide_result = wide.execute(input)
    check(wide_result, Span(source), "64-bit transform indexing")
    var wide_shapes: List[List[Int]] = [[1, 3], [3, 2]]
    var wide_recipe = einsum_plan(Span(wide_shapes), "a k, k b -> a b")
    wide_recipe._strides[0][0] = 1099511627776
    var wide_contraction = DeviceEinsum(ctx, wide_recipe)
    assert_true(not wide_contraction._narrow)
    var three = input.create_sub_buffer[.float32](0, 3)
    var wide_product = wide_contraction.execute(three, input)
    var wide_wanted: List[Float32] = [22, 28]
    check(wide_product, Span(wide_wanted), "64-bit contraction indexing")
    # Plan/input/intermediate go out of scope before this readback synchronizes.
    var result = pending(ctx)
    var sums: List[Float32] = [6, 15]
    check(result, Span(sums), "Queued lifetime ordering")
    var shapes: List[List[Int]] = [[2, 3], [3, 2]]
    var contraction = DeviceEinsum(
        ctx, einsum_plan(Span(shapes), "a k, k b -> a b")
    )
    var small_output = ctx.enqueue_create_buffer[.float32](3)
    var coutput = parent.create_sub_buffer[.float32](8, 4)
    var calias = input.create_sub_buffer[.float32](1, 4)
    var cforeign = other.enqueue_create_buffer[.float32](4)
    for attempt in range(8):
        var rejected = False
        try:
            if attempt == 0:
                contraction.execute_into(short, input, coutput)
            elif attempt == 1:
                contraction.execute_into(input, short, coutput)
            elif attempt == 2:
                contraction.execute_into(input, input, small_output)
            elif attempt == 3:
                contraction.execute_into(foreign, input, coutput)
            elif attempt == 4:
                contraction.execute_into(input, foreign, coutput)
            elif attempt == 5:
                contraction.execute_into(input, input, cforeign)
            elif attempt == 6:
                contraction.execute_into(input, input, calias)
            else:
                var unary: List[List[Int]] = [[2, 3]]
                _ = DeviceEinsum(ctx, einsum_plan(Span(unary), "a k -> a"))
        except:
            rejected = True
        assert_true(rejected, String("Missing contraction rejection ", attempt))
        check(input, Span(source), "Rejected contraction modified source")
    # A Float32 conversion here would lose the low bit.
    var precise: List[Float64] = [16777217, 1, 2, 3, 4, 5]
    var cpu = execute(plan("a b -> a", Span(shape), .reduce), Span(precise))
    var wanted: List[Float64] = [16777220, 12]
    assert_equal(cpu, wanted)
    var special: List[Float32] = [
        0.0,
        -0.0,
        Float32(0) / Float32(0),
        1,
        Float32(1) / Float32(0),
        -Float32(1) / Float32(0),
    ]
    special.append(Float32(1))
    special.append(Float32(0) / Float32(0))
    special.append(Float32(-0.0))
    special.append(Float32(0.0))
    var special_input = upload(ctx, Span(special))
    var pairs: List[Int] = [5, 2]
    for kind in range(5):
        var reduction = Reduction(kind)
        var r = plan("a b -> a", Span(pairs), .reduce)
        var gpu = DeviceTransform(ctx, r, reduction)
        var actual = gpu.execute(special_input)
        var expected = execute(r, Span(special), reduction)
        check(actual, Span(expected), String("NaN/Inf/signed zero ", kind))
    print(
        "PASS: GPU extents, contexts, overlapping/disjoint aliases, plan"
        " copies,"
    )
    print("queued lifetimes, NaN/Inf/signed zero and Float64 CPU precision")
