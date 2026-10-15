"""Matched prepared-plan timings; compilation and allocation of inputs are excluded."""
from std.collections import Array, ImmSpan, List, Span
from std.time import perf_counter
from std.sys.info import size_of
from axisweave import packing_plan, einsum_plan, mix_plan
from axisweave._sources import _OwnedSources, _BorrowedSources, _PairSources


def main() raises:
    var shapes: List[List[Int]] = [[32, 56], [56, 24]]
    var inputs: List[List[Float32]] = [
        List[Float32](length=32 * 56, fill=0.125),
        List[Float32](length=56 * 24, fill=0.25),
    ]
    var contraction = einsum_plan(
        Span(shapes), "batch input, input output -> batch output"
    )
    var flat_shapes: List[List[Int]] = [[32 * 56], [56 * 24]]
    var packing = packing_plan(Span(flat_shapes), "*")
    var product = List[Float32](length=32 * 24, fill=0)
    var packed = List[Float32](length=32 * 56 + 56 * 24, fill=0)
    var owned_trials = List[Float64]()
    var borrowed_trials = List[Float64]()
    var owned_sum = Float64(0)
    var borrowed_sum = Float64(0)
    for trial in range(5):
        for pass_index in range(2):
            var borrowed = (trial + pass_index) % 2 == 0
            var start = perf_counter()
            for iteration in range(100):
                inputs[0][0] = Float32(0.125) + Float32(
                    iteration % 7
                ) * Float32(0.001)
                if borrowed:
                    var spans: Array[
                        ImmSpan[Float32, origin_of(inputs[0])], 2
                    ] = [
                        ImmSpan[Float32, origin_of(inputs[0])](inputs[0]),
                        ImmSpan[Float32, origin_of(inputs[0])](inputs[1]),
                    ]
                    contraction.execute_into(Span(spans), Span(product))
                    packing.pack_into(Span(spans), Span(packed))
                    borrowed_sum += Float64(product[0] + packed[0])
                else:
                    contraction.execute_into(Span(inputs), Span(product))
                    packing.pack_into(Span(inputs), Span(packed))
                    owned_sum += Float64(product[0] + packed[0])
            var elapsed = perf_counter() - start
            if borrowed:
                borrowed_trials.append(elapsed)
            else:
                owned_trials.append(elapsed)
    print(
        String(
            '{"owned_seconds":',
            owned_trials,
            ',"borrowed_seconds":',
            borrowed_trials,
            ',"owned_checksum":',
            owned_sum,
            ',"borrowed_checksum":',
            borrowed_sum,
            ',"adapter_bytes":[',
            size_of[_OwnedSources[.float32, origin_of(inputs[0])]](),
            ",",
            size_of[
                _BorrowedSources[
                    .float32, origin_of(inputs[0]), origin_of(shapes)
                ]
            ](),
            ",",
            size_of[
                _PairSources[
                    .float32, origin_of(inputs[0]), origin_of(inputs[0])
                ]
            ](),
            "]}",
        )
    )
