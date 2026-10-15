"""Pack differently shaped sensor tensors, borrow splits, and project features."""
from std.collections import List, Span
from axisweave import packing_plan, einsum_plan


def main() raises:
    var shapes: List[List[Int]] = [[2, 3], [2, 2, 3]]
    var observations: List[List[Float32]] = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ]
    var layout = packing_plan(Span(shapes), "environment * channel")
    var packed = layout.pack(Span(observations))
    print("packed dimensions", layout.output_shape())
    var split = layout.unpack_views(Span(packed))
    print("second sensor", split[1].shape(), split[1][0])
    var weights: List[Float32] = [1, 0, 0, 1, 1, 1]
    var operands: List[List[Float32]] = [packed^, weights^]
    var dimensions: List[List[Int]] = [[2, 3, 3], [3, 2]]
    var projection = einsum_plan(
        Span(dimensions),
        "environment token channel, channel output -> environment token output",
    )
    print("projected features", projection.execute(Span(operands)))
