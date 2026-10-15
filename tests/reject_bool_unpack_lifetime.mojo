"""Must fail: unpacked Bool views retain their source lifetime."""
from std.collections import List, Span
from axisweave import TensorView, packing_plan


def escaped() raises -> List[TensorView[Bool, ImmStaticOrigin]]:
    var shapes: List[List[Int]] = [[1]]
    var source: List[Bool] = [True]
    return packing_plan(Span(shapes), "*").unpack_views(Span(source))


def main() raises:
    var invalid = escaped()
    print(len(invalid))
