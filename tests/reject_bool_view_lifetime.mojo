"""Must fail: a Bool view cannot escape its source storage."""
from std.collections import List, Span
from axisweave import TensorView, rearrange_view


def escaped() raises -> TensorView[Bool, ImmStaticOrigin]:
    var source: List[Bool] = [True]
    var shape: List[Int] = [1]
    return rearrange_view(Span(source), Span(shape), "row -> row")


def main() raises:
    var invalid = escaped()
    print(len(invalid))
