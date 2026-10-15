from std.collections import List, Span
from std.origin import ImmUntrackedOrigin
from axisweave import plan, ReductionValues


def escape() raises -> (
    ReductionValues[.float32, ImmUntrackedOrigin, ImmUntrackedOrigin]
):
    var data: List[Float32] = [1, 2, 3]
    var shape: List[Int] = [3]
    var recipe = plan("axis ->", Span(shape), .reduce)
    return ReductionValues(Span(data), recipe, 0)


def main() raises:
    _ = escape()
