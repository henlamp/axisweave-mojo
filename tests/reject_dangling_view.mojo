"""This file must fail compilation: a local buffer cannot escape in a view."""
from std.collections import List, Span
from axisweave import TensorView, rearrange_view


def dangling() raises -> TensorView[Float64, ImmStaticOrigin]:
    var local: List[Float64] = [1, 2]
    var shape: List[Int] = [2]
    return rearrange_view(Span(local), Span(shape), "a -> a")


def main() raises:
    var escaped = dangling()
    print(escaped[0])
