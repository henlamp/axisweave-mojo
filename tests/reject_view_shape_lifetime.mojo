from std.collections import List, Span
from std.origin import ImmUntrackedOrigin
from axisweave import rearrange_view


def escape() raises -> Span[Int, ImmUntrackedOrigin]:
    var values: List[Float64] = [1, 2]
    var shape: List[Int] = [2]
    var view = rearrange_view(Span(values), Span(shape), "a -> a")
    return view.shape()


def main() raises:
    print(escape())
