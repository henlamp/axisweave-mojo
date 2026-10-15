from std.collections import List, Span
from std.origin import ImmUntrackedOrigin
from axisweave import packing_plan, TensorView


def escaped() raises -> List[TensorView[Float64, ImmUntrackedOrigin]]:
    var dims: List[List[Int]] = [[2], [3]]
    var recipe = packing_plan(Span(dims), "*")
    var values: List[Float64] = [0, 1, 2, 3, 4]
    return recipe.unpack_views(Span(values))


def main() raises:
    var values = escaped()
    print(len(values))
