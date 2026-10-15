from std.collections import List, Span
from axisweave import plan


def main() raises:
    var dims: List[Int] = [2, 3]
    var recipe = plan("a b -> b a", Span(dims))
    var shape = recipe.output_shape()
    shape[0] = 99
