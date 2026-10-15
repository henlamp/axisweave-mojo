from std.collections import List, Span
from axisweave import packing_plan, rearrange_view


def main() raises:
    var values: List[Float64] = [1, 2]
    var shape: List[Int] = [2]
    var borrowed = rearrange_view(Span(values), Span(shape), "a -> a")
    var dimensions = borrowed.shape()
    dimensions[0] = 9
    var shapes: List[List[Int]] = [[2]]
    var recipe = packing_plan(Span(shapes), "*")
    var input_dimensions = recipe.input_shape(0)
    input_dimensions[0] = 9
