from std.collections import List, Span
from std.origin import ImmUntrackedOrigin
from axisweave import packing_plan


def escape() raises -> Span[Int, ImmUntrackedOrigin]:
    var shapes: List[List[Int]] = [[2, 3]]
    var recipe = packing_plan(Span(shapes), "b *")
    return recipe.input_shape(0)


def main() raises:
    print(escape())
