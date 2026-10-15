from std.collections import List, Span
from std.origin import ImmUntrackedOrigin
from axisweave import plan


def escaped() raises -> Span[Int, ImmUntrackedOrigin]:
    var dims: List[Int] = [2, 3]
    var recipe = plan("a b -> b a", Span(dims))
    return recipe.output_shape()


def main() raises:
    print(escaped())
