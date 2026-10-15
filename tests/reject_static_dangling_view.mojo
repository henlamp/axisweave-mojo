"""Must fail: a static view cannot outlive its local buffer."""
from std.collections import List, Span
from axisweave import StaticRecipe, StaticTensorView


def dangling() raises -> (
    StaticTensorView["a -> a", 2, Element=Float64, origin=ImmStaticOrigin]
):
    var source: List[Float64] = [1, 2]
    var recipe = StaticRecipe["a -> a", 2]()
    return recipe.view(Span(source))


def main() raises:
    var invalid = dangling()
    print(len(invalid))
