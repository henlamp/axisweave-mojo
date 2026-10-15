"""Must fail at compile time: reduction is computation, not a borrowed view."""
from std.collections import List, Span
from axisweave import StaticRecipe


def main() raises:
    var source: List[Float64] = [1, 2]
    var recipe = StaticRecipe["a ->", 2, operation=.reduce]()
    var invalid = recipe.view(Span(source))
    print(len(invalid))
