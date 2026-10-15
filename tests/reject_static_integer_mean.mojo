from std.collections import List, Span
from axisweave import StaticRecipe

def main() raises:
    var data: List[Int64] = [1,2,3]
    var recipe = StaticRecipe["axis ->",3,operation=.reduce]()
    _ = recipe.execute[.mean](Span(data))
