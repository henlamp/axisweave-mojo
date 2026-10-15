"""Must fail at compile time: seven elements cannot split into a*2."""
from axisweave import StaticRecipe


def main():
    var invalid = StaticRecipe["(a b) -> a b", 7, axes="b=2"]()
    print(invalid.output_size)
