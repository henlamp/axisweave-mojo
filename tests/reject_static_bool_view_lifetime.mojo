"""Must fail: a static Bool view cannot escape its source storage."""
from std.collections import List, Span
from axisweave import StaticRecipe, StaticTensorView


def escaped() raises -> (
    StaticTensorView["row -> row", 1, Element=Bool, origin=ImmStaticOrigin]
):
    var source: List[Bool] = [True]
    return StaticRecipe["row -> row", 1]().view(Span(source))


def main() raises:
    var invalid = escaped()
    print(len(invalid))
