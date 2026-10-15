"""Patterns, shapes and split sizes are all resolved by the Mojo compiler."""

from std.collections import List, Span
from axisweave import StaticRecipe


def main() raises:
    var image: List[Float32] = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    var patches = StaticRecipe[
        "b (h ph) (w pw) c -> b (h w) (ph pw c)", 1, 4, 4, 1, axes="ph=2 pw=2"
    ]()
    var borrowed = patches.view(Span(image))
    print("static patch shape:", borrowed.shape())
    print("first patch:", borrowed[0], borrowed[1], borrowed[2], borrowed[3])
    print("static patch values:", borrowed.materialize())
    var pooled = StaticRecipe[
        "b (h ph) (w pw) c -> b h w c",
        1,
        4,
        4,
        1,
        operation=.reduce,
        axes="ph=2 pw=2",
    ]()
    print("static pooling:", pooled.execute[.mean](Span(image)))
