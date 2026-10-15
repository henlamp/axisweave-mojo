"""Turn an image into patches, then pool it, using only Mojo."""

from std.collections import Dict, List, Span
from axisweave import plan, reduce, view


def main() raises:
    var image: List[Float64] = [
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
    var shape: List[Int] = [1, 4, 4, 1]
    var axes = Dict[String, Int]()
    axes["ph"] = 2
    axes["pw"] = 2
    var pattern = String("b (h ph) (w pw) c -> b (h w) (ph pw c)")
    var recipe = plan(pattern, Span(shape), axes_lengths=axes)
    print("patch shape:", List(recipe.output_shape()))
    var patches = view(recipe, Span(image))
    print(
        "first patch without copying:",
        patches[0],
        patches[1],
        patches[2],
        patches[3],
    )
    print("patch values:", patches.materialize())
    print(
        "pooled values:",
        reduce(
            Span(image),
            Span(shape),
            "b (h ph) (w pw) c -> b h w c",
            .mean,
            axes,
        ),
    )
