from std.collections import List, Span
from std.origin import ImmUntrackedOrigin
from max.gpu.host import DeviceContext
from axisweave import plan
from axisweave.gpu import DeviceTransform


def escaped(ctx: DeviceContext) raises -> Span[Int, ImmUntrackedOrigin]:
    var dims: List[Int] = [2, 3]
    var recipe = DeviceTransform(ctx, plan("a b -> b a", Span(dims)))
    return recipe.output_shape()


def main() raises:
    var ctx = DeviceContext()
    print(escaped(ctx))
