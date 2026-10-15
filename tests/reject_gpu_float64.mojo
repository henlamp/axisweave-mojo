from max.gpu.host import DeviceContext
from std.collections import List, Span
from axisweave import plan
from axisweave.gpu import DeviceTransform


def main() raises:
    var ctx = DeviceContext()
    var shape: List[Int] = [2]
    var gpu = DeviceTransform(ctx, plan("a -> a", Span(shape)))
    var source = ctx.enqueue_create_buffer[.float64](2)
    _ = gpu.execute(source)
