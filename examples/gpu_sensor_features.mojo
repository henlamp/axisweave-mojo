# SPDX-License-Identifier: MIT
"""Keep sensor patches, projection and aggregation on the GPU."""
from max.gpu.host import DeviceContext
from std.collections import Dict, List, Span
from std.math import isfinite
from std.sys import argv
from std.testing import assert_true
from axisweave import plan, einsum_plan, execute
from axisweave.gpu import DeviceTransform, DeviceEinsum


def main() raises:
    var ctx = DeviceContext()
    var image_shape: List[Int] = [4, 16, 16, 3]
    var shape: List[List[Int]] = [[256, 12], [12, 8]]
    var hints = Dict[String, Int]()
    hints["ph"] = 2
    hints["pw"] = 2
    var patches = plan(
        "batch (row ph) (col pw) channel -> (batch row col) (ph pw channel)",
        Span(image_shape),
        axes_lengths=hints,
    )
    var projection = einsum_plan(
        Span(shape), "patch channel, channel feature -> patch feature"
    )
    var features_shape: List[Int] = [256, 8]
    var average = plan("patch feature -> patch", Span(features_shape), .reduce)
    var gpu_patches = DeviceTransform(ctx, patches)
    var gpu_projection = DeviceEinsum(ctx, projection)
    var gpu_average = DeviceTransform(ctx, average, .mean)
    var image = ctx.enqueue_create_buffer[.float32](4 * 16 * 16 * 3)
    var weights = ctx.enqueue_create_buffer[.float32](12 * 8)
    var source = List[Float32](length=len(image), fill=0)
    var matrix = List[Float32](length=len(weights), fill=0)
    with image.map_to_host() as host:
        for i in range(len(source)):
            source[i] = Float32(i % 97 - 48) / 32
            host[i] = source[i]
    with weights.map_to_host() as host:
        for i in range(len(matrix)):
            matrix[i] = Float32(i % 11 - 5) / 16
            host[i] = matrix[i]
    var device_patches = gpu_patches.execute(image)
    var device_features = gpu_projection.execute(device_patches, weights)
    var result = gpu_average.execute(device_features)
    var operands: List[List[Float32]] = [execute(patches, Span(source)), matrix^]
    var cpu_features = projection.execute(Span(operands))
    var cpu = execute(average, Span(cpu_features), .mean)
    var values = List[Float32](length=len(result), fill=0)
    with result.map_to_host() as host:
        for i in range(len(values)):
            values[i] = host[i]
            assert_true(isfinite(values[i]))
            assert_true(abs(values[i] - cpu[i]) <= 3e-5 * (1 + abs(cpu[i])))
    if len(argv()) > 1 and argv()[1] == "--values":
        print(String('{"values":', values, "}"))
    else:
        print(
            "PASS:",
            ctx.name(),
            "sensor patches -> projection -> mean;",
            len(values),
            "CPU-checked outputs",
        )
