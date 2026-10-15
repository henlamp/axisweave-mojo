# SPDX-License-Identifier: MIT
"""Native device checks against independent fixtures and the CPU implementation."""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import Dict, List, Span
from std.math import isfinite, isnan
from std.sys import argv
from std.testing import assert_equal, assert_true
from axisweave import plan, execute, einsum_plan, Operation, Reduction
from axisweave.gpu import DeviceTransform, DeviceEinsum
from property_transforms import integers


def numbers(text: String) raises -> List[Float32]:
    var values = List[Float32]()
    if text.byte_length() > 0:
        for part in text.split(","):
            values.append(Float32(Float64(String(part))))
    return values^


def upload(
    ctx: DeviceContext, values: Span[Float32, _]
) raises -> DeviceBuffer[.float32]:
    var result = ctx.enqueue_create_buffer[.float32](len(values))
    with result.map_to_host() as host:
        for i in range(len(values)):
            host[i] = values[i]
    return result^


def check_value(actual: Float32, expected: Float32, label: String) raises:
    if isnan(expected):
        assert_true(isnan(actual), label)
    elif not isfinite(expected):
        assert_equal(actual, expected, label)
    elif expected == 0 and actual == 0:
        assert_equal(Float32(1) / actual, Float32(1) / expected, label)
    else:
        assert_true(isfinite(actual), label)
        assert_true(abs(actual - expected) <= 3e-5 * (1 + abs(expected)), label)


def check(
    device: DeviceBuffer[.float32], expected: Span[Float32, _], label: String
) raises:
    assert_equal(len(device), len(expected), label)
    with device.map_to_host() as host:
        for i in range(len(expected)):
            check_value(host[i], expected[i], String(label, " element ", i))


def transformations(ctx: DeviceContext, path: String) raises:
    var text: String
    with open(path, "r") as input:
        text = input.read()
    var passed = 0
    var rejected = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = String(line).split("|")
        assert_equal(len(fields), 12)
        var operation = Operation.rearrange
        if fields[1] == "repeat":
            operation = .repeat
        elif fields[1] == "reduce":
            operation = .reduce
        var reduction = Reduction.sum
        if fields[2] == "mean":
            reduction = .mean
        elif fields[2] == "min":
            reduction = .min
        elif fields[2] == "max":
            reduction = .max
        elif fields[2] == "prod":
            reduction = .prod
        var shape = integers(String(fields[4]))
        var strides = integers(String(fields[6]))
        var source = numbers(String(fields[8]))
        var expected_shape = integers(String(fields[9]))
        var expected = numbers(String(fields[10]))
        var hints = Dict[String, Int]()
        if fields[5].byte_length() > 0:
            for hint in String(fields[5]).split(","):
                var kv = String(hint).split("=")
                hints[String(kv[0])] = Int(String(kv[1]))
        var sentinel = List[Float32](length=len(expected), fill=987654)
        var input = upload(ctx, Span(source))
        var output = upload(ctx, Span(sentinel))
        var failed = False
        try:
            var recipe = plan(
                String(fields[3]),
                Span(shape),
                operation,
                hints,
                strides=strides,
                offset=Int(String(fields[7])),
            )
            var gpu = DeviceTransform(ctx, recipe, reduction)
            gpu.execute_into(input, output)
            if fields[0] == "valid":
                assert_equal(List(gpu.output_shape()), expected_shape)
                var cpu = execute(recipe, Span(source), reduction)
                check(output, Span(cpu), String("CPU ", fields[11]))
                var owned = gpu.execute(input)
                check(owned, Span(expected), String("owned ", fields[11]))
                # Reuse metadata and output while the buffers remain on device.
                gpu.execute_into(input, output)
        except:
            failed = True
        if fields[0] == "invalid":
            assert_true(failed, String("Expected rejection ", fields[11]))
            check(output, Span(sentinel), "Rejection changed destination")
            rejected += 1
        else:
            assert_true(not failed, String("Unexpected failure ", fields[11]))
            check(output, Span(expected), String("NumPy ", fields[11]))
            check(input, Span(source), "GPU modified source")
            passed += 1
    print(
        "PASS:", passed, "GPU transforms;", rejected, "rejections before writes"
    )


def contractions(ctx: DeviceContext, path: String) raises:
    var text: String
    with open(path, "r") as input:
        text = input.read()
    var count = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = String(line).split("|")
        assert_equal(len(fields), 7)
        var shapes: List[List[Int]] = [
            integers(String(fields[1])),
            integers(String(fields[2])),
        ]
        var inputs: List[List[Float32]] = [
            numbers(String(fields[3])),
            numbers(String(fields[4])),
        ]
        var wanted_shape = integers(String(fields[5]))
        var wanted = numbers(String(fields[6]))
        var recipe = einsum_plan(Span(shapes), String(fields[0]))
        var gpu = DeviceEinsum(ctx, recipe)
        assert_equal(List(gpu.output_shape()), wanted_shape)
        var left = upload(ctx, Span(inputs[0]))
        var right = upload(ctx, Span(inputs[1]))
        var actual = gpu.execute(left, right)
        var cpu = recipe.execute(Span(inputs))
        check(actual, Span(cpu), String("Contraction CPU ", count))
        check(actual, Span(wanted), String("Contraction NumPy ", count))
        gpu.execute_into(left, right, actual)
        check(actual, Span(wanted), "Reused contraction plan")
        check(left, Span(inputs[0]), "Contraction modified left")
        check(right, Span(inputs[1]), "Contraction modified right")
        count += 1
    print("PASS:", count, "binary GPU contractions against CPU/NumPy")


def main() raises:
    var ctx = DeviceContext()
    assert_true(ctx.api() == "metal" or ctx.api() == "cuda")
    print("DEVICE:", ctx.name(), "API:", ctx.api())
    transformations(ctx, argv()[1])
    contractions(ctx, argv()[2])
