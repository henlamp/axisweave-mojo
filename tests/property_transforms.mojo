"""Run a runtime corpus of independently computed transforms and rejections."""
from std.collections import Dict, List, Span
from std.math import isfinite
from std.sys import argv
from std.testing import assert_equal, assert_true
from axisweave import plan, execute_into, view, Operation, Reduction


def integers(text: String) raises -> List[Int]:
    var result = List[Int]()
    if text.byte_length() > 0:
        for part in text.split(","):
            result.append(Int(String(part)))
    return result^


def numbers(text: String) raises -> List[Float64]:
    var result = List[Float64]()
    if text.byte_length() > 0:
        for part in text.split(","):
            result.append(Float64(String(part)))
    return result^


def main() raises:
    var text: String
    with open(argv()[1], "r") as input:
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
        var output = List[Float64](length=len(expected), fill=987654)
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
            execute_into(recipe, Span(source), Span(output), reduction)
            if fields[0] == "valid":
                assert_equal(List(recipe.output_shape()), expected_shape)
                if operation != .reduce:
                    var borrowed = view(recipe, Span(source))
                    assert_equal(borrowed.materialize(), output)
        except:
            failed = True
        if fields[0] == "invalid":
            assert_true(failed, String("Expected rejection: ", fields[11]))
            for value in output:
                assert_equal(value, Float64(987654))
            rejected += 1
        else:
            assert_true(
                not failed, String("Unexpected rejection: ", fields[11])
            )
            for i in range(len(expected)):
                assert_true(isfinite(output[i]))
                assert_true(
                    abs(output[i] - expected[i])
                    <= 1e-11 * (1 + abs(expected[i])),
                    String("Value mismatch: ", fields[11]),
                )
            passed += 1
    print(
        "PASS:",
        passed,
        "generated transforms and",
        rejected,
        "rejections before writes",
    )
