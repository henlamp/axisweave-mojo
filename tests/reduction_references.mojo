# SPDX-License-Identifier: MIT
from std.collections import List, Span
from std.math import sqrt
from std.testing import assert_equal, assert_true
from axisweave import ReductionValues


def variance[
    d: DType, s: Origin[mut=False], p: Origin[mut=False]
](row: ReductionValues[d, s, p]) raises -> Scalar[d]:
    var mean = Float64(0)
    var moment = Float64(0)
    for i in range(len(row)):
        var value = Float64(row[i])
        var delta = value - mean
        mean += delta / Float64(i + 1)
        moment += delta * (value - mean)
    return Scalar[d](moment / Float64(len(row)))


def nanmean[
    d: DType, s: Origin[mut=False], p: Origin[mut=False]
](row: ReductionValues[d, s, p]) raises -> Scalar[d]:
    var total = Float64(0)
    var count = 0
    for i in range(len(row)):
        var value = Float64(row[i])
        if value == value:
            total += value
            count += 1
    return Scalar[d](total / Float64(count))


def rms[
    d: DType, s: Origin[mut=False], p: Origin[mut=False]
](row: ReductionValues[d, s, p]) raises -> Scalar[d]:
    var total = Float64(0)
    for i in range(len(row)):
        var value = Float64(row[i])
        total += value * value
    return Scalar[d](sqrt(total / Float64(len(row))))


def check[
    d: DType, //
](actual: Span[Scalar[d], _], wanted: Span[Float64, _]) raises:
    assert_equal(len(actual), len(wanted))
    for i in range(len(wanted)):
        var value = Float64(actual[i])
        if wanted[i] != wanted[i]:
            assert_true(value != value)
        else:
            assert_true(abs(value - wanted[i]) <= 2e-5 * (1 + abs(wanted[i])))
