# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Mojo 1.1 CPU kernels with explicit captures and typed specialization.

Unchecked pointer access is internal and follows the public span-length checks.
Rows, affine extents and vector tails are derived from a validated recipe.
"""

from std.algorithm import vectorize
from std.collections import ImmSpan, Span
from std.sys import simd_width_of
from .kinds import Reduction
from .pattern import _RecipeData
from ._buffers import _copy_block, _copy_scalar_block


@inline(.always)
def combine[
    dtype: DType, //, kind: Reduction
](result: Scalar[dtype], value: Scalar[dtype], first: Bool) -> Scalar[dtype]:
    comptime if kind == .sum or kind == .mean:
        return result + value
    elif kind == .any:
        return result | value
    elif kind == .all:
        return result & value
    elif kind == .prod:
        return result * value
    else:
        if first or value != value:
            return value
        comptime if kind == .min:
            if value < result:
                return value
        else:
            if value > result:
                return value
        comptime if dtype == .float32 or dtype == .float64:
            if value == 0 and result == 0:
                comptime if kind == .min:
                    if Scalar[dtype](1) / value < 0:
                        return value
                else:
                    if Scalar[dtype](1) / value > 0:
                        return value
        return result


def affine_stride(recipe: _RecipeData, axes: Span[Int, _]) -> Int:
    """Return the flattened stride, or -1 for a non-affine index sequence."""
    var step = 0
    var expected = 0
    var found = False
    for i in range(len(axes) - 1, -1, -1):
        var axis = axes[i]
        if recipe.axis_sizes[axis] <= 1:
            continue
        if not found:
            step = recipe.source_strides[axis]
            expected = step
            found = True
        if recipe.source_strides[axis] != expected:
            return -1
        expected *= recipe.axis_sizes[axis]
    return step


@inline(.always)
def _copy_extent(recipe: _RecipeData) -> Tuple[Int, Int]:
    if len(recipe.output_axes) == 0:
        return 1, 0
    var axis = recipe.output_axes[len(recipe.output_axes) - 1]
    return recipe.axis_sizes[axis], recipe.source_strides[axis]


def copy_rows[
    Element: TrivialRegisterPassable, //
](
    recipe: _RecipeData,
    source: ImmSpan[Element, _],
    destination: Span[mut=True, Element, _],
):
    if recipe.output_size == 0:
        return
    var extent = _copy_extent(recipe)
    var inner = extent[0]
    var stride = extent[1]
    for row in range(recipe.output_size // inner):
        var start = row * inner
        _copy_block(
            source,
            destination,
            recipe._source_index(start),
            start,
            inner,
            stride,
        )


def copy_block[
    dtype: DType, //
](
    source: ImmSpan[Scalar[dtype], _],
    destination: Span[mut=True, Scalar[dtype], _],
    start: Int,
    target: Int,
    count: Int,
    stride: Int = 1,
):
    _copy_scalar_block(
        source.unsafe_ptr(),
        destination.unsafe_ptr(),
        start,
        target,
        count,
        stride,
    )


def scalar_reduce[
    dtype: DType, //, kind: Reduction
](
    recipe: _RecipeData,
    source: ImmSpan[Scalar[dtype], _],
    destination: Span[mut=True, Scalar[dtype], _],
):
    for out_index in range(recipe.output_size):
        var result = Scalar[dtype](1 if kind == .prod or kind == .all else 0)
        for r in range(recipe.reduction_size):
            result = combine[kind](
                result, source[recipe._source_index(out_index, r)], r == 0
            )
        comptime if kind == .mean:
            result /= Scalar[dtype](recipe.reduction_size)
        destination[out_index] = result


def execute_reduction[
    dtype: DType, //, kind: Reduction
](
    recipe: _RecipeData,
    source: ImmSpan[Scalar[dtype], _],
    destination: Span[mut=True, Scalar[dtype], _],
):
    if recipe.output_size == 0:
        return
    comptime if kind == .sum or kind == .mean:
        var output_stride = affine_stride(recipe, Span(recipe.output_axes))
        var reduction_stride = affine_stride(recipe, Span(recipe.reduced_axes))
        if (
            recipe.reduction_size > 0
            and output_stride >= 0
            and reduction_stride >= 0
        ):
            sum_rows[kind](
                source,
                destination,
                recipe.reduction_size,
                output_stride,
                reduction_stride,
                recipe.input_offset,
            )
            return
    scalar_reduce[kind](recipe, source, destination)


def sum_rows[
    dtype: DType, //, kind: Reduction
](
    source: ImmSpan[Scalar[dtype], _],
    destination: Span[mut=True, Scalar[dtype], _],
    count: Int,
    output_stride: Int,
    reduction_stride: Int,
    offset: Int = 0,
):
    comptime assert kind == .sum or kind == .mean
    var sp = source.unsafe_ptr().unsafe_offset(offset)
    var dp = destination.unsafe_ptr()

    def body[
        width: Int
    ](out_index: Int) {
        var sp,
        var dp,
        var count,
        var output_stride,
        var reduction_stride,
    }:
        var result = SIMD[dtype, width](0)
        for r in range(count):
            result += sp.unsafe_offset(
                out_index * output_stride + r * reduction_stride
            ).unsafe_strided_load[width=width](output_stride)
        comptime if kind == .mean:
            result /= SIMD[dtype, width](Scalar[dtype](count))
        dp.unsafe_store(out_index, result)

    vectorize[simd_width_of[dtype]()](len(destination), body)


def bool_rows[
    Index: def(Int, Int) -> Int, //
](
    source: ImmSpan[Bool, _],
    destination: Span[mut=True, Bool, _],
    count: Int,
    reduction: Reduction,
    index: Index,
):
    for row in range(len(destination)):
        var result = reduction == .all
        for r in range(count):
            var value = source[index(row, r)]
            result = (result or value) if reduction == .any else (
                result and value
            )
        destination[row] = result
