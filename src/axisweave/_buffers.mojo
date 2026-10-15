# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Storage checks shared by static and dynamic execution."""

from std.collections import Array, ImmSpan, Span
from std.sys.info import size_of
from std.algorithm import vectorize
from std.sys import simd_width_of


def _require_disjoint[
    Element: AnyType, //
](source: ImmSpan[Element, _], destination: ImmSpan[Element, _],) raises:
    if len(source) == 0 or len(destination) == 0:
        return
    comptime element_size = size_of[Element]()
    comptime if element_size == 0:
        return
    if max(len(source), len(destination)) > 9223372036854775807 // element_size:
        raise Error("Buffer byte extent overflows Int")
    var start = UInt(source.unsafe_ptr())
    var target = UInt(destination.unsafe_ptr())
    if start <= target:
        if target - start < UInt(len(source) * element_size):
            raise Error("Source and destination buffers overlap")
    elif start - target < UInt(len(destination) * element_size):
        raise Error("Source and destination buffers overlap")


def _copy_block[
    Element: TrivialRegisterPassable, //
](
    source: ImmSpan[Element, _],
    destination: Span[mut=True, Element, _],
    start: Int,
    target: Int,
    count: Int,
    stride: Int = 1,
):
    if count == 0:
        return
    if stride == 1:
        destination[target : target + count].copy_from(
            source[start : start + count]
        )
    else:
        comptime dtypes: Array[DType, 13] = [
            DType.float32,
            DType.float64,
            DType.float16,
            DType.bfloat16,
            DType.int8,
            DType.uint8,
            DType.int16,
            DType.uint16,
            DType.int32,
            DType.uint32,
            DType.int64,
            DType.uint64,
            DType.bool,
        ]
        comptime for dtype_index in range(len(dtypes)):
            comptime dtype = dtypes[dtype_index]
            comptime if Element == Scalar[dtype]:
                # The comptime type equality guarantees identical layout; origins are preserved.
                _copy_scalar_block(
                    source.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](),
                    destination.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](),
                    start,
                    target,
                    count,
                    stride,
                )
                return
        for i in range(count):
            destination[target + i] = source[start + i * stride]


def _copy_scalar_block[
    dtype: DType, //
](
    source: ImmPointer[Scalar[dtype], _],
    destination: MutPointer[Scalar[dtype], _],
    start: Int,
    target: Int,
    count: Int,
    stride: Int,
):
    if count == 0:
        return
    var sp = source.unsafe_offset(start)
    var dp = destination.unsafe_offset(target)

    def body[width: Int](i: Int) {var sp, var dp, var stride}:
        var values = sp.unsafe_offset(i * stride).unsafe_strided_load[
            width=width
        ](stride)
        dp.unsafe_store(i, values)

    vectorize[simd_width_of[dtype]()](count, body)
