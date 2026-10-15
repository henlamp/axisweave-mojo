# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Portable Float32 kernels; host-validated, compact Int64 metadata."""

from max.gpu import global_idx
from std.memory import Pointer
from .kinds import Reduction
from ._kernels import combine


@inline(.always)
def _offset[
    index_dtype: DType
](
    flat: Int,
    metadata: Pointer[Int64, MutAnyOrigin],
    start: Int,
    rank: Int,
    step: Int = 2,
    stride_column: Int = 1,
) -> Int:
    var result = Scalar[index_dtype](0)
    var remaining = Scalar[index_dtype](flat)
    for axis in range(rank - 1, -1, -1):
        var at = start + axis * step
        var size = Scalar[index_dtype](metadata[unsafe_offset=at])
        result += (remaining % size) * Scalar[index_dtype](
            metadata[unsafe_offset=at + stride_column]
        )
        remaining //= size
    return Int(result)


def _transform[
    kind: Reduction, reducing: Bool, narrow: Bool
](
    source: Pointer[Float32, MutAnyOrigin],
    destination: Pointer[Float32, MutAnyOrigin],
    metadata: Pointer[Int64, MutAnyOrigin],
    count: Int64,
):
    comptime index_dtype = DType.int32 if narrow else DType.int64
    var row = global_idx.x
    if row >= Int(count):
        return
    var output_rank = Int(metadata[unsafe_offset=0])
    var reduction_rank = Int(metadata[unsafe_offset=1])
    var total = Int(metadata[unsafe_offset=3])
    comptime if reducing:
        if total == 0:
            var empty = Float32(1 if kind == .prod else 0)
            comptime if kind == .mean:
                empty /= Float32(total)
            destination[unsafe_offset=row] = empty
            return
    var base = Int(metadata[unsafe_offset=2]) + _offset[index_dtype](
        row, metadata, 4, output_rank
    )
    comptime if not reducing:
        destination[unsafe_offset=row] = source[unsafe_offset=base]
    else:
        var result = Float32(1 if kind == .prod else 0)
        for r in range(total):
            var index = base + _offset[index_dtype](
                r, metadata, 4 + output_rank * 2, reduction_rank
            )
            result = combine[kind](result, source[unsafe_offset=index], r == 0)
        comptime if kind == .mean:
            result /= Float32(total)
        destination[unsafe_offset=row] = result


def _contract[
    narrow: Bool
](
    left: Pointer[Float32, MutAnyOrigin],
    right: Pointer[Float32, MutAnyOrigin],
    destination: Pointer[Float32, MutAnyOrigin],
    metadata: Pointer[Int64, MutAnyOrigin],
    count: Int64,
):
    comptime index_dtype = DType.int32 if narrow else DType.int64
    var row = global_idx.x
    if row >= Int(count):
        return
    var output_rank = Int(metadata[unsafe_offset=0])
    var reduction_rank = Int(metadata[unsafe_offset=1])
    var reduction_count = Int(metadata[unsafe_offset=2])
    if reduction_count == 0:
        destination[unsafe_offset=row] = 0
        return
    var left_base = _offset[index_dtype](row, metadata, 3, output_rank, 3, 1)
    var right_base = _offset[index_dtype](row, metadata, 3, output_rank, 3, 2)
    var start = 3 + output_rank * 3
    var result = Float32(0)
    for r in range(reduction_count):
        var li = left_base + _offset[index_dtype](
            r, metadata, start, reduction_rank, 3, 1
        )
        var ri = right_base + _offset[index_dtype](
            r, metadata, start, reduction_rank, 3, 2
        )
        result += left[unsafe_offset=li] * right[unsafe_offset=ri]
    destination[unsafe_offset=row] = result
