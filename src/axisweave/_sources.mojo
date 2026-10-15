# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Stack-only adapters for contraction kernels."""

from std.collections import ImmSpan, List, Span
from ._buffers import _require_disjoint


trait _Sources(Deinitable, ImplicitlyCopyable, Sized):
    comptime dtype: DType

    def size(self, index: Int) -> Int:
        ...

    def load[
        width: Int
    ](self, index: Int, offset: Int, stride: Int) -> SIMD[Self.dtype, width]:
        ...

    def check_disjoint(
        self, destination: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        ...


@fieldwise_init
struct _OwnedSources[element_dtype: DType, origin: Origin[mut=False]](_Sources):
    comptime dtype = Self.element_dtype
    var buffers: ImmSpan[List[Scalar[Self.dtype]], Self.origin]

    def __len__(self) -> Int:
        return len(self.buffers)

    def size(self, index: Int) -> Int:
        return len(self.buffers[index])

    def load[
        width: Int
    ](self, index: Int, offset: Int, stride: Int) -> SIMD[Self.dtype, width]:
        return (
            self.buffers[index]
            .unsafe_ptr()
            .unsafe_offset(offset)
            .unsafe_strided_load[width=width](stride)
        )

    def check_disjoint(
        self, destination: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        for buffer in self.buffers:
            _require_disjoint(Span(buffer), destination)


@fieldwise_init
struct _BorrowedSources[
    element_dtype: DType,
    origin: Origin[mut=False],
    descriptors: Origin[mut=False],
](_Sources):
    comptime dtype = Self.element_dtype
    var buffers: ImmSpan[
        ImmSpan[Scalar[Self.dtype], Self.origin], Self.descriptors
    ]

    def __len__(self) -> Int:
        return len(self.buffers)

    def size(self, index: Int) -> Int:
        return len(self.buffers[index])

    def load[
        width: Int
    ](self, index: Int, offset: Int, stride: Int) -> SIMD[Self.dtype, width]:
        return (
            self.buffers[index]
            .unsafe_ptr()
            .unsafe_offset(offset)
            .unsafe_strided_load[width=width](stride)
        )

    def check_disjoint(
        self, destination: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        for buffer in self.buffers:
            _require_disjoint(buffer, destination)


@fieldwise_init
struct _PairSources[
    element_dtype: DType,
    left_origin: Origin[mut=False],
    right_origin: Origin[mut=False],
](_Sources):
    comptime dtype = Self.element_dtype
    var left: ImmSpan[Scalar[Self.dtype], Self.left_origin]
    var right: ImmSpan[Scalar[Self.dtype], Self.right_origin]

    def __len__(self) -> Int:
        return 2

    def size(self, index: Int) -> Int:
        return len(self.left) if index == 0 else len(self.right)

    def load[
        width: Int
    ](self, index: Int, offset: Int, stride: Int) -> SIMD[Self.dtype, width]:
        if index == 0:
            return (
                self.left.unsafe_ptr()
                .unsafe_offset(offset)
                .unsafe_strided_load[width=width](stride)
            )
        return (
            self.right.unsafe_ptr()
            .unsafe_offset(offset)
            .unsafe_strided_load[width=width](stride)
        )

    def check_disjoint(
        self, destination: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        _require_disjoint(self.left, destination)
        _require_disjoint(self.right, destination)
