# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Borrowed buffer adapters for element-preserving packing."""

from std.collections import ImmSpan, List, Span
from ._buffers import _require_disjoint


trait _PackingBuffers(Deinitable, ImplicitlyCopyable, Sized):
    comptime Element: TrivialRegisterPassable & Defaultable

    def size(self, index: Int) -> Int:
        ...

    def check_disjoint(
        self, destination: Span[mut=True, Self.Element, _]
    ) raises:
        ...

    def copy_to(
        self,
        index: Int,
        destination: Span[mut=True, Self.Element, _],
        start: Int,
        target: Int,
        count: Int,
    ):
        ...


@fieldwise_init
struct _OwnedBuffers[
    T: TrivialRegisterPassable & Defaultable, origin: Origin[mut=False]
](_PackingBuffers):
    comptime Element = Self.T
    var buffers: ImmSpan[List[Self.T], Self.origin]

    def __len__(self) -> Int:
        return len(self.buffers)

    def size(self, index: Int) -> Int:
        return len(self.buffers[index])

    def check_disjoint(self, destination: Span[mut=True, Self.T, _]) raises:
        for buffer in self.buffers:
            _require_disjoint(Span(buffer), destination)

    def copy_to(
        self,
        index: Int,
        destination: Span[mut=True, Self.T, _],
        start: Int,
        target: Int,
        count: Int,
    ):
        destination[target : target + count].copy_from(
            Span(self.buffers[index])[start : start + count]
        )


@fieldwise_init
struct _BorrowedBuffers[
    T: TrivialRegisterPassable & Defaultable,
    origin: Origin[mut=False],
    descriptors: Origin[mut=False],
](_PackingBuffers):
    comptime Element = Self.T
    var buffers: ImmSpan[ImmSpan[Self.T, Self.origin], Self.descriptors]

    def __len__(self) -> Int:
        return len(self.buffers)

    def size(self, index: Int) -> Int:
        return len(self.buffers[index])

    def check_disjoint(self, destination: Span[mut=True, Self.T, _]) raises:
        for buffer in self.buffers:
            _require_disjoint(buffer, destination)

    def copy_to(
        self,
        index: Int,
        destination: Span[mut=True, Self.T, _],
        start: Int,
        target: Int,
        count: Int,
    ):
        destination[target : target + count].copy_from(
            self.buffers[index][start : start + count]
        )
