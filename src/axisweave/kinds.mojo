# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Typed tensor operations and reductions."""


@fieldwise_init
struct Operation(Equatable, ImplicitlyCopyable):
    """Typed rearrange, repeat or reduce operation for pattern planning."""

    var _code: Int
    comptime rearrange = Self(0)
    comptime repeat = Self(1)
    comptime reduce = Self(2)

    def valid(self) -> Bool:
        """Whether the stored value denotes a supported operation."""
        return 0 <= self._code and self._code <= 2


@fieldwise_init
struct Reduction(Equatable, ImplicitlyCopyable):
    """Typed numeric and logical reductions."""

    var _code: Int
    comptime sum = Self(0)
    comptime mean = Self(1)
    comptime min = Self(2)
    comptime max = Self(3)
    comptime prod = Self(4)
    comptime any = Self(5)
    comptime all = Self(6)

    def valid(self) -> Bool:
        """Whether the stored value denotes a supported reduction."""
        return 0 <= self._code and self._code <= 6

    def supports[dtype: DType](self) -> Bool:
        """Whether this reduction accepts the input/output scalar DType."""
        if not self.valid():
            return False
        comptime if dtype == .bool:
            return self == .any or self == .all
        elif dtype == .float32 or dtype == .float64:
            return self != .any and self != .all
        elif dtype == .int32 or dtype == .int64 or dtype == .uint32 or dtype == .uint64:
            return self == .sum or self == .prod or self == .min or self == .max
        else:
            return False
