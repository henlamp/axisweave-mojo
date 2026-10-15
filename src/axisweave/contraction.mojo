# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Named-axis tensor contractions with reusable strides and SIMD output rows."""

from std.algorithm import vectorize
from std.collections import Dict, ImmSpan, List, Span
from std.sys import simd_width_of
from .pattern import _check_name, _multiply, _product, _numeric
from ._sources import _Sources, _OwnedSources, _BorrowedSources


def _axes(text: String) raises -> List[String]:
    var result = List[String]()
    for token in text.split():
        var name = String(token)
        _check_name(name)
        if _numeric(name):
            raise Error("Contraction patterns require explicit named axes")
        result.append(name^)
    return result^


def _expand_axes(
    axes: Span[String, _], ellipsis_rank: Int, maximum: Int
) -> List[String]:
    var result = List[String]()
    for name in axes:
        if name == "...":
            for axis in range(maximum - ellipsis_rank, maximum):
                result.append(String("_ellipsis_", axis))
        else:
            result.append(name.copy())
    return result^


def _index(flat: Int, sizes: Span[Int, _], strides: Span[Int, _]) -> Int:
    var remaining = flat
    var offset = 0
    for axis in range(len(sizes) - 1, -1, -1):
        offset += (remaining % sizes[axis]) * strides[axis]
        remaining //= sizes[axis]
    return offset


struct EinsumRecipe(Copyable):
    """Own a named-axis contraction plan, including diagonal and broadcast strides.
    """

    var _shapes: List[List[Int]]
    var _sizes: List[Int]
    var _strides: List[List[Int]]
    var _output_rank: Int

    def __init__(
        out self,
        var shapes: List[List[Int]],
        var sizes: List[Int],
        var strides: List[List[Int]],
        output_rank: Int,
    ) raises:
        self._shapes = shapes^
        self._sizes = sizes^
        self._strides = strides^
        self._output_rank = output_rank
        _ = self._extents()

    def output_shape(self) -> Span[Int, origin_of(self._sizes)]:
        """Borrow the output dimensions in pattern order."""
        return Span(self._sizes)[: self._output_rank]

    def _extents(self) raises -> Tuple[Int, Int]:
        if (
            self._output_rank < 0
            or self._output_rank > len(self._sizes)
            or len(self._shapes) == 0
            or len(self._shapes) != len(self._strides)
        ):
            raise Error("Malformed contraction metadata")
        var sizes = Span(self._sizes)
        var output = _product(sizes[: self._output_rank])
        var reduction = _product(sizes[self._output_rank :])
        var total = _multiply(output, reduction)
        for i in range(len(self._shapes)):
            var storage = _product(Span(self._shapes[i]))
            var strides = Span(self._strides[i])
            if len(strides) != len(sizes):
                raise Error("Contraction stride rank disagrees")
            var last = 0
            for axis in range(len(sizes)):
                if strides[axis] < 0:
                    raise Error("Negative contraction stride")
                if total > 0:
                    var delta = _multiply(sizes[axis] - 1, strides[axis])
                    if storage == 0 or delta > storage - 1 - last:
                        raise Error("Contraction stride exceeds source storage")
                    last += delta
        return output, reduction

    def execute_into[
        dtype: DType, //
    ](
        self,
        sources: Span[List[Scalar[dtype]], _],
        destination: Span[mut=True, Scalar[dtype], _],
    ) raises:
        """Contract owned buffers after checking extents and destination overlap.
        """
        self._execute_into(_OwnedSources(sources.as_imm()), destination)

    def execute_into[
        dtype: DType, origin: Origin[mut=False], //
    ](
        self,
        sources: ImmSpan[ImmSpan[Scalar[dtype], origin], _],
        destination: Span[mut=True, Scalar[dtype], _],
    ) raises:
        """Contract borrowed buffers after checking extents and destination overlap.
        """
        self._execute_into(_BorrowedSources(sources), destination)

    def execute[
        dtype: DType, //
    ](self, sources: Span[List[Scalar[dtype]], _]) raises -> List[
        Scalar[dtype]
    ]:
        """Contract owned buffers after checking extents and destination overlap.
        """
        return self._execute(_OwnedSources(sources.as_imm()))

    def execute[
        dtype: DType, origin: Origin[mut=False], //
    ](self, sources: ImmSpan[ImmSpan[Scalar[dtype], origin], _]) raises -> List[
        Scalar[dtype]
    ]:
        """Contract borrowed buffers after checking extents and destination overlap.
        """
        return self._execute(_BorrowedSources(sources))

    def _validate[
        Sources: _Sources, //
    ](self, sources: Sources) raises -> Tuple[Int, Int]:
        comptime assert (
            Sources.dtype == .float32 or Sources.dtype == .float64
        ), "Contractions require float32 or float64"
        var extents = self._extents()
        if len(sources) != len(self._shapes):
            raise Error("Contraction operand count disagrees")
        for i in range(len(sources)):
            if sources.size(i) != _product(Span(self._shapes[i])):
                raise Error("Contraction source length disagrees")
        return extents

    def _execute_into[
        Sources: _Sources, //
    ](
        self,
        sources: Sources,
        destination: Span[mut=True, Scalar[Sources.dtype], _],
    ) raises:
        """Validate every operand and reject overlap before writing contraction output.
        """
        var extents = self._validate(sources)
        if len(destination) != extents[0]:
            raise Error("Contraction destination length disagrees")
        sources.check_disjoint(destination)
        self._execute_validated(sources, destination, extents[0], extents[1])

    def _execute[
        Sources: _Sources, //
    ](self, sources: Sources) raises -> List[Scalar[Sources.dtype]]:
        """Validate operands before allocating the contraction result."""
        var extents = self._validate(sources)
        var result = List[Scalar[Sources.dtype]](length=extents[0], fill=0)
        self._execute_validated(sources, Span(result), extents[0], extents[1])
        return result^

    def _execute_validated[
        Sources: _Sources, //
    ](
        self,
        sources: Sources,
        destination: Span[mut=True, Scalar[Sources.dtype], _],
        output_count: Int,
        reduction_count: Int,
    ):
        if output_count == 0:
            return
        if reduction_count == 0:
            for i in range(output_count):
                destination[i] = 0
            return
        var sizes = Span(self._sizes)
        var strides = Span(self._strides)
        var last_axis = self._output_rank - 1
        var output_rank = self._output_rank
        var row_size = 1 if last_axis < 0 else sizes[last_axis]
        var reduction_rank = len(sizes) - output_rank
        var bases = List[Int](
            length=len(sources) if reduction_count > 1 else 0, fill=0
        )
        var coordinates = List[Int](
            length=reduction_rank if reduction_count > 1
            and reduction_rank > 1 else 0,
            fill=0,
        )
        var offsets = Span(bases)
        var reduced = Span(coordinates)
        var dp = destination.unsafe_ptr()
        for row in range(output_count // row_size):
            var row_start = row * row_size

            def body[
                width: Int
            ](i: Int) {
                var sizes,
                var strides,
                var sources,
                var dp,
                var reduction_count,
                var row_start,
                var last_axis,
                var output_rank,
                var reduction_rank,
                var offsets,
                var reduced,
            }:
                for axis in range(len(reduced)):
                    reduced[axis] = 0
                if reduction_count > 1:
                    for part in range(len(sources)):
                        offsets[part] = _index(
                            row_start + i,
                            sizes[:output_rank],
                            Span(strides[part])[:output_rank],
                        )
                var result = SIMD[Sources.dtype, width](0)
                for r in range(reduction_count):
                    var product = SIMD[Sources.dtype, width](1)
                    for part in range(len(sources)):
                        var step = (
                            0 if last_axis < 0 else strides[part][last_axis]
                        )
                        var base = (
                            _index(
                                row_start + i,
                                sizes[:output_rank],
                                Span(strides[part])[:output_rank],
                            ) if reduction_count
                            == 1 else offsets[part]
                        )
                        if reduction_count > 1 and reduction_rank == 1:
                            base += r * strides[part][output_rank]
                        product *= sources.load[width](part, base, step)
                    result += product
                    if len(reduced) == 0:
                        continue
                    if r + 1 == reduction_count:
                        break
                    # Advance mixed-radix coordinates without division per operand.
                    for axis in range(len(reduced) - 1, -1, -1):
                        var global_axis = output_rank + axis
                        if reduced[axis] + 1 < sizes[global_axis]:
                            reduced[axis] += 1
                            for part in range(len(sources)):
                                offsets[part] += strides[part][global_axis]
                            break
                        for part in range(len(sources)):
                            offsets[part] -= (
                                reduced[axis] * strides[part][global_axis]
                            )
                        reduced[axis] = 0
                dp.unsafe_store(row_start + i, result)

            vectorize[simd_width_of[Sources.dtype]()](row_size, body)


def einsum_plan(
    shapes: Span[List[Int], _], pattern: String
) raises -> EinsumRecipe:
    """Plan products and sums over named axes, with an explicit output after ->.
    """
    var sides = pattern.split("->")
    if len(sides) != 2 or len(shapes) == 0:
        raise Error("A contraction requires operands and one explicit arrow")
    var terms = String(sides[0]).split(",")
    if len(terms) != len(shapes):
        raise Error("Contraction pattern operand count disagrees")
    var raw = List[List[String]]()
    var ellipsis_ranks = List[Int]()
    var maximum = 0
    for part in range(len(shapes)):
        var axes = _axes(String(terms[part]))
        _ = _product(Span(shapes[part]))
        var ellipses = 0
        for name in axes:
            ellipses += 1 if name == "..." else 0
        if ellipses > 1:
            raise Error("A contraction operand permits only one ellipsis")
        var rank = len(shapes[part]) - len(axes) + ellipses
        if rank < 0 or (ellipses == 0 and rank != 0):
            raise Error("Contraction input rank disagrees")
        ellipsis_ranks.append(rank)
        maximum = max(maximum, rank)
        raw.append(axes^)
    var names = List[String]()
    var lengths = Dict[String, Int]()
    var inputs = List[List[String]]()
    for part in range(len(shapes)):
        var axes = _expand_axes(Span(raw[part]), ellipsis_ranks[part], maximum)
        var shape = Span(shapes[part])
        var local = Dict[String, Int]()
        for axis in range(len(axes)):
            var name = axes[axis].copy()
            var size = shape[axis]
            if name in local and local[name] != size:
                raise Error("Diagonal axis dimensions disagree")
            local[name] = size
            if name not in lengths:
                lengths[name] = size
                names.append(name.copy())
            elif lengths[name] == 1:
                lengths[name] = size
            elif size != 1 and size != lengths[name]:
                raise Error("Shared contraction axis dimensions disagree")
        inputs.append(axes^)
    var raw_output = _axes(String(sides[1]))
    var ellipses = 0
    for name in raw_output:
        ellipses += 1 if name == "..." else 0
    if ellipses > 1 or (maximum > 0 and ellipses == 0):
        raise Error(
            "Broadcast contraction dimensions require one output ellipsis"
        )
    var output = _expand_axes(Span(raw_output), maximum, maximum)
    var ids = Dict[String, Int]()
    var ordered = List[String]()
    for name in output:
        if name not in lengths or name in ids:
            raise Error("Unknown or duplicate contraction output axis")
        ids[name] = len(ordered)
        ordered.append(name.copy())
    for name in names:
        if name not in ids:
            ids[name] = len(ordered)
            ordered.append(name.copy())
    var sizes = List[Int]()
    for name in ordered:
        sizes.append(lengths[name])
    var strides = List[List[Int]]()
    for part in range(len(shapes)):
        var steps = List[Int](length=len(sizes), fill=0)
        if _product(Span(shapes[part])) > 0:
            var stride = 1
            for axis in range(len(shapes[part]) - 1, -1, -1):
                var size = shapes[part][axis]
                if size != 1:
                    var target = ids[inputs[part][axis]]
                    if stride > 9223372036854775807 - steps[target]:
                        raise Error("Diagonal stride overflows Int")
                    steps[target] += stride
                stride = _multiply(stride, size)
        strides.append(steps^)
    return EinsumRecipe(List(shapes), sizes^, strides^, len(output))


def einsum[
    dtype: DType, //
](
    sources: Span[List[Scalar[dtype]], _],
    shapes: Span[List[Int], _],
    pattern: String,
) raises -> List[Scalar[dtype]]:
    """Evaluate an explicit named-axis contraction on contiguous float buffers.
    """
    return einsum_plan(shapes, pattern).execute(sources)


def einsum[
    dtype: DType, origin: Origin[mut=False], //
](
    sources: ImmSpan[ImmSpan[Scalar[dtype], origin], _],
    shapes: Span[List[Int], _],
    pattern: String,
) raises -> List[Scalar[dtype]]:
    """Contract borrowed contiguous buffers without copying operands."""
    return einsum_plan(shapes, pattern).execute(sources)
