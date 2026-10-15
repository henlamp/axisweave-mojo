# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Reusable pack/unpack plans for contiguous tensor buffers."""

from std.collections import Dict, ImmSpan, List, Span
from .pattern import (
    _check_name,
    _multiply,
    _product,
    _numeric,
    _identity_recipe,
)
from ._buffers import _require_disjoint
from ._packing_buffers import _PackingBuffers, _OwnedBuffers, _BorrowedBuffers
from ._buffers import _copy_block
from .view import TensorView


def _packing_axes(pattern: String) raises -> Tuple[Int, Int]:
    var seen = Dict[String, Bool]()
    var star = -1
    var tokens = pattern.split()
    for i in range(len(tokens)):
        var name = String(tokens[i])
        if name in seen:
            raise Error("Duplicate packing axis")
        seen[name] = True
        if name == "*":
            star = i
        else:
            _check_name(name)
            if _numeric(name) or name == "...":
                raise Error("Packing patterns use named axes and one *")
    if star < 0:
        raise Error("A packing pattern requires one *")
    return star, len(tokens) - star - 1


@fieldwise_init
struct _PackLayout(ImplicitlyCopyable):
    var outer: Int
    var inner: Int
    var packed: Int
    var size: Int


struct PackRecipe(Copyable):
    """Own packing shapes; pack and unpack any buffers with those extents."""

    var _inputs: List[List[Int]]
    var _shape: List[Int]
    var _before: Int
    var _after: Int

    def __init__(
        out self,
        var inputs: List[List[Int]],
        var shape: List[Int],
        before: Int,
        after: Int,
    ) raises:
        self._inputs = inputs^
        self._shape = shape^
        self._before = before
        self._after = after
        _ = self._layout()

    def output_shape(self) -> Span[Int, origin_of(self._shape)]:
        """Borrow the packed tensor dimensions."""
        return Span(self._shape)

    def input_shape(
        self, index: Int
    ) raises -> Span[Int, origin_of(self._inputs[index])]:
        """Borrow one input's immutable dimensions for the plan lifetime."""
        if index < 0 or index >= len(self._inputs):
            raise Error("Packing input index out of range")
        return Span(self._inputs[index])

    def packed_shapes(self) raises -> List[List[Int]]:
        """Copy the dimensions matched by * in each input."""
        _ = self._layout()
        var result = List[List[Int]]()
        for shape in self._inputs:
            result.append(
                List(Span(shape)[self._before : len(shape) - self._after])
            )
        return result^

    def _layout(self) raises -> _PackLayout:
        if (
            self._before < 0
            or self._before > len(self._shape)
            or self._after > len(self._shape)
            or self._after < 0
            or len(self._inputs) == 0
            or len(self._shape) != self._before + 1 + self._after
        ):
            raise Error("Malformed packing metadata")
        var output = Span(self._shape)
        var size = _product(output)
        var outer = _product(output[: self._before]) if size > 0 else 0
        var inner = _product(output[self._before + 1 :]) if size > 0 else 0
        var packed = 0
        for shape in self._inputs:
            if len(shape) < self._before + self._after:
                raise Error("Input rank is smaller than the packing pattern")
            for i in range(self._before):
                if shape[i] != output[i]:
                    raise Error("Packing prefix dimensions disagree")
            for i in range(self._after):
                if (
                    shape[len(shape) - self._after + i]
                    != output[self._before + 1 + i]
                ):
                    raise Error("Packing suffix dimensions disagree")
            var part = _product(
                Span(shape)[self._before : len(shape) - self._after]
            )
            if part > 9223372036854775807 - packed:
                raise Error("Packed axis length overflows Int")
            packed += part
        if output[self._before] != packed:
            raise Error("Packed axis length disagrees")
        return _PackLayout(outer, inner, packed, size)

    def pack_into[
        Element: TrivialRegisterPassable & Defaultable, //
    ](
        self,
        sources: Span[List[Element], _],
        destination: Span[mut=True, Element, _],
    ) raises:
        """Pack owned buffers after checking extents and destination overlap."""
        self._pack_into(_OwnedBuffers(sources.as_imm()), destination)

    def pack_into[
        Element: TrivialRegisterPassable & Defaultable,
        origin: Origin[mut=False],
        //,
    ](
        self,
        sources: ImmSpan[ImmSpan[Element, origin], _],
        destination: Span[mut=True, Element, _],
    ) raises:
        """Pack borrowed buffers after checking extents and destination overlap.
        """
        self._pack_into(_BorrowedBuffers(sources), destination)

    def pack[
        Element: TrivialRegisterPassable & Defaultable, //
    ](self, sources: Span[List[Element], _]) raises -> List[Element]:
        """Pack owned buffers after checking extents and destination overlap."""
        return self._pack(_OwnedBuffers(sources.as_imm()))

    def pack[
        Element: TrivialRegisterPassable & Defaultable,
        origin: Origin[mut=False],
        //,
    ](self, sources: ImmSpan[ImmSpan[Element, origin], _]) raises -> List[
        Element
    ]:
        """Pack borrowed buffers after checking extents and destination overlap.
        """
        return self._pack(_BorrowedBuffers(sources))

    def _validate_sources[
        Sources: _PackingBuffers, //
    ](self, sources: Sources) raises -> _PackLayout:
        var layout = self._layout()
        if len(sources) != len(self._inputs):
            raise Error("Packing buffer count disagrees")
        for i in range(len(sources)):
            var dims = Span(self._inputs[i])
            var part = _product(dims[self._before : len(dims) - self._after])
            if sources.size(i) != layout.outer * part * layout.inner:
                raise Error("Packing source length disagrees")
        return layout

    def _pack_into[
        Sources: _PackingBuffers, //
    ](
        self,
        sources: Sources,
        destination: Span[mut=True, Sources.Element, _],
    ) raises:
        """Pack into a disjoint exact-size buffer after validating all inputs.
        """
        var layout = self._validate_sources(sources)
        if len(destination) != layout.size:
            raise Error("Packed destination length disagrees")
        sources.check_disjoint(destination)
        self._pack_validated(sources, destination, layout.outer, layout.inner)

    def _pack_validated[
        Sources: _PackingBuffers, //
    ](
        self,
        sources: Sources,
        destination: Span[mut=True, Sources.Element, _],
        outer: Int,
        inner: Int,
    ):
        if len(destination) == 0:
            return
        var row_size = self._shape[self._before] * inner
        var offset = 0
        for part in range(len(sources)):
            var chunk = sources.size(part) // outer
            for row in range(outer):
                sources.copy_to(
                    part,
                    destination,
                    row * chunk,
                    row * row_size + offset,
                    chunk,
                )
            offset += chunk

    def _pack[
        Sources: _PackingBuffers, //
    ](self, sources: Sources) raises -> List[Sources.Element]:
        """Validate every input before allocating and packing output."""
        var layout = self._validate_sources(sources)
        var result = List[Sources.Element](
            length=layout.size, fill=Sources.Element()
        )
        self._pack_validated(sources, Span(result), layout.outer, layout.inner)
        return result^

    def unpack_views[
        Element: TrivialRegisterPassable & Defaultable,
        origin: Origin[mut=False],
        //,
    ](self, source: Span[Element, origin]) raises -> List[
        TensorView[Element, origin]
    ]:
        """Borrow unpacked tensors without copying values; views retain source lifetimes.
        """
        var layout = self._layout()
        if len(source) != layout.size:
            raise Error("Packed source length disagrees")
        var inner = layout.inner
        var output = List[TensorView[Element, origin]]()
        var offset = 0
        for dims in self._inputs:
            var strides = List[Int](length=len(dims), fill=0)
            var part = _product(
                Span(dims)[self._before : len(dims) - self._after]
            )
            if layout.size > 0 and part > 0:
                var stride = 1
                for i in range(len(dims) - 1, self._before - 1, -1):
                    strides[i] = stride
                    stride = _multiply(stride, dims[i])
                stride = _multiply(layout.packed, inner)
                for i in range(self._before - 1, -1, -1):
                    strides[i] = stride
                    stride = _multiply(stride, dims[i])
            var recipe = _identity_recipe(Span(dims), strides^, offset)
            output.append(TensorView(source, recipe^))
            if len(source) > 0:
                offset += part * inner
        return output^

    def unpack_into[
        Element: TrivialRegisterPassable & Defaultable,
        origin: Origin[mut=True],
        //,
    ](
        self,
        source: ImmSpan[Element, _],
        destinations: ImmSpan[Span[mut=True, Element, origin], _],
    ) raises:
        """Unpack into exact-size, pairwise disjoint caller buffers; validate before writes.
        """
        var layout = self._layout()
        if len(source) != layout.size or len(destinations) != len(self._inputs):
            raise Error(
                "Unpacking source length or destination count disagrees"
            )
        for i in range(len(destinations)):
            var dims = Span(self._inputs[i])
            var part = _product(dims[self._before : len(dims) - self._after])
            if len(destinations[i]) != layout.outer * part * layout.inner:
                raise Error("Unpacking destination length disagrees")
            _require_disjoint(source, destinations[i])
            for j in range(i):
                _require_disjoint(
                    destinations[j].as_imm(), destinations[i].as_imm()
                )
        if layout.size == 0:
            return
        var row_size = layout.packed * layout.inner
        var offset = 0
        for destination in destinations:
            var chunk = len(destination) // layout.outer
            for row in range(layout.outer):
                _copy_block(
                    source,
                    destination,
                    row * row_size + offset,
                    row * chunk,
                    chunk,
                )
            offset += chunk

    def unpack[
        Element: TrivialRegisterPassable & Defaultable, //
    ](self, source: ImmSpan[Element, _]) raises -> List[List[Element]]:
        """Copy each tensor out of a packed buffer using this plan's shapes."""
        var layout = self._layout()
        if len(source) != layout.size:
            raise Error("Packed source length disagrees")
        if len(source) == 0:
            var empty = List[List[Element]]()
            for _ in self._inputs:
                empty.append(List[Element]())
            return empty^
        var outer = layout.outer
        var inner = layout.inner
        var row_size = layout.packed * inner
        var result = List[List[Element]]()
        var offset = 0
        for shape in self._inputs:
            var chunk = (
                _product(Span(shape)[self._before : len(shape) - self._after])
                * inner
            )
            var values = List[Element](length=outer * chunk, fill=Element())
            for row in range(outer):
                _copy_block(
                    source,
                    Span(values),
                    row * row_size + offset,
                    row * chunk,
                    chunk,
                )
            result.append(values^)
            offset += chunk
        return result^


def packing_plan(
    shapes: Span[List[Int], _], pattern: String
) raises -> PackRecipe:
    """Plan concatenation at * while retaining each input's packed dimensions.
    """
    var axes = _packing_axes(pattern)
    var before = axes[0]
    var after = axes[1]
    if len(shapes) == 0 or len(shapes[0]) < before + after:
        raise Error("Packing requires inputs matching the pattern rank")
    var first = Span(shapes[0])
    var shape = List(first[:before])
    var packed = 0
    for dims in shapes:
        if len(dims) < before + after:
            raise Error("Input rank is smaller than the packing pattern")
        var size = _product(Span(dims)[before : len(dims) - after])
        if size > 9223372036854775807 - packed:
            raise Error("Packed axis length overflows Int")
        packed += size
    shape.append(packed)
    for i in range(len(first) - after, len(first)):
        shape.append(first[i])
    return PackRecipe(List(shapes), shape^, before, after)


def pack[
    Element: TrivialRegisterPassable & Defaultable, //
](
    sources: Span[List[Element], _],
    shapes: Span[List[Int], _],
    pattern: String,
) raises -> Tuple[List[Element], PackRecipe]:
    """Pack buffers and return their reusable shape plan."""
    var recipe = packing_plan(shapes, pattern)
    var values = recipe.pack(sources)
    return values^, recipe^


def unpack[
    Element: TrivialRegisterPassable & Defaultable, //
](recipe: PackRecipe, source: ImmSpan[Element, _]) raises -> List[
    List[Element]
]:
    """Unpack with the shapes retained by a packing plan."""
    return recipe.unpack(source)


def unpacking_plan(
    shape: Span[Int, _], packed_shapes: Span[List[Int], _], pattern: String
) raises -> PackRecipe:
    """Plan unpacking from explicit packed dimensions; infer at most one -1."""
    var axes = _packing_axes(pattern)
    var before = axes[0]
    var after = axes[1]
    _ = _product(shape)
    if len(shape) != before + after + 1 or len(packed_shapes) == 0:
        raise Error("Unpacking shape rank or tensor count disagrees")
    var known = 0
    var unknown_part = -1
    var unknown_axis = -1
    var unknown_zero = False
    for part in range(len(packed_shapes)):
        var missing = -1
        var empty = False
        for axis in range(len(packed_shapes[part])):
            var dim = packed_shapes[part][axis]
            if dim == -1:
                if missing >= 0 or unknown_part >= 0:
                    raise Error("Only one unpacked dimension may be inferred")
                missing = axis
            else:
                if dim < 0:
                    raise Error("Axis lengths must be nonnegative")
                empty = empty or dim == 0
        if missing >= 0:
            unknown_part = part
            unknown_axis = missing
            unknown_zero = empty
        else:
            var size = _product(Span(packed_shapes[part]))
            if size > shape[before] - known:
                raise Error("Unpacked dimensions exceed the packed axis")
            known += size
    var inferred = 0
    if unknown_part >= 0:
        var remaining = shape[before] - known
        if unknown_zero:
            raise Error("Cannot infer unpacked dimensions")
        if remaining > 0:
            var dims = Span(packed_shapes[unknown_part])
            var factor = _multiply(
                _product(dims[:unknown_axis]),
                _product(dims[unknown_axis + 1 :]),
            )
            if remaining % factor != 0:
                raise Error("Cannot infer unpacked dimensions")
            inferred = remaining // factor
    elif known != shape[before]:
        raise Error("Unpacked dimensions do not fill the packed axis")
    var inputs = List[List[Int]]()
    for part in range(len(packed_shapes)):
        var dims = List(shape[:before])
        for axis in range(len(packed_shapes[part])):
            dims.append(
                inferred if part == unknown_part
                and axis == unknown_axis else packed_shapes[part][axis]
            )
        for axis in range(before + 1, len(shape)):
            dims.append(shape[axis])
        inputs.append(dims^)
    return PackRecipe(inputs^, List(shape), before, after)


def pack[
    Element: TrivialRegisterPassable & Defaultable,
    origin: Origin[mut=False],
    //,
](
    sources: ImmSpan[ImmSpan[Element, origin], _],
    shapes: Span[List[Int], _],
    pattern: String,
) raises -> Tuple[List[Element], PackRecipe]:
    """Pack borrowed buffers and return their reusable shape plan."""
    var recipe = packing_plan(shapes, pattern)
    var values = recipe.pack(sources)
    return values^, recipe^
