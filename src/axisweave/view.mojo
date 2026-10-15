# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Read-only tensor views that borrow storage without copying its values."""

from std.collections import Dict, List, Span
from .pattern import TransformRecipe, plan, _validate_recipe
from ._buffers import _require_disjoint
from ._kernels import copy_rows


struct TensorView[
    Element: TrivialRegisterPassable & Defaultable, origin: Origin[mut=False]
](Copyable, Sized):
    """A logical row-major tensor view with compiler-checked storage lifetime.

    Shape metadata is owned by the view. The input value buffer is borrowed.
    Composed output axes need not be representable by ordinary affine strides.
    """

    var _source: Span[Self.Element, Self.origin]
    var _recipe: TransformRecipe

    def __init__(
        out self,
        source: Span[Self.Element, Self.origin],
        var recipe: TransformRecipe,
    ) raises:
        self._source = source
        self._recipe = recipe^
        self._validate()

    def _validate(self) raises:
        _validate_recipe(self._recipe._data)
        if self._recipe.operation() == .reduce:
            raise Error("A reduction requires execution, not a tensor view")
        if len(self._source) < self._recipe._data.required_storage:
            raise Error("Source buffer is smaller than the required storage")

    def __len__(self) -> Int:
        return self._recipe.output_size()

    def shape(self) -> Span[Int, origin_of(self._recipe._data.output_shape)]:
        """Borrow immutable dimensions for this view's lifetime."""
        return self._recipe.output_shape()

    def __getitem__(self, index: Int) raises -> Self.Element:
        if index < 0 or index >= len(self):
            raise Error("Tensor view index is outside its logical shape")
        return self._source[self._recipe._data._source_index(index)]

    def materialize(self) raises -> List[Self.Element]:
        """Copy the borrowed logical values into independent contiguous storage.
        """
        self._validate()
        var result = List[Self.Element](length=len(self), fill=Self.Element())
        copy_rows(self._recipe._data, self._source, Span(result))
        return result^

    def materialize_into(
        self, destination: Span[mut=True, Self.Element, _]
    ) raises:
        """Copy logical values into a disjoint, exact-size caller buffer."""
        self._validate()
        if len(destination) != len(self):
            raise Error("Tensor view destination length disagrees")
        _require_disjoint(self._source, destination)
        copy_rows(self._recipe._data, self._source, destination)


def view[
    Element: TrivialRegisterPassable & Defaultable,
    origin: Origin[mut=False],
    //,
](
    recipe: TransformRecipe,
    source: Span[Element, origin],
) raises -> TensorView[
    Element, origin
]:
    """Borrow a source buffer through a prepared rearrange or repeat recipe."""
    return TensorView(source, recipe.copy())


def rearrange_view[
    Element: TrivialRegisterPassable & Defaultable,
    origin: Origin[mut=False],
    //,
](
    source: Span[Element, origin],
    shape: Span[Int, _],
    pattern: String,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> TensorView[Element, origin]:
    """Rearrange logical coordinates without allocating an output value buffer.
    """
    return TensorView(source, plan(pattern, shape, .rearrange, axes_lengths))


def repeat_view[
    Element: TrivialRegisterPassable & Defaultable,
    origin: Origin[mut=False],
    //,
](
    source: Span[Element, origin],
    shape: Span[Int, _],
    pattern: String,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> TensorView[Element, origin]:
    """Broadcast new axes without copying the repeated values."""
    return TensorView(source, plan(pattern, shape, .repeat, axes_lengths))
