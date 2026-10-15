# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Custom row reductions over borrowed, pattern-indexed values."""

from std.collections import ImmSpan, List, Span
from std.memory import Pointer
from .pattern import TransformRecipe, _validate_recipe
from ._kernels import copy_rows
from ._buffers import _require_disjoint


struct ReductionValues[
    dtype: DType,
    source: Origin[mut=False],
    layout: Origin[mut=False],
](ImplicitlyCopyable, Sized):
    """Borrow one output row's reduction values without flattening or copying.

    Iteration order follows the plan's reduced axes. Indexing checks the row
    extent; source storage and metadata are checked before invoking a callback.
    """

    var _source: ImmSpan[Scalar[Self.dtype], Self.source]
    var _recipe: Pointer[TransformRecipe, Self.layout]
    var _row: Int

    def __init__(
        out self,
        values: ImmSpan[Scalar[Self.dtype], Self.source],
        ref[Self.layout] recipe: TransformRecipe,
        row: Int,
    ) raises:
        """Borrow a validated reduction row and its input storage."""
        _validate_recipe(recipe._data)
        if (
            recipe.operation() != .reduce
            or row < 0
            or row >= recipe.output_size()
            or len(values) < recipe._data.required_storage
        ):
            raise Error("Invalid custom reduction row or source storage")
        self._source = values
        self._recipe = Pointer(to=recipe)
        self._row = row

    def __len__(self) -> Int:
        return self._recipe[]._data.reduction_size

    def __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        if index < 0 or index >= len(self):
            raise Error("Reduction value index is outside the row")
        return self._source[
            self._recipe[]._data._source_index(self._row, index)
        ]


def _validate_custom[
    dtype: DType, //
](recipe: TransformRecipe, source: ImmSpan[Scalar[dtype], _]) raises:
    _validate_recipe(recipe._data)
    if (
        recipe.operation() != .reduce
        or len(source) < recipe._data.required_storage
    ):
        raise Error(
            "Custom reductions require a reduce plan and sufficient storage"
        )


def reduce_custom_into[
    dtype: DType,
    source: Origin[mut=False],
    Reducer: def[d: DType, s: Origin[mut=False], p: Origin[mut=False]](
        ReductionValues[d, s, p]
    ) raises -> Scalar[d],
    //,
](
    recipe: TransformRecipe,
    values: ImmSpan[Scalar[dtype], source],
    destination: Span[mut=True, Scalar[dtype], _],
    reducer: Reducer,
) raises:
    """Validate storage before writes; a callback failure retains completed rows.
    """
    _validate_custom(recipe, values)
    if len(destination) != recipe.output_size():
        raise Error("Custom reduction destination length disagrees")
    _require_disjoint(values, destination)
    _custom_validated(recipe, values, destination, reducer)


def reduce_custom[
    dtype: DType,
    source: Origin[mut=False],
    Reducer: def[d: DType, s: Origin[mut=False], p: Origin[mut=False]](
        ReductionValues[d, s, p]
    ) raises -> Scalar[d],
    //,
](
    recipe: TransformRecipe,
    values: ImmSpan[Scalar[dtype], source],
    reducer: Reducer,
) raises -> List[Scalar[dtype]]:
    """Materialize a custom reduction using a native function or captured closure.
    """
    _validate_custom(recipe, values)
    var result = List[Scalar[dtype]](length=recipe.output_size(), fill=0)
    _custom_validated(recipe, values, Span(result), reducer)
    return result^


def _custom_validated[
    dtype: DType,
    source: Origin[mut=False],
    Reducer: def[d: DType, s: Origin[mut=False], p: Origin[mut=False]](
        ReductionValues[d, s, p]
    ) raises -> Scalar[d],
    //,
](
    recipe: TransformRecipe,
    values: ImmSpan[Scalar[dtype], source],
    destination: Span[mut=True, Scalar[dtype], _],
    reducer: Reducer,
) raises:
    if len(destination) == 0:
        return
    if len(recipe._data.reduced_axes) == 0:
        copy_rows(recipe._data, values, destination)
        return
    var part = ReductionValues(values, recipe, 0)
    for row in range(len(destination)):
        part._row = row
        destination[row] = reducer[dtype, source, origin_of(recipe)](part)
