# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Typed native CPU execution on caller-owned buffers."""

from std.collections import Dict, ImmSpan, List, Span
from .pattern import TransformRecipe, plan, _validate_recipe, _RecipeData
from .kinds import Operation, Reduction
from ._kernels import (
    copy_rows,
    execute_reduction,
    bool_rows,
)
from .custom import ReductionValues, reduce_custom
from ._buffers import _require_disjoint


def _validate_input[
    dtype: DType, //
](
    recipe: _RecipeData,
    source: ImmSpan[Scalar[dtype], _],
    reduction: Reduction,
) raises:
    _validate_recipe(recipe)
    if len(source) < recipe.required_storage:
        raise Error("Source buffer is smaller than the required storage")
    if recipe.operation == .reduce:
        if not reduction.supports[dtype]():
            raise Error("Unsupported reduction for this scalar DType")
        if (
            recipe.reduction_size == 0
            and (reduction == .min or reduction == .max)
            and recipe.output_size > 0
        ):
            raise Error("Minimum or maximum of an empty reduction")


def _execute_validated[
    dtype: DType, //
](
    recipe: _RecipeData,
    source: ImmSpan[Scalar[dtype], _],
    destination: Span[mut=True, Scalar[dtype], _],
    reduction: Reduction,
):
    if recipe.operation != .reduce:
        copy_rows(recipe, source, destination)
        return
    comptime if dtype == .bool:
        if reduction == .any:
            execute_reduction[.any](recipe, source, destination)
        else:
            execute_reduction[.all](recipe, source, destination)
    else:
        if reduction == .sum:
            execute_reduction[.sum](recipe, source, destination)
        elif reduction == .mean:
            execute_reduction[.mean](recipe, source, destination)
        elif reduction == .min:
            execute_reduction[.min](recipe, source, destination)
        elif reduction == .max:
            execute_reduction[.max](recipe, source, destination)
        else:
            execute_reduction[.prod](recipe, source, destination)


def execute_into[
    dtype: DType, //
](
    recipe: TransformRecipe,
    source: ImmSpan[Scalar[dtype], _],
    destination: Span[mut=True, Scalar[dtype], _],
    reduction: Reduction = .sum,
) raises:
    """Write to an exact-size, disjoint buffer; errors precede all writes."""
    _validate_input(recipe._data, source, reduction)
    if len(destination) != recipe.output_size():
        raise Error("Destination length does not match the output shape")
    _require_disjoint(source, destination)
    _execute_validated(recipe._data, source, destination, reduction)


def execute[
    dtype: DType, //
](
    recipe: TransformRecipe,
    source: ImmSpan[Scalar[dtype], _],
    reduction: Reduction = .sum,
) raises -> List[Scalar[dtype]]:
    """Validate a recipe and source before allocating its owned output."""
    _validate_input(recipe._data, source, reduction)
    var result = List[Scalar[dtype]](length=recipe.output_size(), fill=0)
    _execute_validated(recipe._data, source, Span(result), reduction)
    return result^


def rearrange[
    dtype: DType, //
](
    source: ImmSpan[Scalar[dtype], _],
    shape: Span[Int, _],
    pattern: String,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Scalar[dtype]]:
    """Plan and copy a rearrangement, preserving every input axis."""
    return execute(plan(pattern, shape, .rearrange, axes_lengths), source)


def repeat[
    dtype: DType, //
](
    source: ImmSpan[Scalar[dtype], _],
    shape: Span[Int, _],
    pattern: String,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Scalar[dtype]]:
    """Plan and copy a broadcast with explicitly sized new axes."""
    return execute(plan(pattern, shape, .repeat, axes_lengths), source)


def reduce[
    dtype: DType, //
](
    source: ImmSpan[Scalar[dtype], _],
    shape: Span[Int, _],
    pattern: String,
    reduction: Reduction,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Scalar[dtype]]:
    """Plan a supported numeric or logical reduction with the same output DType.
    """
    return execute(
        plan(pattern, shape, .reduce, axes_lengths), source, reduction
    )


def _validate_bool(
    recipe: _RecipeData, source: ImmSpan[Bool, _], reduction: Reduction
) raises:
    _validate_recipe(recipe)
    if len(source) < recipe.required_storage:
        raise Error("Source buffer is smaller than the required storage")
    if recipe.operation == .reduce and not reduction.supports[.bool]():
        raise Error("Bool reductions require any or all")


def _execute_bool(
    recipe: _RecipeData,
    source: ImmSpan[Bool, _],
    destination: Span[mut=True, Bool, _],
    reduction: Reduction,
):
    if recipe.operation != .reduce:
        copy_rows(recipe, source, destination)
        return

    def index(row: Int, r: Int) {imm recipe} -> Int:
        return recipe._source_index(row, r)

    bool_rows(
        source,
        destination,
        recipe.reduction_size,
        reduction if recipe.operation == .reduce else Reduction.any,
        index,
    )


def execute_into(
    recipe: TransformRecipe,
    source: ImmSpan[Bool, _],
    destination: Span[mut=True, Bool, _],
    reduction: Reduction = .any,
) raises:
    """Write a Bool transformation/reduction after checking metadata and overlap.
    """
    _validate_bool(recipe._data, source, reduction)
    if len(destination) != recipe.output_size():
        raise Error("Destination length does not match the output shape")
    _require_disjoint(source, destination)
    _execute_bool(recipe._data, source, destination, reduction)


def execute(
    recipe: TransformRecipe,
    source: ImmSpan[Bool, _],
    reduction: Reduction = .any,
) raises -> List[Bool]:
    """Materialize a Bool transformation or logical reduction."""
    _validate_bool(recipe._data, source, reduction)
    var result = List[Bool](length=recipe.output_size(), fill=False)
    _execute_bool(recipe._data, source, Span(result), reduction)
    return result^


def reduce(
    source: ImmSpan[Bool, _],
    shape: Span[Int, _],
    pattern: String,
    reduction: Reduction,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Bool]:
    """Reduce native Bool buffers with any/all, preserving empty-row identities.
    """
    return execute(
        plan(pattern, shape, .reduce, axes_lengths), source, reduction
    )


def rearrange(
    source: ImmSpan[Bool, _],
    shape: Span[Int, _],
    pattern: String,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Bool]:
    """Rearrange native Bool buffers, preserving the planned logical order."""
    return execute(plan(pattern, shape, .rearrange, axes_lengths), source)


def repeat(
    source: ImmSpan[Bool, _],
    shape: Span[Int, _],
    pattern: String,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Bool]:
    """Broadcast native Bool buffers along explicitly sized axes."""
    return execute(plan(pattern, shape, .repeat, axes_lengths), source)


def reduce[
    dtype: DType,
    origin: Origin[mut=False],
    Reducer: def[d: DType, s: Origin[mut=False], p: Origin[mut=False]](
        ReductionValues[d, s, p]
    ) raises -> Scalar[d],
    //,
](
    source: ImmSpan[Scalar[dtype], origin],
    shape: Span[Int, _],
    pattern: String,
    *,
    reducer: Reducer,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
) raises -> List[Scalar[dtype]]:
    """Plan a reduction and invoke a native row callback, including captured state.
    """
    var recipe = plan(pattern, shape, .reduce, axes_lengths)
    return reduce_custom(recipe, source, reducer)
