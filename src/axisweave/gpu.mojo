# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Explicit Float32 GPU execution. Requires the matching MAX Core SDK."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import List, Span
from .pattern import TransformRecipe, _validate_recipe, _product
from .contraction import EinsumRecipe
from .kinds import Reduction
from ._gpu_kernels import _transform, _contract


comptime _BLOCK = 128


def _grid(count: Int) raises -> Int:
    var blocks = (count - 1) // _BLOCK + 1
    if count <= 0 or blocks > 2147483647:
        raise Error(
            "GPU output exceeds the portable one-dimensional grid limit"
        )
    return blocks


def _require_gpu(ctx: DeviceContext) raises:
    if ctx.api() != "metal" and ctx.api() != "cuda":
        raise Error("Axisweave GPU execution requires Metal or CUDA")


def _require_context(ctx: DeviceContext, buffer: DeviceBuffer[.float32]) raises:
    if buffer.context() != ctx:
        raise Error("Device buffer belongs to another context")


def _require_disjoint(
    source: DeviceBuffer[.float32], destination: DeviceBuffer[.float32]
) raises:
    if len(source) == 0 or len(destination) == 0:
        return
    if max(len(source), len(destination)) > 9223372036854775807 // 4:
        raise Error("Device buffer byte extent overflows")
    var start = UInt(source.unsafe_ptr())
    var target = UInt(destination.unsafe_ptr())
    if start <= target:
        if target - start < UInt(len(source) * 4):
            raise Error("Device source and destination overlap")
    elif start - target < UInt(len(destination) * 4):
        raise Error("Device source and destination overlap")


def _narrow_metadata(values: Span[Int64, _]) -> Bool:
    for value in values:
        if value < 0 or value > 2147483647:
            return False
    return True


def _upload(
    ctx: DeviceContext, values: Span[Int64, _]
) raises -> DeviceBuffer[.int64]:
    var device = ctx.enqueue_create_buffer[.int64](len(values))
    ctx.enqueue_copy(dst_buf=device, src=values.as_imm())
    # Keep the borrowed host metadata alive until the queued copy completes.
    ctx.synchronize()
    return device^


struct DeviceTransform(Copyable):
    """Reusable Float32 transform/reduction on buffers from one GPU context."""

    var _ctx: DeviceContext
    var _recipe: TransformRecipe
    var _reduction: Reduction
    var _metadata: DeviceBuffer[.int64]
    var _narrow: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        recipe: TransformRecipe,
        reduction: Reduction = .sum,
    ) raises:
        """Validate a CPU plan and upload its compact indexing metadata once."""
        _require_gpu(ctx)
        var data = recipe._data.copy()
        _validate_recipe(data)
        if data.output_size > 0:
            _ = _grid(data.output_size)
        if data.operation == .reduce:
            if not reduction.supports[.float32]():
                raise Error(
                    "Float32 GPU reductions require sum/mean/min/max/prod"
                )
            if (
                data.reduction_size == 0
                and data.output_size > 0
                and (reduction == .min or reduction == .max)
            ):
                raise Error("Minimum or maximum of an empty reduction")
        self._ctx = ctx
        self._recipe = recipe.copy()
        self._reduction = reduction
        var values: List[Int64] = [
            Int64(len(data.output_axes)),
            Int64(len(data.reduced_axes)),
            Int64(data.input_offset),
            Int64(data.reduction_size),
        ]
        for axis in data.output_axes:
            values.append(Int64(data.axis_sizes[axis]))
            values.append(Int64(data.source_strides[axis]))
        for axis in data.reduced_axes:
            values.append(Int64(data.axis_sizes[axis]))
            values.append(Int64(data.source_strides[axis]))
        self._narrow = (
            data.output_size <= 2147483647
            and data.required_storage <= 2147483647
            and _narrow_metadata(Span(values))
        )
        self._metadata = _upload(ctx, Span(values))

    def output_shape(
        self,
    ) -> Span[Int, origin_of(self._recipe._data.output_shape)]:
        """Borrow the planned dimensions without mutable metadata access."""
        return self._recipe.output_shape()

    def execute(
        self, source: DeviceBuffer[.float32]
    ) raises -> DeviceBuffer[.float32]:
        """Allocate output and enqueue execution; no host transfer or CPU fallback.
        """
        _require_context(self._ctx, source)
        if len(source) < self._recipe._data.required_storage:
            raise Error("Device source is smaller than the required storage")
        var result = self._ctx.enqueue_create_buffer[.float32](
            self._recipe.output_size()
        )
        self.execute_into(source, result)
        return result^

    def execute_into(
        self,
        source: DeviceBuffer[.float32],
        destination: DeviceBuffer[.float32],
    ) raises:
        """Check contexts, extents and overlap before enqueuing any output writes.
        """
        _require_context(self._ctx, source)
        _require_context(self._ctx, destination)
        if len(source) < self._recipe._data.required_storage:
            raise Error("Device source is smaller than the required storage")
        var count = self._recipe.output_size()
        if len(destination) != count:
            raise Error("Device destination length disagrees")
        _require_disjoint(source, destination)
        if count == 0:
            return
        if self._recipe.operation() != .reduce:
            self._launch[.sum, False](source, destination, count)
        elif self._reduction == .sum:
            self._launch[.sum, True](source, destination, count)
        elif self._reduction == .mean:
            self._launch[.mean, True](source, destination, count)
        elif self._reduction == .min:
            self._launch[.min, True](source, destination, count)
        elif self._reduction == .max:
            self._launch[.max, True](source, destination, count)
        else:
            self._launch[.prod, True](source, destination, count)

    def _launch[
        kind: Reduction, reducing: Bool
    ](
        self,
        source: DeviceBuffer[.float32],
        destination: DeviceBuffer[.float32],
        count: Int,
    ) raises:
        if self._narrow:
            self._enqueue[kind, reducing, True](source, destination, count)
        else:
            self._enqueue[kind, reducing, False](source, destination, count)

    def _enqueue[
        kind: Reduction, reducing: Bool, narrow: Bool
    ](
        self,
        source: DeviceBuffer[.float32],
        destination: DeviceBuffer[.float32],
        count: Int,
    ) raises:
        self._ctx.enqueue_function[_transform[kind, reducing, narrow]](
            source,
            destination,
            self._metadata,
            Int64(count),
            grid_dim=_grid(count),
            block_dim=_BLOCK,
        )


struct DeviceEinsum(Copyable):
    """Reusable binary Float32 contraction, including diagonals and broadcasting.
    """

    var _ctx: DeviceContext
    var _recipe: EinsumRecipe
    var _metadata: DeviceBuffer[.int64]
    var _narrow: Bool
    var _output_count: Int
    var _left_count: Int
    var _right_count: Int

    def __init__(out self, ctx: DeviceContext, recipe: EinsumRecipe) raises:
        """Validate a two-operand CPU contraction plan and upload its metadata.
        """
        _require_gpu(ctx)
        var extents = recipe._extents()
        if extents[0] > 0:
            _ = _grid(extents[0])
        if len(recipe._shapes) != 2:
            raise Error("GPU contractions require exactly two operands")
        self._ctx = ctx
        self._recipe = recipe.copy()
        self._output_count = extents[0]
        self._left_count = _product(Span(recipe._shapes[0]))
        self._right_count = _product(Span(recipe._shapes[1]))
        var values: List[Int64] = [
            Int64(recipe._output_rank),
            Int64(len(recipe._sizes) - recipe._output_rank),
            Int64(extents[1]),
        ]
        for axis in range(len(recipe._sizes)):
            values.append(Int64(recipe._sizes[axis]))
            values.append(Int64(recipe._strides[0][axis]))
            values.append(Int64(recipe._strides[1][axis]))
        self._narrow = (
            self._output_count <= 2147483647
            and max(self._left_count, self._right_count) <= 2147483647
            and _narrow_metadata(Span(values))
        )
        self._metadata = _upload(ctx, Span(values))

    def output_shape(self) -> Span[Int, origin_of(self._recipe._sizes)]:
        """Borrow the contraction's output dimensions."""
        return self._recipe.output_shape()

    def execute(
        self, left: DeviceBuffer[.float32], right: DeviceBuffer[.float32]
    ) raises -> DeviceBuffer[.float32]:
        """Allocate output and enqueue a binary contraction on the GPU."""
        self._validate(left, right)
        var result = self._ctx.enqueue_create_buffer[.float32](
            self._output_count
        )
        self.execute_into(left, right, result)
        return result^

    def _validate(
        self, left: DeviceBuffer[.float32], right: DeviceBuffer[.float32]
    ) raises:
        _require_context(self._ctx, left)
        _require_context(self._ctx, right)
        if len(left) != self._left_count or len(right) != self._right_count:
            raise Error("GPU contraction source length disagrees")

    def execute_into(
        self,
        left: DeviceBuffer[.float32],
        right: DeviceBuffer[.float32],
        destination: DeviceBuffer[.float32],
    ) raises:
        """Validate both sources and disjoint exact output before writing."""
        self._validate(left, right)
        _require_context(self._ctx, destination)
        if len(destination) != self._output_count:
            raise Error("GPU contraction destination length disagrees")
        _require_disjoint(left, destination)
        _require_disjoint(right, destination)
        if self._output_count == 0:
            return
        if self._narrow:
            self._enqueue[True](left, right, destination)
        else:
            self._enqueue[False](left, right, destination)

    def _enqueue[
        narrow: Bool
    ](
        self,
        left: DeviceBuffer[.float32],
        right: DeviceBuffer[.float32],
        destination: DeviceBuffer[.float32],
    ) raises:
        self._ctx.enqueue_function[_contract[narrow]](
            left,
            right,
            destination,
            self._metadata,
            Int64(self._output_count),
            grid_dim=_grid(self._output_count),
            block_dim=_BLOCK,
        )
