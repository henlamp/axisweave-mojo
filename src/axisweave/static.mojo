# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Pattern and shape lowering at compile time, using native Mojo 1.1 values."""

from std.collections import Array, Dict, ImmSpan, List, Span
from .kinds import Operation, Reduction
from ._kernels import affine_stride, combine, sum_rows, copy_block, bool_rows
from .pattern import _RecipeData, plan, _integer
from ._buffers import _require_disjoint, _copy_block


@fieldwise_init
struct _Prepared(Copyable):
    var recipe: _RecipeData
    var valid: Bool


def _make_static_plan[
    pattern: StringLiteral,
    operation: Operation,
    axes: StringLiteral,
    *shape: Int,
]() -> _Prepared:
    var sizes = Array[Int, len(shape)]()
    comptime for i in range(len(shape)):
        sizes[i] = shape[i]
    try:
        var hints = Dict[String, Int]()
        for entry in String(axes).split():
            var parts = String(entry).split("=")
            if len(parts) != 2:
                raise Error(
                    "Static axes use space-separated name=length entries"
                )
            var name = String(parts[0])
            if name in hints:
                raise Error("Duplicate static axis length")
            hints[name] = _integer(String(parts[1]))
        return _Prepared(
            plan(String(pattern), Span(sizes), operation, hints)._data.copy(),
            True,
        )
    except:
        return _Prepared(_RecipeData(), False)


def _checked_plan[
    pattern: StringLiteral,
    operation: Operation,
    axes: StringLiteral,
    *shape: Int,
]() -> _RecipeData:
    comptime prepared = _make_static_plan[pattern, operation, axes, *shape]()
    comptime assert (
        prepared.valid
    ), "Invalid static einops recipe: check pattern, shape and axes"
    return materialize[prepared.recipe]()


def _array[count: Int](values: List[Int]) -> Array[Int, count]:
    var result = Array[Int, count]()
    comptime for i in range(count):
        result[i] = values[i]
    return result^


def _flat_output_step(plan: _RecipeData) -> Int:
    return affine_stride(plan, Span(plan.output_axes))


def _flat_reduction_step(plan: _RecipeData) -> Int:
    return affine_stride(plan, Span(plan.reduced_axes))


def _last_size(plan: _RecipeData) -> Int:
    if len(plan.output_axes) == 0:
        return 1
    return plan.axis_sizes[plan.output_axes[len(plan.output_axes) - 1]]


def _last_stride(plan: _RecipeData) -> Int:
    if len(plan.output_axes) == 0:
        return 0
    return plan.source_strides[plan.output_axes[len(plan.output_axes) - 1]]


struct StaticRecipe[
    pattern: StringLiteral,
    *shape: Int,
    operation: Operation = .rearrange,
    axes: StringLiteral = "",
](ImplicitlyCopyable):
    """A zero-field recipe: parsing, validation and metadata run at compile time.

    Shape dimensions are variadic parameters. Named split/repeat lengths use
    axes="patch=2 copies=3". Runtime buffers are borrowed through checked spans.
    """

    comptime _plan = _checked_plan[
        Self.pattern, Self.operation, Self.axes, *Self.shape
    ]()
    comptime input_size = Self._plan.input_size
    comptime output_size = Self._plan.output_size
    comptime output_rank = len(Self._plan.output_shape)
    comptime _output_shape = _array[Self.output_rank](Self._plan.output_shape)
    comptime _sizes = _array[len(Self._plan.axis_sizes)](Self._plan.axis_sizes)
    comptime _strides = _array[len(Self._plan.source_strides)](
        Self._plan.source_strides
    )
    comptime _output_axes = _array[len(Self._plan.output_axes)](
        Self._plan.output_axes
    )
    comptime _reduced_axes = _array[len(Self._plan.reduced_axes)](
        Self._plan.reduced_axes
    )
    comptime _output_axis_count = len(Self._plan.output_axes)
    comptime _reduced_axis_count = len(Self._plan.reduced_axes)
    comptime reduction_size = Self._plan.reduction_size
    comptime _output_step = _flat_output_step(Self._plan)
    comptime _reduction_step = _flat_reduction_step(Self._plan)
    comptime _affine_reduction = (
        Self.reduction_size > 0
        and Self._output_step >= 0
        and Self._reduction_step >= 0
    )
    comptime _inner_size = _last_size(Self._plan)
    comptime _inner_stride = _last_stride(Self._plan)

    def __init__(out self):
        pass

    def output_shape(self) -> Array[Int, Self.output_rank]:
        """Materialize the compile-time output dimensions as a fixed-size array.
        """
        return materialize[Self._output_shape]()

    @inline(.always)
    def _index(self, output_index: Int, reduction_index: Int = 0) -> Int:
        var result = 0
        var remaining = output_index
        comptime if Self.output_size > 0:
            comptime for j in range(Self._output_axis_count - 1, -1, -1):
                comptime axis = Self._output_axes[j]
                comptime size = Self._sizes[axis]
                comptime stride = Self._strides[axis]
                result += (remaining % size) * stride
                remaining //= size
        remaining = reduction_index
        comptime if Self.reduction_size > 0:
            comptime for j in range(Self._reduced_axis_count - 1, -1, -1):
                comptime axis = Self._reduced_axes[j]
                comptime size = Self._sizes[axis]
                comptime stride = Self._strides[axis]
                result += (remaining % size) * stride
                remaining //= size
        return result

    def _validate_source[
        dtype: DType, //, reduction: Reduction
    ](self, source: ImmSpan[Scalar[dtype], _]) raises:
        if len(source) < Self.input_size:
            raise Error("Source buffer is smaller than the static input shape")
        comptime if Self.operation == .reduce:
            comptime assert reduction.supports[
                dtype
            ](), "Unsupported static reduction for this scalar DType"
            if (
                Self.reduction_size == 0
                and (reduction == .min or reduction == .max)
                and Self.output_size > 0
            ):
                raise Error("Minimum or maximum of an empty reduction")

    def execute_into[
        dtype: DType, //, reduction: Reduction = .sum
    ](
        self,
        source: ImmSpan[Scalar[dtype], _],
        destination: Span[mut=True, Scalar[dtype], _],
    ) raises:
        """Write to a disjoint exact-size buffer after validating the input."""
        self._validate_source[reduction](source)
        if len(destination) != Self.output_size:
            raise Error(
                "Destination length does not match the static output shape"
            )
        _require_disjoint(source, destination)
        self._execute_validated[reduction](source, destination)

    def _execute_validated[
        dtype: DType, //, reduction: Reduction
    ](
        self,
        source: ImmSpan[Scalar[dtype], _],
        destination: Span[mut=True, Scalar[dtype], _],
    ):
        comptime if Self.output_size == 0:
            return
        else:
            comptime if Self.operation != .reduce:
                comptime inner_size = Self._inner_size
                comptime inner_stride = Self._inner_stride
                for row in range(Self.output_size // inner_size):
                    var output_base = row * inner_size
                    copy_block(
                        source,
                        destination,
                        self._index(output_base),
                        output_base,
                        inner_size,
                        inner_stride,
                    )
            elif (
                reduction == .sum or reduction == .mean
            ) and Self._affine_reduction:
                sum_rows[reduction](
                    source,
                    destination,
                    Self.reduction_size,
                    Self._output_step,
                    Self._reduction_step,
                )
            else:
                for out_index in range(Self.output_size):
                    var result = Scalar[dtype](
                        1 if reduction == .prod or reduction == .all else 0
                    )
                    for r in range(Self.reduction_size):
                        result = combine[reduction](
                            result, source[self._index(out_index, r)], r == 0
                        )
                    comptime if reduction == .mean:
                        result /= Scalar[dtype](Self.reduction_size)
                    destination[out_index] = result

    def execute[
        dtype: DType, //, reduction: Reduction = .sum
    ](self, source: ImmSpan[Scalar[dtype], _]) raises -> List[Scalar[dtype]]:
        """Validate the source and reduction before allocating output."""
        self._validate_source[reduction](source)
        var result = List[Scalar[dtype]](length=Self.output_size, fill=0)
        self._execute_validated[reduction](source, Span(result))
        return result^

    def _validate_bool_source[
        reduction: Reduction
    ](self, source: ImmSpan[Bool, _]) raises:
        if len(source) < Self.input_size:
            raise Error("Source buffer is smaller than the static input shape")
        comptime if Self.operation == .reduce:
            comptime assert reduction.supports[
                .bool
            ](), "Static Bool reductions require any or all"

    def _execute_bool[
        reduction: Reduction
    ](self, source: ImmSpan[Bool, _], destination: Span[mut=True, Bool, _]):
        def index(row: Int, r: Int) {var self} -> Int:
            return self._index(row, r)

        bool_rows(
            source,
            destination,
            Self.reduction_size,
            reduction if Self.operation == .reduce else Reduction.any,
            index,
        )

    def execute_into[
        reduction: Reduction = .any
    ](
        self, source: ImmSpan[Bool, _], destination: Span[mut=True, Bool, _]
    ) raises:
        """Write a static Bool transformation or any/all reduction into caller storage.
        """
        self._validate_bool_source[reduction](source)
        if len(destination) != Self.output_size:
            raise Error(
                "Destination length does not match the static output shape"
            )
        _require_disjoint(source, destination)
        self._execute_bool[reduction](source, destination)

    def execute[
        reduction: Reduction = .any
    ](self, source: ImmSpan[Bool, _]) raises -> List[Bool]:
        """Materialize a static Bool transformation or logical reduction."""
        self._validate_bool_source[reduction](source)
        var result = List[Bool](length=Self.output_size, fill=False)
        self._execute_bool[reduction](source, Span(result))
        return result^

    def view[
        Element: TrivialRegisterPassable & Defaultable,
        origin: Origin[mut=False],
        //,
    ](self, source: Span[Element, origin]) raises -> StaticTensorView[
        Self.pattern,
        *Self.shape,
        Element=Element,
        origin=origin,
        operation=Self.operation,
        axes=Self.axes,
    ]:
        """Borrow a static rearrange or repeat; reduction views fail compilation.
        """
        comptime assert (
            Self.operation != .reduce
        ), "A reduction cannot be a static tensor view"
        return StaticTensorView[
            Self.pattern,
            *Self.shape,
            Element=Element,
            origin=origin,
            operation=Self.operation,
            axes=Self.axes,
        ](source)


struct StaticTensorView[
    pattern: StringLiteral,
    *dimensions: Int,
    origin: Origin[mut=False],
    Element: TrivialRegisterPassable & Defaultable,
    operation: Operation = .rearrange,
    axes: StringLiteral = "",
](Sized, TrivialRegisterPassable):
    """Borrowed storage; no runtime shape lists or pattern dictionaries."""

    comptime Recipe = StaticRecipe[
        Self.pattern, *Self.dimensions, operation=Self.operation, axes=Self.axes
    ]
    var _source: Span[Self.Element, Self.origin]

    def __init__(out self, source: Span[Self.Element, Self.origin]) raises:
        comptime assert (
            Self.operation != .reduce
        ), "A reduction cannot be a static tensor view"
        if len(source) < Self.Recipe.input_size:
            raise Error("Source buffer is smaller than the static input shape")
        self._source = source

    def __len__(self) -> Int:
        return Self.Recipe.output_size

    def shape(self) -> Array[Int, Self.Recipe.output_rank]:
        """Materialize logical dimensions without a heap-backed shape list."""
        return materialize[Self.Recipe._output_shape]()

    @inline(.always)
    def __getitem__(self, index: Int) raises -> Self.Element:
        if index < 0 or index >= len(self):
            raise Error("Static view index is outside its logical shape")
        return self._source[Self.Recipe()._index(index)]

    def materialize(self) raises -> List[Self.Element]:
        """Copy borrowed values into independent contiguous storage."""
        var result = List[Self.Element](length=len(self), fill=Self.Element())
        self._copy_into(Span(result))
        return result^

    def materialize_into(
        self, destination: Span[mut=True, Self.Element, _]
    ) raises:
        """Copy static logical values into a disjoint, exact-size caller buffer.
        """
        if len(destination) != len(self):
            raise Error("Static view destination length disagrees")
        _require_disjoint(self._source, destination)
        self._copy_into(destination)

    def _copy_into(self, destination: Span[mut=True, Self.Element, _]):
        comptime if Self.Recipe.output_size > 0:
            comptime inner = Self.Recipe._inner_size
            for row in range(Self.Recipe.output_size // inner):
                var start = row * inner
                _copy_block(
                    self._source,
                    destination,
                    Self.Recipe()._index(start),
                    start,
                    inner,
                    Self.Recipe._inner_stride,
                )
