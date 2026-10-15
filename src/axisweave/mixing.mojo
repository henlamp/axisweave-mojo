# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Named-axis linear inference with caller-owned weights and optional bias."""

from std.origin import ImmStaticOrigin
from std.collections import Dict, ImmSpan, List, Optional, Span
from .contraction import EinsumRecipe, einsum_plan, _axes
from .pattern import TransformRecipe, plan, _validate_recipe
from ._buffers import _require_disjoint
from ._sources import _PairSources


struct MixRecipe(Copyable):
    """Apply a two-operand named contraction and optional broadcast bias.

    Weights and bias remain caller-owned. This plan supplies inference only;
    initialization, gradient calculation and parameter updates are separate.
    """

    var _contraction: EinsumRecipe
    var _bias: Optional[TransformRecipe]

    def __init__(
        out self,
        var contraction: EinsumRecipe,
        var bias: Optional[TransformRecipe],
    ) raises:
        """Own validated contraction and optional bias metadata."""
        _ = contraction._extents()
        if len(contraction._shapes) != 2:
            raise Error("Mixing requires input and weight operands")
        if bias:
            _validate_recipe(bias.value()._data)
            if (
                bias.value().operation() != .repeat
                or bias.value().output_shape() != contraction.output_shape()
            ):
                raise Error(
                    "Mixing bias must broadcast to the contraction output"
                )
        self._contraction = contraction^
        self._bias = bias^

    def output_shape(self) -> Span[Int, origin_of(self._contraction._sizes)]:
        """Borrow the named linear output dimensions."""
        return self._contraction.output_shape()

    def _validate_bias[
        dtype: DType, //
    ](self, bias: ImmSpan[Scalar[dtype], _]) raises:
        if self._bias:
            _validate_recipe(self._bias.value()._data)
            if (
                self._bias.value().operation() != .repeat
                or self._bias.value().output_shape()
                != self._contraction.output_shape()
            ):
                raise Error(
                    "Mixing bias must broadcast to the contraction output"
                )
            if (
                len(bias) != self._bias.value().input_size()
                or len(bias) < self._bias.value()._data.required_storage
            ):
                raise Error("Mixing bias length disagrees")
        elif len(bias) != 0:
            raise Error("This mixing plan has no bias")

    def execute_into[
        dtype: DType, //
    ](
        self,
        input: ImmSpan[Scalar[dtype], _],
        weights: ImmSpan[Scalar[dtype], _],
        destination: Span[mut=True, Scalar[dtype], _],
        *,
        bias: ImmSpan[Scalar[dtype], _] = ImmSpan[
            Scalar[dtype], ImmStaticOrigin
        ](),
    ) raises:
        """Write a named linear result, validating every buffer before writes.
        """
        self._validate_bias(bias)
        _require_disjoint(bias, destination)
        var sources = _PairSources(input, weights)
        self._contraction._execute_into(sources, destination)
        self._add_bias(bias, destination)

    def execute[
        dtype: DType, //
    ](
        self,
        input: ImmSpan[Scalar[dtype], _],
        weights: ImmSpan[Scalar[dtype], _],
        *,
        bias: ImmSpan[Scalar[dtype], _] = ImmSpan[
            Scalar[dtype], ImmStaticOrigin
        ](),
    ) raises -> List[Scalar[dtype]]:
        """Materialize a named linear result without copying weights or bias."""
        self._validate_bias(bias)
        var sources = _PairSources(input, weights)
        var result = self._contraction._execute(sources)
        self._add_bias(bias, Span(result))
        return result^

    def _add_bias[
        dtype: DType, //
    ](
        self,
        bias: ImmSpan[Scalar[dtype], _],
        destination: Span[mut=True, Scalar[dtype], _],
    ):
        if self._bias:
            for i in range(len(destination)):
                destination[i] += bias[
                    self._bias.value()._data._source_index(i)
                ]


def mix_plan(
    input_shape: ImmSpan[Int, _],
    weight_shape: ImmSpan[Int, _],
    pattern: String,
) raises -> MixRecipe:
    """Plan a two-operand named linear transform without bias."""
    var shapes: List[List[Int]] = [List(input_shape), List(weight_shape)]
    return MixRecipe(
        einsum_plan(Span(shapes), pattern), Optional[TransformRecipe]()
    )


def mix_plan(
    input_shape: ImmSpan[Int, _],
    weight_shape: ImmSpan[Int, _],
    pattern: String,
    *,
    bias_shape: ImmSpan[Int, _],
    bias_axes: String,
) raises -> MixRecipe:
    """Plan a named linear transform with bias axes selected from its output."""
    var shapes: List[List[Int]] = [List(input_shape), List(weight_shape)]
    var contraction = einsum_plan(Span(shapes), pattern)
    var output = String(pattern.split("->")[1])
    var names = _axes(output)
    for name in _axes(bias_axes):
        if name == "..." or name not in names:
            raise Error("Bias axes must be named contraction output axes")
    var sizes = contraction.output_shape()
    var hints = Dict[String, Int]()
    var expanded = List[String]()
    var next = 0
    for name in names:
        if name == "...":
            for i in range(len(sizes) - len(names) + 1):
                var generated = String("mixbatch", i)
                while generated in names:
                    generated += "x"
                hints[generated] = sizes[next]
                expanded.append(generated^)
                next += 1
        else:
            hints[name] = sizes[next]
            expanded.append(name.copy())
            next += 1
    var bias = plan(
        bias_axes + " -> " + String(" ").join(expanded),
        bias_shape,
        .repeat,
        hints,
    )
    return MixRecipe(contraction^, Optional(bias^))
