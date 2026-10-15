# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Pattern parsing and shape planning; independent of tensor storage."""

from std.collections import Dict, List, Span
from .kinds import Operation


def _multiply(left: Int, right: Int) raises -> Int:
    if left < 0 or right < 0:
        raise Error("Axis lengths must be nonnegative")
    if right != 0 and left > 9223372036854775807 // right:
        raise Error("Shape product exceeds Int range")
    return left * right


def _product(values: Span[Int, _]) raises -> Int:
    var empty = False
    for value in values:
        if value < 0:
            raise Error("Axis lengths must be nonnegative")
        empty = empty or value == 0
    if empty:
        return 0
    var result = 1
    for value in values:
        result = _multiply(result, value)
    return result


def _axis_product(sizes: Span[Int, _], axes: Span[Int, _]) raises -> Int:
    for axis in axes:
        if sizes[axis] == 0:
            return 0
    var product = 1
    for axis in axes:
        product = _multiply(product, sizes[axis])
    return product


def _storage_extent(
    size: Int, shape: Span[Int, _], strides: Span[Int, _], offset: Int
) raises -> Int:
    if size == 0:
        return 0
    var last = offset
    for axis in range(len(shape)):
        var delta = _multiply(shape[axis] - 1, strides[axis])
        if last > 9223372036854775806 - delta:
            raise Error("Recipe storage extent overflows")
        last += delta
    return last + 1


def _integer(text: String) raises -> Int:
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("Empty axis length")
    var value = 0
    for byte in bytes:
        var digit = Int(byte) - 48
        if (
            digit < 0
            or digit > 9
            or value > (9223372036854775807 - digit) // 10
        ):
            raise Error("Axis length must be a nonnegative Int")
        value = value * 10 + digit
    return value


def _numeric(name: String) -> Bool:
    if name.byte_length() == 0:
        return False
    var first = Int(name.as_bytes()[0])
    return 48 <= first and first <= 57


struct _Expression(Copyable):
    var groups: List[List[String]]
    var composed: List[Bool]
    var anonymous_count: Int

    def __init__(out self):
        self.groups = List[List[String]]()
        self.composed = List[Bool]()
        self.anonymous_count = 0


def _check_name(name: String) raises:
    if name == "...":
        return
    if _numeric(name):
        if _integer(name) == 0:
            raise Error("Anonymous axes must be positive")
        return
    if name.byte_length() == 0:
        raise Error("Empty axis name")
    var bytes = name.as_bytes()
    for i in range(len(bytes)):
        var c = Int(bytes[i])
        var letter = (65 <= c and c <= 90) or (97 <= c and c <= 122)
        var digit = 48 <= c and c <= 57
        if not (letter or (i > 0 and (digit or c == 95))):
            raise Error("Axis names must be ASCII identifiers")
    if bytes[len(bytes) - 1] == 95:
        raise Error("Axis names cannot end with an underscore")


def _flush(
    mut expression: _Expression,
    mut group: List[String],
    token: String,
    grouped: Bool,
    output_side: Bool,
) raises:
    if token.byte_length() == 0:
        return
    _check_name(token)
    var singleton = _numeric(token) and _integer(token) == 1
    var name = token.copy()
    if _numeric(token) and not singleton:
        name = String(
            "#",
            "out" if output_side else "in",
            expression.anonymous_count,
            ":",
            token,
        )
        expression.anonymous_count += 1
    if grouped:
        if not singleton:
            group.append(name^)
    else:
        var single = List[String]()
        if not singleton:
            single.append(name^)
        expression.groups.append(single^)
        expression.composed.append(False)


def _parse(text: String, output_side: Bool = False) raises -> _Expression:
    var expression = _Expression()
    var group = List[String]()
    var grouped = False
    var start = 0
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        var byte = bytes[i]
        if byte > 127:
            raise Error("Patterns require ASCII axis names")
        if (
            byte == 40
            or byte == 41
            or byte == 32
            or byte == 9
            or byte == 10
            or byte == 13
        ):
            _flush(
                expression,
                group,
                String(text[byte=start:i]),
                grouped,
                output_side,
            )
            start = i + 1
            if byte == 40:
                if grouped:
                    raise Error("Nested parentheses are not supported")
                grouped = True
            elif byte == 41:
                if not grouped:
                    raise Error("Unmatched closing parenthesis")
                expression.groups.append(group^)
                expression.composed.append(True)
                group = List[String]()
                grouped = False
    _flush(expression, group, String(text[byte=start:]), grouped, output_side)
    if grouped:
        raise Error("Unclosed parenthesis")
    var seen = Dict[String, Bool]()
    for g in range(len(expression.groups)):
        for a in range(len(expression.groups[g])):
            var name = expression.groups[g][a].copy()
            if name in seen:
                raise Error(
                    "An axis can occur only once on each side of a pattern"
                )
            seen[name] = True
    return expression^


def _has_ellipsis(expression: _Expression) -> Bool:
    for g in range(len(expression.groups)):
        for a in range(len(expression.groups[g])):
            if expression.groups[g][a] == "...":
                return True
    return False


def _expand(
    expression: _Expression, count: Int, input_side: Bool
) raises -> _Expression:
    var expanded = _Expression()
    for g in range(len(expression.groups)):
        var group = List[String]()
        for a in range(len(expression.groups[g])):
            var name = expression.groups[g][a].copy()
            if name != "...":
                group.append(name.copy())
            else:
                if input_side and expression.composed[g]:
                    raise Error("Input ellipsis cannot be inside parentheses")
                for e in range(count):
                    var label = String("_ellipsis_", e)
                    if expression.composed[g]:
                        group.append(label^)
                    else:
                        var single: List[String] = [label^]
                        expanded.groups.append(single^)
                        expanded.composed.append(False)
        if (
            len(expression.groups[g]) == 1
            and expression.groups[g][0] == "..."
            and not expression.composed[g]
        ):
            continue
        expanded.groups.append(group^)
        expanded.composed.append(expression.composed[g])
    return expanded^


struct _RecipeData(Copyable):
    var input_shape: List[Int]
    var output_shape: List[Int]
    var axis_sizes: List[Int]
    var source_strides: List[Int]
    var output_axes: List[Int]
    var reduced_axes: List[Int]
    var input_size: Int
    var input_axis_count: Int
    var output_size: Int
    var reduction_size: Int
    var required_storage: Int
    var input_offset: Int
    var operation: Operation

    def __init__(out self):
        self.input_shape = List[Int]()
        self.output_shape = List[Int]()
        self.axis_sizes = List[Int]()
        self.source_strides = List[Int]()
        self.output_axes = List[Int]()
        self.reduced_axes = List[Int]()
        self.input_size = 1
        self.input_axis_count = 0
        self.output_size = 1
        self.reduction_size = 1
        self.required_storage = 1
        self.input_offset = 0
        self.operation = .rearrange

    def _source_index(self, output_index: Int, reduction_index: Int = 0) -> Int:
        var result = self.input_offset
        var remaining = output_index
        for j in range(len(self.output_axes) - 1, -1, -1):
            var axis = self.output_axes[j]
            result += (remaining % self.axis_sizes[axis]) * self.source_strides[
                axis
            ]
            remaining //= self.axis_sizes[axis]
        remaining = reduction_index
        for j in range(len(self.reduced_axes) - 1, -1, -1):
            var axis = self.reduced_axes[j]
            result += (remaining % self.axis_sizes[axis]) * self.source_strides[
                axis
            ]
            remaining //= self.axis_sizes[axis]
        return result


struct TransformRecipe(Copyable):
    """Owned transform metadata with immutable, lifetime-tracked shape access.
    """

    var _data: _RecipeData

    def __init__(out self):
        self._data = _RecipeData()

    def __init__(out self, var data: _RecipeData):
        self._data = data^

    def input_shape(self) -> Span[Int, origin_of(self._data.input_shape)]:
        """Borrow the input dimensions without exposing mutable storage."""
        return Span(self._data.input_shape)

    def output_shape(self) -> Span[Int, origin_of(self._data.output_shape)]:
        """Borrow the output dimensions for this recipe's lifetime."""
        return Span(self._data.output_shape)

    def input_size(self) -> Int:
        """Number of logical input elements."""
        return self._data.input_size

    def output_size(self) -> Int:
        """Number of logical output elements."""
        return self._data.output_size

    def operation(self) -> Operation:
        """The typed transform operation."""
        return self._data.operation


def _identity_recipe(
    shape: Span[Int, _], var strides: List[Int], offset: Int
) raises -> TransformRecipe:
    if len(strides) != len(shape):
        raise Error("Stride rank does not match shape rank")
    var data = _RecipeData()
    data.input_shape = List(shape)
    data.output_shape = List(shape)
    data.axis_sizes = List(shape)
    data.source_strides = strides^
    data.input_axis_count = len(shape)
    data.input_size = _product(shape)
    data.output_size = data.input_size
    data.input_offset = offset
    data.required_storage = _storage_extent(
        data.input_size, shape, Span(data.source_strides), offset
    )
    for axis in range(len(shape)):
        data.output_axes.append(axis)
    return TransformRecipe(data^)


def plan(
    pattern: String,
    shape: Span[Int, _],
    operation: Operation = .rearrange,
    axes_lengths: Dict[String, Int] = Dict[String, Int](),
    *,
    strides: List[Int] = List[Int](),
    offset: Int = 0,
) raises -> TransformRecipe:
    """Prepare a pattern for a shape; strides and offset are in elements.

    Supports ASCII named axes, composition, singleton axes, and ellipsis.
    Strides may be zero (broadcasting) or positive; negative strides are rejected.
    """
    if not operation.valid():
        raise Error("Unknown operation")
    if offset < 0:
        raise Error("Storage offset must be nonnegative")
    var sides = pattern.split("->")
    if len(sides) != 2:
        raise Error("A pattern must contain exactly one arrow")
    var left = _parse(String(sides[0]))
    var right = _parse(String(sides[1]), True)
    if operation == .rearrange and (
        left.anonymous_count > 0 or right.anonymous_count > 0
    ):
        raise Error("Nonunit anonymous axes require repeat or reduce")
    var ellipsis_count = 0
    if _has_ellipsis(left):
        ellipsis_count = len(shape) - len(left.groups) + 1
        if ellipsis_count < 0:
            raise Error("Input rank does not match the pattern")
    elif _has_ellipsis(right):
        raise Error("Output ellipsis requires input ellipsis")
    left = _expand(left, ellipsis_count, True)
    right = _expand(right, ellipsis_count, False)
    if len(left.groups) != len(shape):
        raise Error("Input rank does not match the pattern")

    var names = List[String]()
    var ids = Dict[String, Int]()
    for g in range(len(left.groups)):
        for a in range(len(left.groups[g])):
            var name = left.groups[g][a].copy()
            ids[name] = len(names)
            names.append(name.copy())
    var input_axes_count = len(names)
    for g in range(len(right.groups)):
        for a in range(len(right.groups[g])):
            var name = right.groups[g][a].copy()
            if name not in ids:
                if operation != .repeat:
                    raise Error("Only repeat may introduce new named axes")
                ids[name] = len(names)
                names.append(name.copy())

    var recipe = _RecipeData()
    recipe.operation = operation
    recipe.input_offset = offset
    recipe.input_shape = List(shape)
    recipe.input_size = _product(shape)
    recipe.input_axis_count = input_axes_count
    recipe.axis_sizes = List[Int](length=len(names), fill=-1)
    recipe.source_strides = List[Int](length=len(names), fill=0)
    for axis in range(len(names)):
        if names[axis].startswith("#"):
            recipe.axis_sizes[axis] = _integer(
                String(names[axis].split(":")[1])
            )
    for item in axes_lengths.items():
        var name = item.key.copy()
        var size = item.value
        if name not in ids:
            raise Error("An axis length was supplied for an unknown axis")
        if size < 0:
            raise Error("Axis lengths must be nonnegative")
        recipe.axis_sizes[ids[name]] = size

    for g in range(len(left.groups)):
        var unknown = -1
        var empty = False
        for name in left.groups[g]:
            var axis = ids[name]
            if recipe.axis_sizes[axis] < 0:
                if unknown != -1:
                    raise Error(
                        "A composed input axis has more than one unknown length"
                    )
                unknown = axis
            else:
                empty = empty or recipe.axis_sizes[axis] == 0
        if shape[g] == 0:
            if unknown != -1:
                if empty:
                    raise Error("Cannot infer a composed axis length")
                recipe.axis_sizes[unknown] = 0
            elif not empty:
                raise Error(
                    "Provided axis lengths do not match the input shape"
                )
            continue
        var known_product = 0 if empty else 1
        for name in left.groups[g]:
            var size = recipe.axis_sizes[ids[name]]
            if size >= 0:
                known_product = _multiply(known_product, size)
        if unknown != -1:
            if known_product == 0 or shape[g] % known_product != 0:
                raise Error("Cannot infer a composed axis length")
            recipe.axis_sizes[unknown] = shape[g] // known_product
        elif known_product != shape[g]:
            raise Error("Provided axis lengths do not match the input shape")
    for axis in range(input_axes_count, len(names)):
        if recipe.axis_sizes[axis] < 0:
            raise Error("Every new repeat axis needs an explicit length")

    var physical_strides = strides.copy()
    if len(physical_strides) == 0:
        physical_strides = List[Int](length=len(shape), fill=0)
        if recipe.input_size > 0:
            var stride = 1
            for g in range(len(shape) - 1, -1, -1):
                physical_strides[g] = stride
                stride = _multiply(stride, shape[g])
    if len(physical_strides) != len(shape):
        raise Error("Stride rank does not match shape rank")
    recipe.required_storage = _storage_extent(
        recipe.input_size, shape, Span(physical_strides), offset
    )
    for g in range(len(shape)):
        if physical_strides[g] < 0:
            raise Error("Negative strides are not supported yet")
        var stride = physical_strides[g]
        for a in range(len(left.groups[g]) - 1, -1, -1):
            var axis = ids[left.groups[g][a]]
            recipe.source_strides[axis] = stride if recipe.input_size > 0 else 0
            if recipe.input_size > 0:
                stride = _multiply(stride, recipe.axis_sizes[axis])

    var output_seen = Dict[Int, Bool]()
    for g in range(len(right.groups)):
        var first_axis = len(recipe.output_axes)
        for a in range(len(right.groups[g])):
            var axis = ids[right.groups[g][a]]
            recipe.output_axes.append(axis)
            output_seen[axis] = True
        recipe.output_shape.append(
            _axis_product(
                Span(recipe.axis_sizes), Span(recipe.output_axes)[first_axis:]
            )
        )
    recipe.output_size = _product(Span(recipe.output_shape))
    for axis in range(input_axes_count):
        if axis not in output_seen:
            if operation != .reduce:
                raise Error("Only reduce may remove named axes")
            recipe.reduced_axes.append(axis)
    recipe.reduction_size = _axis_product(
        Span(recipe.axis_sizes), Span(recipe.reduced_axes)
    )
    _validate_recipe(recipe)
    return TransformRecipe(recipe^)


def _mark_axis(
    axis: Int, first: Span[Int, _], second: Span[Int, _], mut seen: UInt64
) raises:
    if axis < 64:
        var bit = UInt64(1) << UInt64(axis)
        if seen & bit != 0:
            raise Error("Malformed or duplicate recipe axis")
        seen |= bit
    else:
        for previous in first:
            if previous == axis:
                raise Error("Malformed or duplicate recipe axis")
        for previous in second:
            if previous == axis:
                raise Error("Malformed or duplicate recipe axis")


def _validate_recipe(recipe: _RecipeData) raises:
    """Validate internal metadata before unchecked pointer access; no allocation.
    """
    var count = len(recipe.axis_sizes)
    if (
        count != len(recipe.source_strides)
        or recipe.input_axis_count < 0
        or recipe.input_axis_count > count
        or recipe.input_offset < 0
    ):
        raise Error("Malformed recipe metadata")
    if not recipe.operation.valid():
        raise Error("Unknown operation")
    for i in range(count):
        if recipe.axis_sizes[i] < 0 or recipe.source_strides[i] < 0:
            raise Error("Malformed recipe axis or stride")
        if i >= recipe.input_axis_count and (
            recipe.operation != .repeat or recipe.source_strides[i] != 0
        ):
            raise Error("Malformed introduced axis")
    var input_size = _product(
        Span(recipe.axis_sizes)[: recipe.input_axis_count]
    )
    var seen = UInt64(0)
    var output_axes = Span(recipe.output_axes)
    var reduced_axes = Span(recipe.reduced_axes)
    for i in range(len(output_axes)):
        var axis = output_axes[i]
        if axis < 0 or axis >= count:
            raise Error("Malformed output axis")
        _mark_axis(axis, output_axes[:i], reduced_axes[:0], seen)
    for i in range(len(reduced_axes)):
        var axis = reduced_axes[i]
        if (
            axis < 0
            or axis >= recipe.input_axis_count
            or recipe.operation != .reduce
        ):
            raise Error("Malformed reduced axis")
        _mark_axis(axis, output_axes, reduced_axes[:i], seen)
    if len(output_axes) + len(reduced_axes) != count:
        raise Error("Unrepresented recipe axis")
    var output_size = _axis_product(Span(recipe.axis_sizes), output_axes)
    var reduction_size = _axis_product(Span(recipe.axis_sizes), reduced_axes)
    if (
        output_size != recipe.output_size
        or _product(Span(recipe.output_shape)) != recipe.output_size
        or reduction_size != recipe.reduction_size
        or input_size != recipe.input_size
        or _product(Span(recipe.input_shape)) != recipe.input_size
    ):
        raise Error("Recipe extents disagree")
    var required = _storage_extent(
        input_size,
        Span(recipe.axis_sizes)[: recipe.input_axis_count],
        Span(recipe.source_strides)[: recipe.input_axis_count],
        recipe.input_offset,
    )
    if required != recipe.required_storage:
        raise Error("Recipe storage extent disagrees")
