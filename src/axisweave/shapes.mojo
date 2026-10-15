# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Named dimensions for explicit flat-buffer shapes."""

from std.collections import Dict, Span
from .pattern import _check_name, _product, _numeric, _integer


def parse_shape(
    shape: Span[Int, _], pattern: String
) raises -> Dict[String, Int]:
    """Map named axes to lengths; _ and ... skip dimensions, 1 checks a singleton.
    """
    _ = _product(shape)
    var tokens = pattern.split()
    var ellipsis = -1
    for i in range(len(tokens)):
        if tokens[i] == "...":
            if ellipsis >= 0:
                raise Error("A shape pattern permits only one ellipsis")
            ellipsis = i
    var fixed = len(tokens) - (1 if ellipsis >= 0 else 0)
    if len(shape) < fixed or (ellipsis < 0 and len(shape) != fixed):
        raise Error("Shape rank does not match the pattern")
    var result = Dict[String, Int]()
    var index = 0
    for token in tokens:
        var name = String(token)
        if name == "...":
            index += len(shape) - fixed
            continue
        if name != "_":
            _check_name(name)
        if _numeric(name):
            if shape[index] != _integer(name):
                raise Error("Numeric shape axis length disagrees")
        elif name != "_":
            _check_name(name)
            if name in result:
                raise Error("Duplicate shape axis")
            result[name] = shape[index]
        index += 1
    return result^
