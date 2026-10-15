# SPDX-FileCopyrightText: 2026 henlamp
# SPDX-License-Identifier: MIT

"""Native Mojo tensor transformations inspired by Python einops."""

from .pattern import TransformRecipe, plan
from .operations import (
    execute,
    execute_into,
    rearrange,
    reduce,
    repeat,
)
from .view import TensorView, view, rearrange_view, repeat_view
from .kinds import Operation, Reduction
from .static import StaticRecipe, StaticTensorView

from .packing import PackRecipe, packing_plan, unpacking_plan, pack, unpack
from .shapes import parse_shape
from .contraction import EinsumRecipe, einsum_plan, einsum
from .mixing import MixRecipe, mix_plan
from .custom import ReductionValues, reduce_custom, reduce_custom_into
