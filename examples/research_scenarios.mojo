"""Synthetic research analysis: scenario/order/metric layouts and missing data.

This demonstrates data preparation only; it is not a routing or fleet solver.
"""

from std.collections import List, Span
from std.testing import assert_equal
from axisweave import StaticRecipe


def main() raises:
    # Three scenarios, three orders, two metrics (duration and operating cost).
    # 999 is deliberately invalid filler; the separate validity mask decides
    # which observations exist. An unobserved order must not become a zero mean.
    var observations: List[Float64] = [
        10,
        20,
        12,
        24,
        999,
        999,
        14,
        28,
        999,
        999,
        999,
        999,
        999,
        999,
        16,
        32,
        999,
        999,
    ]
    var valid: List[Float64] = [1, 1, 0, 1, 0, 0, 0, 1, 0]
    var layout = StaticRecipe[
        "scenario order metric -> order scenario metric", 3, 3, 2
    ]()
    var by_order = layout.view(Span(observations))
    print("order/scenario/metric shape:", by_order.shape())
    var mask_recipe = StaticRecipe[
        "scenario order -> scenario order metric",
        3,
        3,
        operation=.repeat,
        axes="metric=2",
    ]()
    var expanded_mask = mask_recipe.view(Span(valid))
    var weighted = List[Float64](length=len(observations), fill=0)
    for i in range(len(observations)):
        if expanded_mask[i] == 1:
            weighted[i] = observations[i]
    var sum_recipe = StaticRecipe[
        "scenario order metric -> order metric", 3, 3, 2, operation=.reduce
    ]()
    var totals = sum_recipe.execute[.sum](Span(weighted))
    var count_recipe = StaticRecipe[
        "scenario order -> order", 3, 3, operation=.reduce
    ]()
    var counts = count_recipe.execute[.sum](Span(valid))
    var expected_totals: List[Float64] = [24, 48, 28, 56, 0, 0]
    var expected_counts: List[Float64] = [2, 2, 0]
    assert_equal(totals, expected_totals)
    assert_equal(counts, expected_counts)
    for order in range(3):
        if counts[order] == 0:
            print("order", order, "has no observations")
        else:
            print(
                "order",
                order,
                "observations",
                counts[order],
                "mean duration",
                totals[2 * order] / counts[order],
                "mean cost",
                totals[2 * order + 1] / counts[order],
            )
