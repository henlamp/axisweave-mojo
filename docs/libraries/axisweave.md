# Axisweave

Native CPU tensor layouts, packing, named contractions and linear inference for Mojo 1.1.
The pattern vocabulary follows [einops](https://github.com/arogozhnikov/einops).
Axisweave is an independent implementation with explicit buffers and ownership.

## Static use

```mojo
from std.collections import List, Span
from axisweave import StaticRecipe


def main() raises:
    var values: List[Float64] = [0, 1, 2, 3, 4, 5]
    var transpose = StaticRecipe["height width -> width height", 2, 3]()
    var borrowed = transpose.view(Span(values))
    print(borrowed.shape())
    print(borrowed.materialize())
    var pooling = StaticRecipe["height width -> height", 2, 3, operation=.reduce]()
    print(pooling.execute[.mean](Span(values)))
```

Patterns and variadic dimensions are compile-time parameters. Named split/repeat
lengths use `axes="patch=2 copies=3"`. Parsing, validation, shape inference and
stride planning run at compile time. Invalid static patterns fail compilation.
Static views borrow contiguous input and store no heap shape lists. SIMD loops
use runtime trip counts even for static shapes, so large dimensions do not
fully unroll into proportional compiler IR.

## Dynamic use

`plan(pattern, shape, operation=.rearrange, axes_lengths=..., strides=..., offset=...)`
returns an owned `TransformRecipe`. `execute` allocates output; `execute_into`
writes to an exact-size caller buffer. `rearrange`, `repeat` and
`reduce(..., .mean)` combine planning and execution. `view`, `rearrange_view` and
`repeat_view` borrow input storage.

Static recipes use `execute[.mean](source)` and `execute_into[.mean](source, output)`.
Dynamic recipes use `execute(recipe, source, .mean)` and
`execute_into(recipe, source, output, .mean)`. Both APIs take typed operations
and reductions; String-based adapters are removed. Recipe metadata is checked
before execution and view construction. `input_shape()` and `output_shape()`
borrow immutable dimensions for the recipe lifetime. `input_size()` and
`output_size()` return element counts. Internal fields have an underscore prefix;
public operations revalidate them without allocating scratch lists.
`TensorView.materialize_into(output)` and `StaticTensorView.materialize_into(output)`
write a view into disjoint, exact-size caller storage.
`TensorView.shape()` and `PackRecipe.input_shape(i)` also borrow immutable
dimensions. Use `List(...)` when an independent mutable copy is needed.

## Supported surface

- ASCII named axes, composition, singleton `1`/`()`, ellipsis and scalar tensors.
  Numeric axes above 1 are supported for repeat/reduce, including grouped pooling.
- Zero-length dimensions and nonnegative dynamic strides, including broadcast
  strides; strides/offsets count elements. Negative strides are rejected.
  Zero products are independent of axis order; unused empty-storage strides are
  canonicalized to zero. Each logical output dimension must still fit `Int`.
- Reorder/repeat over native scalar DTypes and Bool. Packing and views preserve
  defaultable, trivially register-passable elements, including native records.
- Float32/Float64 sum, mean, min, max and product reductions, including empty
  dimensions and NaNs. On a mixed signed-zero tie, minimum returns negative
  zero and maximum positive zero; this deterministic rule can differ from
  NumPy's CPU-dependent tie result. Reductions are materialized operations.
- Borrowed views with checked logical indices and compiler-tracked origins.

Input values are borrowed immutably. Allocating execution validates the source,
recipe and reduction before creating output storage, including empty min/max
reductions on both static and dynamic paths. Source and destination buffers
must not overlap: tracked aliases are rejected by the compiler; runtime address
checks also reject full or partial overlap in externally supplied spans before
writing. Adjacent disjoint buffers are permitted. `execute_into` requires the
exact output length. Unicode names, negative strides and framework/device
adapters are unsupported. Supported integer, Bool and callback reductions are
documented below.

## Verification

The transform suite covers 209 dynamic and 205 static cases, 21 rejected
patterns, SIMD tails, storage/metadata errors, and lifetime rejection. All three
einops examples run from a freshly compiled package. Static/dynamic CPU
measurements are in `benchmarks/local-results.json`; static view construction
can be optimized away and its timing is not a throughput speedup.

## Pack and unpack

`packing_plan(shapes, "batch * feature")` stores a reusable plan. Inputs share
named prefix/suffix dimensions but may have different ranks and dimensions at
`*`. `PackRecipe.pack` accepts either owned flat buffers or borrowed immutable
spans. `pack_into` writes to an exact-size disjoint destination. Packing validates every
buffer before allocating or writing. Contiguous blocks use `Span.copy_from`;
strided Float32/Float64 blocks retain SIMD copies. Bool and boolean
Scalar buffers keep their distinct native representations.

`packed_shapes()` copies the dimensions captured by `*`. `unpack` returns owned
copies. `unpack_into(source, destinations)` writes to mutable caller spans.
Destinations must have the planned lengths and be pairwise disjoint and disjoint
from the source. Every check precedes any write. `unpack_views` returns borrowed
`TensorView` values, including strided
splits of a middle axis, without copying data or reparsing an axis pattern.
Empty views retain their shapes without computing unused storage strides. The
source must outlive those views. `unpacking_plan(shape, packed_shapes, pattern)`
permits one inferred `-1` and reuses its parsed axis layout directly;
ambiguous zero products and overfull/underfull splits are rejected.

```mojo
from std.collections import List, Span
from axisweave import packing_plan


def main() raises:
    var shapes: List[List[Int]] = [[2, 3], [2, 2, 3]]
    var values: List[List[Float32]] = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ]
    var recipe = packing_plan(Span(shapes), "batch * feature")
    var packed = recipe.pack(Span(values))
    print(recipe.output_shape())
    var sensors = recipe.unpack_views(Span(packed))
    print(sensors[1].shape())
    print(sensors[1][0])
```

## Bool masks and view types

Packing, unpacking and views use the native element type. Explicit annotations
use `TensorView[Float64, origin]` or `TensorView[Bool, origin]`;
`StaticTensorView` takes `Element=Float64` or `Element=Bool`. The public API
replaces the earlier alpha DType view parameter. An explicit pack type parameter is
likewise `pack[Float32](...)`; ordinary calls infer it from the buffer.

```mojo
from std.collections import List, Span
from std.testing import assert_equal
from axisweave import packing_plan, StaticRecipe


def main() raises:
    var shapes: List[List[Int]] = [[2], [2, 2]]
    var masks: List[List[Bool]] = [[True, False], [False, True, True, False]]
    var recipe = packing_plan(Span(shapes), "batch *")
    var packed = recipe.pack(Span(masks))
    assert_equal(recipe.unpack(Span(packed)), masks)
    var transposed = StaticRecipe["batch sensor -> sensor batch", 2, 3]().view(Span(packed))
    var output = List[Bool](length=6, fill=False)
    transposed.materialize_into(Span(output))
    var wanted: List[Bool] = [True, False, False, True, True, False]
    assert_equal(output, wanted)
```

## Named contractions and shape parsing

`einsum_plan(shapes, pattern)` caches row-major strides. `execute` and
`execute_into` accept an immutable span of contiguous Float32/Float64 buffers.
Patterns require an explicit `->` output. They support any positive operand
count, scalar operands/results, diagonal axes, shared-axis broadcasting and
right-aligned ellipsis broadcasting. SIMD runs along each final output axis;
zero-length contractions return additive zeros, including caller-provided output
buffers. A single reduction axis uses direct strides; multiple axes advance a
mixed-radix cursor. Non-reducing and size-one contractions need no scratch
buffers. Other contractions cache operand offsets; only multidimensional
reductions allocate coordinate storage. Buffers are allocated once per execution.

`parse_shape(shape, "batch ... _ feature")` returns named lengths for use in
other plans. `_` and ellipsis skip dimensions; numeric tokens check exact
positive lengths. Composite axes are not accepted by this shape helper.

```mojo
from std.collections import List, Span
from axisweave import einsum_plan, parse_shape


def main() raises:
    var shapes: List[List[Int]] = [[2, 3], [3, 2]]
    var values: List[List[Float64]] = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
    ]
    var recipe = einsum_plan(Span(shapes), "sample input, input output -> sample output")
    print(recipe.output_shape())
    print(recipe.execute(Span(values)))
    print(parse_shape(Span(shapes[0]), "sample input"))
```

The extension suite compares 54 packing cases, 54 inferred unpacking cases,
52 contractions, 64 named linear inference cases and 9 shape mappings against
actual einops/NumPy. All packing and contraction cases also exercise borrowed
operands; all packing cases check direct unpacking into caller storage. Native
contracts include exact Int64 values above 2^53, empty common dimensions,
overflowing unused strides in empty unpack views, external aliases, damaged
strides and high-axis duplicate detection. The
compiler rejects mutable shape access and escaping recipe/view shapes or
packed-data views. Native contracts also cover all permutations of large empty
shapes, grouped zero inference and overflowing logical dimensions.
The element-buffer suite adds 81 exact packing/unpacking cases and 18 dynamic
and static view cases for native Bool, boolean Scalar and UInt64. It checks
caller-buffer materialization, inferred splits and UInt64 maximum values.
Native contracts cover record fields, zero-size elements, alias and extent
errors before writes; three additional compile failures check Bool lifetimes.

Contraction accumulation uses ordinary floating-point arithmetic; it is not
arbitrary precision. Framework layers, autograd and GPU execution remain outside this CPU library.

## Borrowed operands

Pack plans accept `ImmSpan[ImmSpan[Element, origin], _]`; contractions accept
`ImmSpan[ImmSpan[Scalar[dtype], origin], _]`.
The descriptor collection can be a fixed `Array`; it need not allocate. The
compiler tracks the input lifetime, including an origin union for separate
buffers. Runtime overlap checks also cover spans obtained through foreign
pointers. The private adapters contain span descriptors only; they never copy
operand values or allocate descriptor lists.

```mojo
from std.collections import Array, ImmSpan, List, Span
from axisweave import packing_plan


def main() raises:
    var first: List[Float32] = [1, 2, 3]
    var second: List[Float32] = [4, 5]
    var buffers: Array[ImmSpan[Float32, origin_of(first, second)], 2] = [
        ImmSpan[Float32, origin_of(first, second)](first),
        ImmSpan[Float32, origin_of(first, second)](second),
    ]
    var shapes: List[List[Int]] = [[3], [2]]
    print(packing_plan(Span(shapes), "*").pack(Span(buffers)))
```

## Named linear inference

`mix_plan(input_shape, weight_shape, pattern)` prepares a named linear
contraction with two operands. `bias_shape=` and `bias_axes=` add a broadcast
bias. Bias axes are named output axes in the bias buffer's order; an empty
string selects a scalar bias. Input and weight patterns retain einsum's diagonal
and ellipsis rules. Bias cannot bind unnamed ellipsis axes.

`MixRecipe.execute(input, weights, bias=...)` allocates its output;
`execute_into(input, weights, output, bias=...)` reuses caller storage. All
operands are borrowed. Weights may change between calls without replanning.
Bias is added directly to the output; no full-size broadcast buffer is created.
Input, weight and bias lengths and all output overlap checks precede writes.
Float32 and Float64 use ordinary floating-point accumulation.

```mojo
from std.collections import List, Span
from axisweave import mix_plan


def main() raises:
    var input_shape: List[Int] = [2, 3]
    var weight_shape: List[Int] = [3, 2]
    var bias_shape: List[Int] = [2]
    var input: List[Float32] = [1, 2, 3, 4, 5, 6]
    var weights: List[Float32] = [1, 0, 0, 1, 1, 1]
    var bias: List[Float32] = [10, 20]
    var recipe = mix_plan(
        Span(input_shape), Span(weight_shape),
        "batch feature, feature output -> batch output",
        bias_shape=Span(bias_shape), bias_axes="output",
    )
    print(recipe.execute(Span(input), Span(weights), bias=Span(bias)))
```

This supports inference with supplied parameters. It does not initialize
weights, compute gradients, update parameters or register a framework layer.
It shares EinMix's named-axis motivation without reproducing its training API.

`tests/buffer-benchmarks.json` records matched owned/borrowed input timings and
adapter sizes. Both paths use prepared plans and caller outputs. Contractions
can still allocate reduction scratch; allocating APIs also allocate their
results. These measurements do not establish a general application speedup.

## Logical and fixed-width reductions

`Reduction.any` and `Reduction.all` accept native `Bool` buffers and
`Scalar[.bool]` buffers. Empty rows produce False for any and True for all.
`execute`, `execute_into` and `StaticRecipe.execute[.any/.all]` share these
identities. Native Bool rearrangements and repetitions use the same pattern
planner and checked storage boundaries.

Int32, Int64, UInt32 and UInt64 support sum, product, minimum and maximum.
The output retains the input DType. Sum and product wrap modulo its bit width;
minimum and maximum compare exact integers. This deliberately explicit policy
can differ from NumPy's default promotion of narrow integer sums. Integer means
are rejected. Float32/Float64 keep their existing floating-point behavior.

```mojo
from std.collections import List, Span
from axisweave import StaticRecipe


def main() raises:
    var terminated: List[Bool] = [False, True, False, False, False, False]
    var episodes = StaticRecipe["environment step -> environment", 2, 3, operation=.reduce]()
    print(episodes.execute[.any](Span(terminated)))
    var counts: List[Int64] = [9007199254740993, 1, 2, 3, 4, 5]
    print(episodes.execute[.sum](Span(counts)))
```

## Custom reductions

`reduce_custom(recipe, values, reducer)` and
`reduce_custom_into(recipe, values, output, reducer)` accept a native runtime
function or closure. `reduce(values, shape, pattern, reducer=...)` also performs
planning. The callback signature is generic over the scalar DType and the two
borrow origins. It receives `ReductionValues`: a checked, read-only row in the
plan's reduced-axis order. No dense row copies or temporary index lists are
created. Output retains the source Scalar DType.
`ReductionValues(values, recipe, row)` borrows a public `TransformRecipe` and
checks the selected row and storage. Both borrows must outlive the row view.

```mojo
from std.collections import List, Span
from axisweave import ReductionValues, plan, reduce_custom


def variance[d: DType, s: Origin[mut=False], p: Origin[mut=False]](
    values: ReductionValues[d, s, p],
) raises -> Scalar[d]:
    var mean = Float64(0)
    var moment = Float64(0)
    for i in range(len(values)):
        var x = Float64(values[i])
        var delta = x - mean
        mean += delta / Float64(i + 1)
        moment += delta * (x - mean)
    return Scalar[d](moment / Float64(len(values)))


def main() raises:
    var shape: List[Int] = [2, 3]
    var values: List[Float32] = [1, 2, 3, 4, 5, 6]
    var recipe = plan("environment sample -> environment", Span(shape), .reduce)
    print(reduce_custom(recipe, Span(values), variance))
```

Callbacks declare ordinary `raises`, matching checked row indexing. Captured
state uses Mojo 1.1's runtime closure syntax. Storage, metadata, destination
length and overlap errors are checked before the callback runs. If the callback
raises after earlier rows completed, those rows remain written; the failing row
is not assigned. The allocating form returns no partial result on failure.
A callback decides the value for an empty reduction row. When the pattern removes
no axes, it is not called and the operation only rearranges/copies, matching
einops. Output with no elements also invokes no callback. Borrowed rows cannot
outlive their source and plan. Callable reductions currently use dynamic plans.

## GPU buffers

Import `DeviceTransform` and `DeviceEinsum` from `axisweave.gpu` and construct them
with a `max.gpu.host.DeviceContext` plus the corresponding CPU plan. Both expose
`output_shape()`, allocating `execute(...)` and caller-output `execute_into(...)`.
Transforms take one Float32 `DeviceBuffer`; binary contractions take two.
The context owns the work queue. Plans retain their metadata; source and output
buffers remain on device between operations. See [compatibility](../COMPATIBILITY.md#gpu-execution)
and the executable [sensor pipeline](../../examples/gpu_sensor_features.mojo).
