# Compatibility

The source API targets Axisweave on Mojo 1.1.0, macOS ARM64
and experimental Linux x86-64. The compiler version is an exact dependency. A `.mojoc`
artifact belongs to the compiler that built it; another compiler needs a rebuild
and its own acceptance run.

## Axisweave

The contract includes named rearrangement/repetition, numeric and logical
reductions, native custom row callbacks, pack/unpack and inferred splits,
read-only dynamic/static views, named contractions, shape parsing and linear
inference with caller-owned weights and optional bias.

Float32/Float64 support the five numeric reductions and contractions. Int32,
Int64, UInt32 and UInt64 support sum/product/min/max; sum and product wrap at
the input width and integer mean is rejected. Bool and boolean Scalar support
any/all. Packing and views preserve defaultable, trivially register-passable
elements. Arithmetic on arbitrary records, autograd and framework adapters are outside this contract.

Shapes use nonnegative Int extents. Dynamic strides are nonnegative and count
elements. Empty products are independent of axis order. Logical axes and byte
extents must fit the checked integer bounds. Plans own metadata; views borrow
source storage through tracked origins. Returned shape spans are immutable.
Underscore-prefixed fields/modules and raw metadata constructors are internal.
Use the documented planning functions to construct dynamic recipes.

Caller outputs have exact planned lengths. Source/output overlap and overlapping
unpack destinations are rejected before writes. Structural/input failures leave
outputs unchanged. Custom callbacks may retain completed rows when a later
callback raises; the failing row remains unchanged. Non-reducing custom plans
retain identity semantics and do not call the reducer. Floating-point reductions
use the documented NaN and signed-zero rules, without a reproducibility promise
across different execution architectures.

## GPU execution

`axisweave.gpu.DeviceTransform` accepts an existing `TransformRecipe` and implements
Float32 rearrange/repeat and sum/mean/min/max/prod. `DeviceEinsum` accepts a binary
`EinsumRecipe`, including diagonals, ellipsis and broadcasting. Packing, custom
callbacks, logical/integer operations and contractions with other operand counts
remain CPU APIs. GPU signatures accept only Float32 buffers; no Float64 narrowing
or automatic CPU fallback occurs. Apple Float64 remains in the CPU API.

GPU execution requires matching MAX Core (26.6.0 for Mojo 1.1.0), Metal or CUDA.
Plan construction uploads rank-sized metadata and synchronizes once. Calls accept
buffers from that exact context and enqueue work without host transfers. Synchronize
before host access. Input storage must stay unchanged until queued work completes.
Exact output lengths and source/output overlap, including sub-buffers, are checked
before writes. Output grids must fit 2^31-1 blocks of 128 threads. A driver or kernel failure can leave output incomplete.

Apple M4 Pro Float32 has CPU/NumPy parity checks. NVIDIA sm_80/sm_90/sm_120 has PTX
compilation checks; numerical execution and performance there remain unverified.
Floating-point reductions/contractions can differ by rounding between backends;
NaN and signed-zero rules match the CPU contract. Kernels parallelize output rows;
large reductions and matrix multiplication are not tuned vendor implementations.
Validated small layouts use 32-bit indexing; larger metadata retains the 64-bit path.

## Compatibility

Patch releases fix documented behavior without changing signatures, ownership,
error categories or accepted inputs. Minor releases add compatible public APIs;
during 0.x development, breaking changes require a new minor version and
migration instructions. Within 0.1.x, the reviewed contract remains compatible.
Security fixes may require a compatibility change; publish its impact and migration instructions. Undocumented
implementation details and internal symbols carry no compatibility promise.

`tests/api` records public declarations and exports. `check_api.py` rejects drift;
baseline changes require a compatibility review.
