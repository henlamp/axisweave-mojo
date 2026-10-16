# Axisweave

[![CodeQL](https://github.com/henlamp/axisweave-mojo/actions/workflows/codeql.yml/badge.svg)](https://github.com/henlamp/axisweave-mojo/actions/workflows/codeql.yml)

Named-axis tensor operations in native Mojo: rearrange, repeat, reduce,
pack/unpack and contract. Borrowed views reorder indexing without copying.

## Install

With [Pixi](https://pixi.sh):

```sh
pixi init my-project
cd my-project
pixi workspace channel add https://repo.prefix.dev/modular-community https://conda.modular.com/max
pixi add axisweave-mojo
```

Save as `example.mojo`, then run `pixi run mojo run example.mojo`:

```mojo
from std.collections import ImmSpan, List
from axisweave import StaticRecipe

def main() raises:
    var values: List[Float32] = [0, 1, 2, 3, 4, 5]
    var transposed = StaticRecipe["row column -> column row", 2, 3]().view(
        ImmSpan(values)
    )
    print(transposed.materialize())  # [0, 3, 1, 4, 2, 5]
```

Version 0.1.0 packages use Mojo 1.1.0 on macOS ARM64. Source checks also cover
1.2.0.dev2026100905; Linux is experimental. CPU operations retain Float32/Float64.
`axisweave.gpu` adds explicit Float32 device-buffer transforms, reductions and binary
contractions using MAX Core 26.6.0. Apple Metal is hardware-tested; NVIDIA is
cross-compiled and awaits hardware testing. Float64 stays on the CPU on Apple.
See the [GPU sensor example](examples/gpu_sensor_features.mojo).
Framework adapters and autograd are outside the API.

[API](docs/libraries/axisweave.md) · [Compatibility](docs/COMPATIBILITY.md) ·
[Checks](docs/VALIDATION.md). Run `uv run python tools/check_all.py` to verify the checkout.

[MIT](LICENSE). Inspired by einops; see [notices](THIRD_PARTY_NOTICES.md)
and its unchanged [license](UPSTREAM_LICENSE).
