"""Seeded independent transform properties; one native executable for the corpus."""

from __future__ import annotations

import hashlib
import random

import einops
import numpy as np
from _support import BUILD, FIXTURES, MOJO, ROOT, run, source_hash, write_report


def corpus(seed=20261010, count=1800):
    rng = random.Random(seed)
    values = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        rank = rng.randrange(5)
        shape = [rng.randrange(5) for _ in range(rank)]
        names = list("abcd"[:rank])
        operation = rng.choice(("rearrange", "repeat", "reduce"))
        kind = rng.choice(("sum", "mean", "min", "max", "prod"))
        hints = {}
        output = names.copy()
        rng.shuffle(output)
        if operation == "repeat":
            output.insert(rng.randrange(len(output) + 1), "copies")
            hints = {"copies": rng.randrange(4)}
        elif operation == "reduce":
            output = [axis for axis in output if rng.randrange(2)]
        if operation == "reduce" and kind in {"mean", "min", "max"}:
            removed = [i for i, axis in enumerate(names) if axis not in output]
            if any(shape[i] == 0 for i in removed):
                kind = "sum"
        if len(output) > 1 and rng.randrange(2):
            at = rng.randrange(len(output) - 1)
            output[at : at + 2] = ["(" + " ".join(output[at : at + 2]) + ")"]
        pattern = " ".join(names) + " -> " + " ".join(output)
        step = rng.choice((1, 2, 3))
        stride = step
        strides = [0] * rank
        for axis in reversed(range(rank)):
            strides[axis] = stride
            stride *= max(shape[axis], 1)
        if rank and rng.randrange(4) == 0:
            strides[rng.randrange(rank)] = 0
        offset = rng.randrange(3)
        required = (
            offset
            + 1
            + sum((dim - 1) * st for dim, st in zip(shape, strides, strict=True))
            if all(shape)
            else offset + 1
        )
        storage = values.integers(-3, 4, size=required).astype(np.float64)
        tensor = np.ndarray(
            tuple(shape),
            dtype=np.float64,
            buffer=storage,
            offset=offset * 8,
            strides=tuple(s * 8 for s in strides),
        )
        if operation == "reduce":
            expected = einops.reduce(tensor, pattern, kind, **hints)
        else:
            expected = getattr(einops, operation)(tensor, pattern, **hints)
        entry = [
            "valid",
            operation,
            kind,
            pattern,
            shape,
            hints,
            strides,
            offset,
            storage,
            list(expected.shape),
            expected.reshape(-1),
            str(index),
        ]
        cases.append(entry)
        if index % 3 == 0:
            broken = entry.copy()
            broken[0] = "invalid"
            broken[3] = pattern.replace("->", "-> ->")
            broken[-1] += "-arrow"
            cases.append(broken)
        if index % 5 == 0 and storage.size > 1 and all(shape):
            broken = entry.copy()
            broken[0] = "invalid"
            broken[8] = storage[:offset]
            broken[-1] += "-storage"
            cases.append(broken)
        if index % 7 == 0 and rank:
            broken = entry.copy()
            broken[0] = "invalid"
            broken[6] = [-1, *strides[1:]]
            broken[-1] += "-stride"
            cases.append(broken)
    return cases


def encode(entry):
    text = []
    for index, value in enumerate(entry):
        if index in {4, 6, 9}:
            text.append(",".join(str(int(v)) for v in value))
        elif index in {8, 10}:
            text.append(",".join(format(float(v), ".17g") for v in value))
        elif index == 5:
            text.append(",".join(f"{k}={v}" for k, v in value.items()))
        else:
            text.append(str(value))
    return "|".join(text)


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    cases = corpus()
    path = FIXTURES / "property-transforms.txt"
    path.write_text("\n".join(encode(case) for case in cases) + "\n")
    executable = BUILD / "property-transforms"
    run(
        [
            MOJO,
            "build",
            "--Werror",
            "-I",
            ROOT / "src",
            ROOT / "tests/property_transforms.mojo",
            "-o",
            executable,
        ],
        "property-transforms-build",
        timeout=300,
    )
    print(run([executable, path], "property-transforms-run", timeout=180))
    write_report(
        ROOT / "tests/properties-verification.json",
        {
            "status": "passed",
            "seed": 20261010,
            "valid_cases": sum(case[0] == "valid" for case in cases),
            "invalid_cases": sum(case[0] == "invalid" for case in cases),
            "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sha256": source_hash("axisweave"),
            "einops": einops.__version__,
            "numpy": np.__version__,
        },
    )


if __name__ == "__main__":
    main()
