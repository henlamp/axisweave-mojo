# SPDX-License-Identifier: MIT
"""Prepared Float32 transpose: CPU, queued GPU, and GPU with explicit transfers."""
from max.gpu.host import DeviceContext
from std.collections import List, Span
from std.testing import assert_equal
from std.time import perf_counter
from axisweave import plan, execute_into
from axisweave.gpu import DeviceTransform


def main() raises:
    comptime ROWS = 512
    comptime COLS = 768
    comptime N = ROWS * COLS
    comptime ITERATIONS = 20
    var ctx = DeviceContext()
    var shape: List[Int] = [ROWS, COLS]
    var recipe = plan("row col -> col row", Span(shape))
    var gpu = DeviceTransform(ctx, recipe)
    var source = List[Float32](length=N, fill=0)
    var target = List[Float32](length=N, fill=0)
    var host_input = ctx.enqueue_create_host_buffer[.float32](N)
    var host_output = ctx.enqueue_create_host_buffer[.float32](N)
    var input = ctx.enqueue_create_buffer[.float32](N)
    var output = ctx.enqueue_create_buffer[.float32](N)
    ctx.synchronize()
    for i in range(N):
        source[i] = Float32(i % 97 - 48) / 32
        host_input[i] = source[i]
    ctx.enqueue_copy(src_buf=host_input, dst_buf=input)
    for _ in range(3):
        execute_into(recipe, Span(source), Span(target))
        gpu.execute_into(input, output)
    ctx.synchronize()
    var cpu = List[Float64]()
    var resident = List[Float64]()
    var transferred = List[Float64]()
    for trial in range(5):
        source[0] = Float32(trial) / 32
        host_input[0] = source[0]
        ctx.enqueue_copy(src_buf=host_input, dst_buf=input)
        ctx.synchronize()
        for pass_index in range(3):
            var mode = (trial + pass_index) % 3
            var start = perf_counter()
            for _ in range(ITERATIONS):
                if mode == 0:
                    execute_into(recipe, Span(source), Span(target))
                else:
                    if mode == 2:
                        ctx.enqueue_copy(src_buf=host_input, dst_buf=input)
                    gpu.execute_into(input, output)
                    if mode == 2:
                        ctx.enqueue_copy(src_buf=output, dst_buf=host_output)
            if mode != 0:
                ctx.synchronize()
            var elapsed = (perf_counter() - start) / ITERATIONS
            if mode == 0:
                cpu.append(elapsed)
            elif mode == 1:
                resident.append(elapsed)
            else:
                transferred.append(elapsed)
        ctx.enqueue_copy(src_buf=output, dst_buf=host_output)
        ctx.synchronize()
        for row in range(ROWS):
            for col in range(COLS):
                var at = col * ROWS + row
                assert_equal(host_output[at], source[row * COLS + col])
                assert_equal(target[at], host_output[at])
    print(
        String(
            '{"cpu_seconds":',
            cpu,
            ',"gpu_resident_seconds":',
            resident,
            ',"gpu_with_transfers_seconds":',
            transferred,
            ',"shape":[',
            ROWS,
            ",",
            COLS,
            '],"iterations":',
            ITERATIONS,
            ',"trials":5}',
        )
    )
