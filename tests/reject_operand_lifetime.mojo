from std.origin import ImmStaticOrigin
from std.collections import Array, ImmSpan, List, Span

def dangling() -> Array[ImmSpan[Float32, ImmStaticOrigin], 1]:
    var data: List[Float32] = [1,2,3]
    var operands: Array[ImmSpan[Float32, origin_of(data)], 1] = [Span(data).as_imm()]
    return operands

def main():
    _ = dangling()
