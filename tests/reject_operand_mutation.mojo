from std.collections import Array, ImmSpan, List, Span

def main():
    var data: List[Float32] = [1,2,3]
    var operands: Array[ImmSpan[Float32, origin_of(data)], 1] = [Span(data).as_imm()]
    operands[0][0] = 7
