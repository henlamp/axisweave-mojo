from std.origin import MutUntrackedOrigin
from std.collections import Array, List, Span

def dangling() -> Array[Span[mut=True, Float32, MutUntrackedOrigin], 1]:
    var data: List[Float32] = [1,2,3]
    var outputs: Array[Span[mut=True, Float32, origin_of(data)], 1] = [Span(data)]
    return outputs

def main():
    _ = dangling()
