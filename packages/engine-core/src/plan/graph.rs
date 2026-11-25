// The lifetime of one tensor, which this file computes.
use super::TensorLife;

// Bytes in one f32 value. Every activation of the engine is f32.
const F32_BYTES: u64 = 4;

// A handle to one tensor of a graph: its position in the tensor list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorId(usize);

// The size of a tensor: three sizes, and 1 for each size that a tensor does not use.
// An image tensor is [channels, height, width]. A token tensor is [tokens, channels, 1]. A vector is [values, 1, 1].
pub type Dims = [u32; 3];

// One tensor: the name of the op that wrote it, and its size.
struct Tensor {
    // For example "down_blocks.0.resnets.0.conv1".
    name: String,
    // For example [320, 64, 64].
    dims: Dims,
}

// One op: the tensors it reads and the tensors it writes. The kernels come later. The plan only needs this.
struct Op {
    // The tensors that the op reads. An op that changes a tensor in place lists it here.
    reads: Vec<TensorId>,
    // The new tensors that the op writes.
    writes: Vec<TensorId>,
}

// A forward pass, op by op, in the order the engine runs them.
#[derive(Default)]
pub struct Graph {
    // Every tensor, in the order they were made.
    tensors: Vec<Tensor>,
    // Every op, in run order.
    ops: Vec<Op>,
    // Tensors that must live to the end, such as the network output.
    kept: Vec<TensorId>,
}

impl Graph {
    // An op that reads `reads` and writes one new tensor of size `dims`. The tensor gets the name of the op.
    pub fn op(&mut self, name: impl Into<String>, reads: &[TensorId], dims: Dims) -> TensorId {
        // The new tensor.
        let output = self.new_tensor(name.into(), dims);
        // The op that writes it.
        self.ops.push(Op { reads: reads.to_vec(), writes: vec![output] });
        // The handle for later ops.
        output
    }

    // An op that changes `target` in place with the values of `reads`, for example a residual add. No new tensor.
    pub fn update(&mut self, target: TensorId, reads: &[TensorId]) {
        // The op reads the target and the other tensors. That keeps all of them live at this op.
        self.ops.push(Op { reads: [target].iter().chain(reads).copied().collect(), writes: Vec::new() });
    }

    // Keep a tensor live until the last op, for example the output that the next stage reads.
    pub fn keep(&mut self, tensor: TensorId) {
        self.kept.push(tensor);
    }

    // The size of a tensor per dimension.
    pub fn dims(&self, tensor: TensorId) -> Dims {
        self.tensors[tensor.0].dims
    }

    // The name of a tensor.
    pub fn name(&self, tensor: usize) -> &str {
        &self.tensors[tensor].name
    }

    // The size of the tensor at position `tensor` of the tensor list.
    pub fn dims_at(&self, tensor: usize) -> Dims {
        self.tensors[tensor].dims
    }

    // The number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    // The number of ops.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    // The ids of the tensors that op `op` reads or writes. The plan tests of the models use it to check that they never share a slot.
    pub fn tensors_of_op(&self, op: usize) -> impl Iterator<Item = usize> + '_ {
        // The reads, then the writes, as positions in the tensor list.
        self.ops[op].reads.iter().chain(&self.ops[op].writes).map(|tensor| tensor.0)
    }

    // The name of the first tensor that op `op` writes, or of the tensor it changes in place.
    pub fn op_name(&self, op: usize) -> &str {
        // An in-place op has no writes, so name it by its target, which is its first read.
        let tensor = self.ops[op].writes.first().or(self.ops[op].reads.first());
        // Every op reads or writes something, so the fallback is never used.
        tensor.map_or("", |tensor| self.name(tensor.0))
    }

    // The lifetime of every tensor, in the order of the tensor list.
    // A tensor is live from the op that writes it to the last op that reads it. A tensor that nothing reads dies at its writer.
    pub fn lifetimes(&self) -> Vec<TensorLife> {
        // Start with each tensor live only at its writer. Every tensor has a writer, because only ops make tensors.
        let mut lives: Vec<TensorLife> = self.tensors.iter().map(|tensor| TensorLife { bytes: bytes_of(&tensor.dims), first: 0, last: 0 }).collect();
        // Walk the ops in run order.
        for (position, op) in self.ops.iter().enumerate() {
            // A written tensor starts here.
            op.writes.iter().for_each(|tensor| lives[tensor.0] = TensorLife { first: position, last: position, ..lives[tensor.0] });
            // A read tensor lives at least to here. Ops come in order, so the last read wins.
            op.reads.iter().for_each(|tensor| lives[tensor.0].last = position);
        }
        // The kept tensors live to the last op.
        let end = self.ops.len().saturating_sub(1);
        // Stretch each kept tensor to the end.
        self.kept.iter().for_each(|tensor| lives[tensor.0].last = end);
        // One lifetime per tensor.
        lives
    }

    // Add a tensor to the list and give its handle.
    fn new_tensor(&mut self, name: String, dims: Dims) -> TensorId {
        // Its position is the current length.
        let id = TensorId(self.tensors.len());
        // Store its name and size.
        self.tensors.push(Tensor { name, dims });
        // The handle.
        id
    }
}

// The bytes of an f32 tensor of this size. u64, so large tensors cannot overflow.
fn bytes_of(dims: &Dims) -> u64 {
    // The product of the dimensions, times 4 bytes.
    dims.iter().map(|&size| u64::from(size)).product::<u64>() * F32_BYTES
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the graph and the lifetime into the tests.
    use super::{Graph, TensorLife};

    // Each tensor must live from its writer to its last reader, and a kept tensor to the end.
    #[test]
    fn lifetimes_follow_the_reads() {
        // An empty graph.
        let mut graph = Graph::default();
        // Op 0 writes the input, 2 values = 8 bytes.
        let input = graph.op("input", &[], [2, 1, 1]);
        // Op 1 reads it and writes an output of 3 values.
        let hidden = graph.op("hidden", &[input], [3, 1, 1]);
        // Op 2 adds the input into the output in place.
        graph.update(hidden, &[input]);
        // Op 3 reads the output and writes the result, which lives to the end.
        let output = graph.op("output", &[hidden], [1, 1, 1]);
        // Op 4 is a last op that reads nothing.
        graph.op("other", &[], [1, 1, 1]);
        // Keep the result.
        graph.keep(output);
        // Tensor order: input, hidden, output, other.
        let expected = [
            TensorLife { bytes: 8, first: 0, last: 2 },
            TensorLife { bytes: 12, first: 1, last: 3 },
            TensorLife { bytes: 4, first: 3, last: 4 },
            TensorLife { bytes: 4, first: 4, last: 4 },
        ];
        // Compare.
        assert_eq!(graph.lifetimes(), expected);
    }
}
