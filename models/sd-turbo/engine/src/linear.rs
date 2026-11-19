// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A kernel with its buffers.
use crate::dispatch::BoundKernel;
// Where the bias comes from, and the bias reader of the slot layers.
use crate::conv::{Bias, slot_bias};
// The linear kernels, connected to slots.
use crate::layers::{LayerShape, LinearShape, VectorShape, bind_matrix_into, bind_vector_matrix_into};
// A tensor in a slot of the activation plan.
use crate::plan::SlotTensor;
// The weight buffers of the slot kernels.
use crate::weights::{MatrixWeight, WeightBuffers};

// nn.Linear on one row: input [in, 1, 1], output [out, 1, 1]. `silu_input` applies SiLU to the input as the kernel reads it.
// The weight can be float or packed.
pub(crate) fn bind_vector_linear(gpu: &Gpu, weights: &impl WeightBuffers, prefix: &str, input: &SlotTensor, silu_input: bool, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The input of a one-row layer is a vector.
    let [in_features, 1, 1] = input.dims else {
        return Err(EngineError::PlanMismatch { tensor: input.name.to_string(), planned: input.dims, actual: [input.dims[0], 1, 1] });
    };
    // The weight and its output count. Every one-row layer of the UNet has a bias.
    let (weight, out_features) = linear_weight(gpu, weights, prefix, in_features)?;
    let bias = slot_bias(gpu, weights, prefix, Bias::FromWeights, out_features as usize)?;
    // The output size must be the planned size.
    check_output(output, [out_features, 1, 1])?;
    // The sizes of this layer.
    let shape = VectorShape { in_features, out_features, silu_input };
    // Connect the slots, with the float or the packed kernel. This is the only bind group of the kernel.
    Ok(bind_vector_matrix_into(gpu, &shape, input.buffer, weight.matrix, bias.buffer(), output.buffer)?.keeping(bias.into_owned()))
}

// nn.Linear on every row of a token tensor: input [rows, in, 1], output [rows, out, 1]. The weight can be float or packed.
pub(crate) fn bind_rows_linear(gpu: &Gpu, weights: &impl WeightBuffers, prefix: &str, input: &SlotTensor, bias: Bias, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The input of a token layer has rows of in_features values.
    let [rows, in_features, 1] = input.dims else {
        return Err(EngineError::PlanMismatch { tensor: input.name.to_string(), planned: input.dims, actual: [input.dims[0], input.dims[1], 1] });
    };
    // The weight and its output count. to_q, to_k and to_v of the UNet have no bias.
    let (weight, out_features) = linear_weight(gpu, weights, prefix, in_features)?;
    let bias = slot_bias(gpu, weights, prefix, bias, out_features as usize)?;
    // The output size must be the planned size.
    check_output(output, [rows, out_features, 1])?;
    // The sizes of this layer.
    let shape = LayerShape::Linear(LinearShape { rows, in_features, out_features });
    // Connect the slots, with the float or the packed kernel.
    Ok(bind_matrix_into(gpu, &shape, input.buffer, weight.matrix, bias.buffer(), output.buffer)?.keeping(bias.into_owned()))
}

// Fail if the kernel would write another size than the plan has for `output`.
fn check_output(output: &SlotTensor, actual: [u32; 3]) -> Result<(), EngineError> {
    // Equal sizes are fine.
    if output.dims == actual {
        return Ok(());
    }
    // Show both sizes.
    Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual })
}

// Get `<prefix>.weight` [out, in] and check it against `in_features`. Returns the weight and the output count.
fn linear_weight<'a>(gpu: &Gpu, weights: &'a impl WeightBuffers, prefix: &str, in_features: u32) -> Result<(MatrixWeight<'a>, u32), EngineError> {
    // For example "time_embedding.linear_1.weight".
    let name = format!("{prefix}.weight");
    // The weight, as the weights give it.
    let weight = weights.matrix(gpu, &name)?;
    // PyTorch's layout is [out_features, in_features], and in_features must be the input length.
    let out_features = match weight.shape[..] {
        // The right rank and input length. A count that does not fit in u32 cannot come from a real model.
        [out_features, weight_in] if weight_in == in_features as usize => u32::try_from(out_features).map_err(|_| EngineError::WeightShape { name: name.clone(), shape: weight.shape.clone(), expected: "fewer than 2^32 outputs".to_string() })?,
        // Anything else cannot run on this input.
        _ => return Err(EngineError::WeightShape { name, shape: weight.shape, expected: format!("[out, {in_features}]") }),
    };
    // The weight and its output count.
    Ok((weight, out_features))
}
