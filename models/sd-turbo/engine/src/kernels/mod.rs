// add.rs connects an in-place add of one slot into another.
mod add;
// concat.rs joins two tensors along the channels.
mod concat;
// geglu.rs connects GEGLU, value times GELU(gate).
mod geglu;
// gelu.rs connects GELU in place.
mod gelu;
// group_norm.rs connects GroupNorm (with an optional SiLU) between two slots.
mod group_norm;
// layer_norm.rs connects LayerNorm over the channels of each token row.
mod layer_norm;
// transpose_add.rs turns token rows back into channels and adds the transformer input.
mod transpose_add;
// upsample.rs connects the nearest 2x upsample to slots.
mod upsample;
// weight.rs reads the scale and the shift of a norm.
mod weight;

// The parts that the UNet blocks use.
pub(crate) use add::bind_add;
pub(crate) use concat::bind_concat;
pub(crate) use geglu::bind_geglu;
pub(crate) use gelu::bind_gelu;
pub(crate) use group_norm::{NormLayout, bind_group_norm_into};
pub(crate) use layer_norm::bind_layer_norm;
pub(crate) use transpose_add::bind_transpose_add;
pub(crate) use upsample::bind_upsample;
