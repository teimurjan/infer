// The compare crate is a command line tool and a library. The engine tests use the library to read golden files.

// diff.rs compares two lists of values.
pub mod diff;
// error.rs lists every way the tool can fail.
pub mod error;
// float16.rs turns float16 bits into f32.
mod float16;
// safetensors.rs reads the file format by hand.
pub mod safetensors;
