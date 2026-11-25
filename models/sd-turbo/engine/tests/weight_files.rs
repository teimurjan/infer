// The weight index of the shared engine base, against the real SD-Turbo weight files.
// It lives here, not in packages/engine-core, because it reads the files of this model.

// The index parser of the shared engine base.
use engine_core::weights::WeightIndex;

// In the real files, each tensor must end in the 256 bytes before the next one starts.
// convert_weights.py pads only to the next multiple of 256, so a wrong size rule shows up as a larger gap or an overlap.
#[test]
fn stored_lengths_match_the_real_files() {
    // The weights folder is in the model folder, one level above engine/.
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../weights");
    // Every index that the converter wrote. The packed files have int4 and int8 tensors, text_encoder_q8 also float32 ones.
    for name in ["taesd", "unet_packed", "unet_q8", "text_encoder", "text_encoder_q8", "unet", "vae"] {
        // Read and parse the index.
        let json = std::fs::read_to_string(folder.join(format!("{name}.json"))).unwrap_or_else(|error| panic!("{name}.json: {error}"));
        // The entries in file order.
        let entries = WeightIndex::parse(&json).unwrap().into_file_order();
        // Each tensor and the one after it.
        for pair in entries.windows(2) {
            // Where the first tensor ends.
            let end = pair[0].offset + pair[0].stored_byte_length().unwrap();
            // The gap is the alignment padding: 0 to 255 bytes.
            let gap = pair[1].offset.checked_sub(end).unwrap_or_else(|| panic!("{name}: {} overlaps {}", pair[0].name, pair[1].name));
            assert!(gap < 256, "{name}: {} bytes between {} and {}", gap, pair[0].name, pair[1].name);
        }
    }
}
