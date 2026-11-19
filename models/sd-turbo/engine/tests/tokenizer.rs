// fs reads the tokenizer files and the cases.
use std::fs;
// Path builds the file paths.
use std::path::Path;

// The engine's tokenizer.
use engine::{CONTEXT_IDS, Tokenizer};

// One prompt and the ids that transformers gave for it.
#[derive(serde::Deserialize)]
struct Case {
    // The prompt text.
    prompt: String,
    // The 77 ids.
    ids: Vec<u32>,
}

// The file that testdata/tokenizer/make_cases.py writes.
#[derive(serde::Deserialize)]
struct Cases {
    // The transformers version that made the ids.
    transformers: String,
    // The cases, in order.
    cases: Vec<Case>,
}

// Read one text file, or stop with the command that makes it.
fn read(path: &Path, command: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}. Run `{command}` first.", path.display()))
}

// The engine must give the same 77 ids as the SD-Turbo tokenizer of transformers for every case.
// A different id is a different prompt for the text encoder, so the image would change.
#[test]
fn tokenizer_matches_transformers() {
    // The project folder, one level above engine/.
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("engine/ has a parent folder");
    // The two tokenizer files.
    let vocab = read(&project.join("weights/tokenizer_vocab.json"), "uv run export_tokenizer.py");
    let merges = read(&project.join("weights/tokenizer_merges.txt"), "uv run export_tokenizer.py");
    let tokenizer = Tokenizer::new(&vocab, &merges).unwrap_or_else(|error| panic!("{error}"));
    // The cases.
    let cases: Cases = serde_json::from_str(&read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/tokenizer/cases.json"), "uv run testdata/tokenizer/make_cases.py")).expect("cases.json is valid JSON");
    // The prompts whose ids differ, with both lists.
    let failures: Vec<String> = cases
        .cases
        .iter()
        .filter_map(|case| {
            let ids = tokenizer.encode(&case.prompt);
            (ids != case.ids).then(|| format!("{:?}:\n  engine       {ids:?}\n  transformers {:?}", case.prompt, case.ids))
        })
        .collect();
    // One summary line.
    println!("tokenizer: {} of {} prompts match transformers {}, {CONTEXT_IDS} ids each", cases.cases.len() - failures.len(), cases.cases.len(), cases.transformers);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
