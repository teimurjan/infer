// Display prints the stage names.
use std::fmt;

// The three stages of one generation, in run order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    // CLIPTextModel.
    TextEncoder,
    // The SD-Turbo UNet.
    Unet,
    // The TAESD decoder.
    Decoder,
}

// The name of a stage, as the weight files call it.
impl fmt::Display for Stage {
    // Write the name.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(match self {
            Self::TextEncoder => "text_encoder",
            Self::Unet => "unet",
            Self::Decoder => "taesd",
        })
    }
}

// What the pipeline reports about a stage, in this order: Prepared, Ran, Freed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StageEvent {
    // The stage loaded its weights and made its slots and bind groups. Nothing ran yet.
    Prepared {
        // The stage.
        stage: Stage,
        // The bytes of the slot buffers of the plan. None for a stage without a plan.
        planned_activations: Option<u64>,
        // The bytes that the run will hold at most: the live bytes now plus the output copy. None for a stage without a plan.
        planned_peak: Option<u64>,
        // The bytes of every engine buffer that is live now.
        live: u64,
    },
    // The stage ran.
    Ran {
        // The stage.
        stage: Stage,
        // The most bytes held at one time since the stage started.
        measured_peak: u64,
        // The seconds of the run, from the first upload to the output on the CPU.
        seconds: f64,
    },
    // The stage was dropped.
    Freed {
        // The stage.
        stage: Stage,
        // The bytes of every engine buffer that is still live. 0 means that the stage freed everything.
        live: u64,
    },
}

// Bytes in one MB, as in PROGRESS.md.
const MB: f64 = 1e6;

// One line per event, as the example and the web page print it.
impl fmt::Display for StageEvent {
    // Write the line.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // Pick the words of the event.
        match self {
            // The plan, before the run.
            Self::Prepared { stage, planned_activations: Some(activations), planned_peak: Some(peak), live } => write!(formatter, "{stage}: prepared; planned activations {activations} bytes ({:.1} MB); planned peak {peak} bytes ({:.1} MB); live {live} bytes", *activations as f64 / MB, *peak as f64 / MB),
            // A stage without a plan.
            Self::Prepared { stage, live, .. } => write!(formatter, "{stage}: not planned; live {live} bytes"),
            // The measured peak, after the run.
            Self::Ran { stage, measured_peak, seconds } => write!(formatter, "{stage}: ran in {seconds:.2} s; measured peak {measured_peak} bytes ({:.1} MB)", *measured_peak as f64 / MB),
            // What is left after the stage is dropped.
            Self::Freed { stage, live } => write!(formatter, "{stage}: freed; live {live} bytes"),
        }
    }
}
