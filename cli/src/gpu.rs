//! GPU-backed Safe address mining.
//!
//! The CPU owns Safe configuration and final verification. The GPU only scans
//! candidate salt nonces and returns a matching nonce for the CPU to re-check.
//!
mod adapter;
mod dispatch;
mod input;
mod miner;

#[cfg(test)]
mod tests;

use self::{
    adapter::{adapter_description, adapters, NO_ADAPTERS},
    dispatch::DispatchLayout,
    input::{pack_words, GpuInput},
    miner::{Miner, MinerGuard},
};
use clap::ValueEnum;
use deadbeef_core::Safe;
use rand::Rng as _;
use std::{
    error::Error,
    io,
    time::{Duration, Instant},
};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const DEFAULT_BATCH_SIZE: u32 = dispatch::DEFAULT_BATCH_SIZE;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[default]
    Primary,
    Metal,
    Vulkan,
    Dx12,
    Gl,
}

impl Backend {
    /// Translate the user-facing backend choice into the bitset wgpu expects.
    pub(super) fn to_wgpu_bitset(self) -> wgpu::Backends {
        match self {
            Self::Primary => wgpu::Backends::PRIMARY,
            Self::Metal => wgpu::Backends::METAL,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Gl => wgpu::Backends::GL,
        }
    }
}

/// Runtime options for GPU mining.
#[derive(Copy, Clone, Debug)]
pub struct Options {
    /// GPU backend to enumerate. `Primary` lets wgpu choose the native backend.
    pub backend: Backend,
    /// Adapter index from `--list-gpus`; `None` selects the first adapter.
    pub adapter: Option<usize>,
    /// Candidate count per compute dispatch. Rounded to `WORKGROUP_SIZE`.
    pub batch_size: u32,
    /// Allow adapters that wgpu reports as CPU/software renderers.
    pub allow_software_adapter: bool,
    /// Print progress to stderr between GPU dispatches.
    pub progress: bool,
}

struct Progress {
    start: Instant,
    next_report: Instant,
    dispatched: u64,
}

pub fn list_adapters(backend: Backend) -> Result<()> {
    let adapters = adapters(backend);
    if adapters.is_empty() {
        return Err(err(NO_ADAPTERS));
    }

    for (index, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("{index}: {}", adapter_description(&info));
    }

    Ok(())
}

pub fn search(safe: &mut Safe, prefix: &[u8], options: Options) -> Result<()> {
    if prefix.len() > 20 {
        return Err(err("prefix cannot be longer than an Ethereum address"));
    }

    let context = safe.search_context();
    let dispatch = DispatchLayout::for_requested_candidates(options.batch_size);
    let batch_size = dispatch.candidates;
    let miner = MinerGuard::new(
        Miner::new(
            options.backend,
            options.adapter,
            options.allow_software_adapter,
        )?,
        options.backend,
    );
    let mut rng = rand::rng();
    let mut nonce_prefix = [0_u8; 24];
    rng.fill(&mut nonce_prefix);
    let mut counter = 0_u64;
    let mut progress = options
        .progress
        .then(|| Progress::new(prefix.len(), batch_size));

    // Nonce generation is split across CPU and GPU:
    //
    //   saltNonce = random_24_byte_prefix || sequential_8_byte_counter
    //
    // The CPU chooses the random prefix once per 2^64-counter range, while the
    // GPU scans the counter range sequentially. This keeps dispatch input small
    // and avoids per-candidate RNG in the shader.
    let mut input = GpuInput::for_search(&context, prefix, &nonce_prefix, dispatch);

    loop {
        // Exhausting one 64-bit counter range is theoretical; if it happens,
        // re-randomize the nonce prefix instead of repeating the same range.
        if counter.checked_add(u64::from(batch_size)).is_none() {
            rng.fill(&mut nonce_prefix);
            input.nonce_prefix = pack_words(&nonce_prefix);
            counter = 0;
        }

        input.set_counter(counter);

        if let Some(nonce) = miner.run_batch(&input, &dispatch)? {
            let address = context.creation_address(nonce);

            // Fail closed if the untrusted GPU candidate disagrees with the CPU
            // derivation; continuing would hide a broken shader or driver path.
            if !address.0.starts_with(prefix) {
                return Err(err("GPU result did not match the requested prefix"));
            }

            safe.update_salt_nonce(|n| *n = nonce);
            return Ok(());
        }

        if let Some(progress) = &mut progress {
            progress.record_batch(batch_size);
        }
        counter += u64::from(batch_size);
    }
}

impl Progress {
    fn new(prefix_len: usize, batch_size: u32) -> Self {
        let now = Instant::now();
        eprintln!(
            "GPU mining: scanning {prefix_len}-byte prefix with {batch_size} candidates per dispatch"
        );
        Self {
            start: now,
            next_report: now + PROGRESS_INTERVAL,
            dispatched: 0,
        }
    }

    fn record_batch(&mut self, batch_size: u32) {
        self.dispatched += u64::from(batch_size);

        let now = Instant::now();
        if now < self.next_report {
            return;
        }

        let elapsed = now.duration_since(self.start).as_secs_f64();
        let rate = self.dispatched as f64 / elapsed / 1_000_000.0;
        eprintln!(
            "GPU mining: dispatched {:.2}M candidates in {:.1}s ({rate:.2} Maddr/s)",
            self.dispatched as f64 / 1_000_000.0,
            elapsed,
        );
        self.next_report = now + PROGRESS_INTERVAL;
    }
}

fn err(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}
