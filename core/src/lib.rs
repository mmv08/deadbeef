#[macro_use]
mod address;
mod create2;
mod safe;

pub use self::{
    address::Address,
    safe::{Contracts, Safe, Transaction},
};
pub use hex_literal::hex;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{rngs::SmallRng, Rng as _, SeedableRng as _};

// This kernel cannot be used with the current Safe Proxy implementation
// as the safe uses a hash of the initialiser and a nonce for the create2 salt
// while the kernel is search for a create2 salt.
static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

/// Search for a vanity address with the specified Safe parameters and prefix.
pub fn search(safe: &mut Safe, prefix: &[u8]) {
    let mut rng = SmallRng::from_entropy();
    while !safe.creation_address().0.starts_with(prefix) {
        safe.update_salt_nonce(|n| rng.fill(n));
    }
}

pub fn search_gpu(safe: &mut Safe, prefix: &[u8], gpu_device: &u8) {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        *gpu_device
    );

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    let device = Device::by_idx_wrap(platform, *gpu_device as usize)?;

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build();

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;
}
