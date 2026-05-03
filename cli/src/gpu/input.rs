use super::dispatch::DispatchLayout;
use deadbeef_core::SearchContext;
use std::mem;

pub(super) const GPU_INPUT_WORDS: usize = 36;
pub(super) const GPU_RESULT_WORDS: usize = 9;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuInput {
    // Host-side storage-buffer layout. Keep this field order in sync with
    // `struct Input` in `gpu.wgsl`; the tests below pin the host word offsets.
    pub(super) initializer_hash: [u32; 8],
    pub(super) factory: [u32; 5],
    pub(super) init_code_hash: [u32; 8],
    pub(super) prefix: [u32; 5],
    // First 24 bytes of the 32-byte Safe salt nonce. The final 8 bytes are the
    // counter in big-endian byte order: high bytes first, then low bytes.
    pub(super) nonce_prefix: [u32; 6],
    pub(super) prefix_len: u32,
    // Candidate count in one dispatch row.
    pub(super) candidates_per_row: u32,
    pub(super) counter_low: u32,
    pub(super) counter_high: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuResult {
    // Host-side storage-buffer layout. Keep this field order in sync with
    // `struct ResultBuffer` in `gpu.wgsl`; the tests below pin the host word
    // offsets.
    pub(super) found: u32,
    pub(super) nonce: [u32; 8],
}

const _: () = {
    assert!(mem::size_of::<GpuInput>() == GPU_INPUT_WORDS * mem::size_of::<u32>());
    assert!(mem::align_of::<GpuInput>() == mem::align_of::<u32>());
    assert!(mem::size_of::<GpuResult>() == GPU_RESULT_WORDS * mem::size_of::<u32>());
    assert!(mem::align_of::<GpuResult>() == mem::align_of::<u32>());
};

impl GpuInput {
    pub(super) fn for_search(
        context: &SearchContext,
        prefix: &[u8],
        nonce_prefix: &[u8; 24],
        dispatch: DispatchLayout,
    ) -> Self {
        Self {
            initializer_hash: pack_words(&context.initializer_hash),
            factory: pack_words(&context.factory.0),
            init_code_hash: pack_words(&context.init_code_hash),
            prefix: pack_words(prefix),
            nonce_prefix: pack_words(nonce_prefix),
            prefix_len: prefix.len() as u32,
            candidates_per_row: dispatch.candidates_per_row,
            counter_low: 0,
            counter_high: 0,
        }
    }

    // The shader receives the counter split into two u32 values because WGSL
    // u64 support is not portable across all wgpu backends.
    pub(super) fn set_counter(&mut self, counter: u64) {
        self.counter_low = counter as u32;
        self.counter_high = (counter >> 32) as u32;
    }
}

pub(super) fn pack_words<const N: usize>(bytes: &[u8]) -> [u32; N] {
    // Pack host byte strings into the same little-endian u32 words used by WGSL
    // and Keccak lanes. Keccak's spec stores each 64-bit lane as little-endian
    // bytes, which is the same byte order WGSL uses inside a u32, so the host
    // packing matches the shader's view without any extra swapping. See the
    // `GpuInput` layout note above for the full host/device ABI. Missing bytes
    // remain zero, which is useful for shorter prefixes.
    let mut words = [0_u32; N];
    for (index, byte) in bytes.iter().copied().take(N * 4).enumerate() {
        words[index / 4] |= u32::from(byte) << ((index % 4) * 8);
    }
    words
}

pub(super) fn unpack_words<const WORDS: usize, const BYTES: usize>(
    words: [u32; WORDS],
) -> [u8; BYTES] {
    // Inverse of `pack_words`, used for the nonce returned by the shader before
    // CPU verification. The output must fit inside the input words.
    debug_assert!(BYTES <= WORDS * 4);
    let mut bytes = [0_u8; BYTES];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = ((words[index / 4] >> ((index % 4) * 8)) & 0xff) as u8;
    }
    bytes
}
