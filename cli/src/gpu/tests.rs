use super::{
    adapter::adapters,
    dispatch::{
        DispatchLayout, CANDIDATES_PER_ROW, DEFAULT_BATCH_SIZE, DEFAULT_DISPATCH_ROWS,
        MAX_BATCH_SIZE, MAX_WORKGROUP_COLUMNS, WORKGROUP_SIZE,
    },
    input::{GpuInput, GpuResult, GPU_INPUT_WORDS, GPU_RESULT_WORDS},
    miner::{Miner, MinerGuard, SHADER},
    Backend,
};
use deadbeef_core::{config, Configuration, Safe, SearchContext};
use std::{mem, time::Instant};

const BENCH_BATCHES: u32 = 128;
// Keep the ignored benchmark quick unless the caller opts into a larger batch.
const DEFAULT_BENCH_BATCH_SIZE: u32 = 1_048_576;

fn test_safe() -> Safe {
    Safe::new(Configuration {
        proxy: config::Proxy {
            factory: "0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67"
                .parse()
                .unwrap(),
            init_code: vec![],
            singleton: "0x41675C099F32341bf84BFc5382aF534df5C7461a"
                .parse()
                .unwrap(),
        },
        account: config::Account {
            owners: vec!["0x1111111111111111111111111111111111111111"
                .parse()
                .unwrap()],
            threshold: 1,
            setup: None,
            fallback_handler: None,
            identifier: None,
        },
    })
}

fn primary_miner() -> Option<MinerGuard> {
    if !primary_adapter_available() {
        return None;
    }

    Some(MinerGuard::new(
        Miner::new(Backend::Primary, None, true).unwrap(),
        Backend::Primary,
    ))
}

fn primary_adapter_available() -> bool {
    !adapters(Backend::Primary).is_empty()
}

fn hardware_adapter_available(backend: Backend) -> bool {
    adapters(backend)
        .iter()
        .any(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
}

fn benchmark_backend() -> Backend {
    match std::env::var("DEADBEEF_GPU_BENCH_BACKEND").as_deref() {
        Ok("metal") => Backend::Metal,
        Ok("vulkan") => Backend::Vulkan,
        Ok("dx12") => Backend::Dx12,
        Ok("gl") => Backend::Gl,
        Ok("primary") | Err(_) => Backend::Primary,
        Ok(backend) => panic!("unsupported DEADBEEF_GPU_BENCH_BACKEND={backend}"),
    }
}

fn benchmark_miner() -> Option<MinerGuard> {
    let backend = benchmark_backend();
    if !hardware_adapter_available(backend) {
        return None;
    }

    Some(MinerGuard::new(
        Miner::new(backend, None, false).unwrap(),
        backend,
    ))
}

fn fixed_nonce_prefix() -> [u8; 24] {
    [
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
    ]
}

fn nonce_from_counter(nonce_prefix: [u8; 24], counter: u64) -> [u8; 32] {
    let mut nonce = [0_u8; 32];
    nonce[0..24].copy_from_slice(&nonce_prefix);
    nonce[24..32].copy_from_slice(&counter.to_be_bytes());
    nonce
}

fn assert_gpu_finds_planted_nonce(
    miner: &Miner,
    context: &SearchContext,
    nonce_prefix: [u8; 24],
    start_counter: u64,
    offset: u32,
) {
    let expected_counter = start_counter + u64::from(offset);
    let expected_nonce = nonce_from_counter(nonce_prefix, expected_counter);
    let expected_address = context.creation_address(expected_nonce);
    let dispatch = DispatchLayout::for_requested_candidates(WORKGROUP_SIZE);

    let mut input = GpuInput::for_search(context, &expected_address.0, &nonce_prefix, dispatch);
    input.set_counter(start_counter);
    let found_nonce = miner
        .run_batch(&input, &dispatch)
        .unwrap()
        .expect("GPU did not find the planted nonce");

    assert_eq!(found_nonce, expected_nonce);
    assert_eq!(context.creation_address(found_nonce), expected_address);
}

fn rounded(requested: u32) -> u32 {
    DispatchLayout::for_requested_candidates(requested).candidates
}

fn word_offset(byte_offset: usize) -> usize {
    assert_eq!(byte_offset % mem::size_of::<u32>(), 0);
    byte_offset / mem::size_of::<u32>()
}

#[test]
fn rounds_batch_size_to_workgroups() {
    assert_eq!(rounded(1), WORKGROUP_SIZE);
    assert_eq!(rounded(WORKGROUP_SIZE + 1), WORKGROUP_SIZE * 2);
    assert_eq!(
        rounded(CANDIDATES_PER_ROW + 1),
        CANDIDATES_PER_ROW + WORKGROUP_SIZE
    );
    assert_eq!(rounded(u32::MAX), MAX_BATCH_SIZE);
}

#[test]
fn lays_out_multi_row_dispatches_densely() {
    let dispatch = DispatchLayout::for_requested_candidates(CANDIDATES_PER_ROW + 1);
    assert_eq!(dispatch.candidates, CANDIDATES_PER_ROW + WORKGROUP_SIZE);
    assert_eq!(dispatch.workgroup_columns, 32_768);
    assert_eq!(dispatch.workgroup_rows, 2);
    assert_eq!(dispatch.candidates_per_row, 32_768 * WORKGROUP_SIZE);

    let dispatch = DispatchLayout::for_requested_candidates(DEFAULT_BATCH_SIZE);
    assert_eq!(dispatch.candidates, DEFAULT_BATCH_SIZE);
    assert_eq!(dispatch.workgroup_columns, MAX_WORKGROUP_COLUMNS);
    assert_eq!(dispatch.workgroup_rows, DEFAULT_DISPATCH_ROWS);
    assert_eq!(dispatch.candidates_per_row, CANDIDATES_PER_ROW);
}

#[test]
fn shader_workgroup_size_matches_host() {
    assert_eq!(WORKGROUP_SIZE, 256);
    let shader_workgroup_size = SHADER
        .lines()
        .find_map(|line| {
            let line = line.trim();
            line.strip_prefix("const WORKGROUP_SIZE: u32 = ")
                .and_then(|value| value.strip_suffix("u;"))
                .and_then(|value| value.parse::<u32>().ok())
        })
        .expect("shader should define WORKGROUP_SIZE");
    assert_eq!(shader_workgroup_size, WORKGROUP_SIZE);
}

#[test]
fn host_input_layout_matches_shader_contract() {
    assert_eq!(
        mem::size_of::<GpuInput>(),
        GPU_INPUT_WORDS * mem::size_of::<u32>()
    );
    assert_eq!(mem::offset_of!(GpuInput, initializer_hash), 0);
    assert_eq!(word_offset(mem::offset_of!(GpuInput, factory)), 8);
    assert_eq!(word_offset(mem::offset_of!(GpuInput, init_code_hash)), 13);
    assert_eq!(word_offset(mem::offset_of!(GpuInput, prefix)), 21);
    assert_eq!(word_offset(mem::offset_of!(GpuInput, nonce_prefix)), 26);
    assert_eq!(word_offset(mem::offset_of!(GpuInput, prefix_len)), 32);
    assert_eq!(
        word_offset(mem::offset_of!(GpuInput, candidates_per_row)),
        33
    );
    assert_eq!(word_offset(mem::offset_of!(GpuInput, counter_low)), 34);
    assert_eq!(word_offset(mem::offset_of!(GpuInput, counter_high)), 35);
}

#[test]
fn host_result_layout_matches_shader_contract() {
    assert_eq!(
        mem::size_of::<GpuResult>(),
        GPU_RESULT_WORDS * mem::size_of::<u32>()
    );
    assert_eq!(word_offset(mem::offset_of!(GpuResult, found)), 0);
    assert_eq!(word_offset(mem::offset_of!(GpuResult, nonce)), 1);
}

#[test]
fn gpu_derivation_matches_cpu_for_counter_byte_order_and_carry() {
    let safe = test_safe();
    let context = safe.search_context();
    let Some(miner) = primary_miner() else {
        return;
    };

    for (start_counter, offset) in [(0x0102_0304_0506_0708, 0), (u64::from(u32::MAX), 1)] {
        assert_gpu_finds_planted_nonce(
            &miner,
            &context,
            fixed_nonce_prefix(),
            start_counter,
            offset,
        );
    }
}

#[test]
fn gpu_matches_every_address_prefix_length() {
    let safe = test_safe();
    let context = safe.search_context();
    let Some(miner) = primary_miner() else {
        return;
    };
    let nonce_prefix = fixed_nonce_prefix();
    let expected_nonce = nonce_from_counter(nonce_prefix, 0x0102_0304_0506_0708);
    let expected_address = context.creation_address(expected_nonce);
    let dispatch = DispatchLayout::for_requested_candidates(WORKGROUP_SIZE);

    for prefix_len in 1..=20 {
        let prefix = &expected_address.0[..prefix_len];
        let mut input = GpuInput::for_search(&context, prefix, &nonce_prefix, dispatch);
        input.set_counter(0x0102_0304_0506_0708);
        let found_nonce = miner
            .run_batch(&input, &dispatch)
            .unwrap()
            .expect("GPU did not find any nonce for planted prefix");
        assert!(
            context.creation_address(found_nonce).0.starts_with(prefix),
            "GPU returned a nonce that does not match a {prefix_len}-byte prefix",
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter and prints throughput"]
fn gpu_benchmark_fixed_batches() {
    let safe = test_safe();
    let context = safe.search_context();
    let Some(miner) = benchmark_miner() else {
        return;
    };
    let nonce_prefix = fixed_nonce_prefix();

    let start = Instant::now();
    let batch_size = std::env::var("DEADBEEF_GPU_BENCH_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_BENCH_BATCH_SIZE);
    let dispatch = DispatchLayout::for_requested_candidates(batch_size);
    let batch_size = dispatch.candidates;
    let mut input = GpuInput::for_search(&context, &[0xff; 20], &nonce_prefix, dispatch);

    for batch in 0..BENCH_BATCHES {
        let counter = u64::from(batch) * u64::from(batch_size);
        input.set_counter(counter);
        let result = miner.run_batch(&input, &dispatch).unwrap();
        assert!(result.is_none());
    }

    let elapsed = start.elapsed().as_secs_f64();
    let candidates = u64::from(BENCH_BATCHES) * u64::from(batch_size);
    eprintln!(
        "gpu benchmark: {candidates} candidates ({BENCH_BATCHES} x {batch_size}) in {elapsed:.3}s = {:.2} Maddr/s",
        candidates as f64 / elapsed / 1_000_000.0
    );
}
