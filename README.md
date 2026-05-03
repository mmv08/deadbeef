# `0xdeadbeef`

Tool used for computing vanity Safe addresses.

This tool only officially supports the latest Safe deployment [`v1.4.1`](https://github.com/safe-global/safe-deployments/tree/main/src/assets/v1.4.1).

Since this version of the Safe proxy factory uses `CREATE2` op-code, we can change the final address by fiddling with the user-specified `saltNonce` parameter.
It works by randomly trying out different values for the `saltNonce` parameter until it find ones that creates an address matching the desired prefix.

## Building

For longer prefixes, this can take a **very** long time, so be sure to build with release:

```sh
cargo build --release
```

## Usage

```sh
deadbeef --help
```

For example, to generate calldata for creating a Safe with initial owners of `0x1111111111111111111111111111111111111111` and `0x2222222222222222222222222222222222222222` and prefix `0xdeadbeef`:

```sh
deadbeef \
  --owner 0x1111111111111111111111111111111111111111 \
  --owner 0x2222222222222222222222222222222222222222 \
  --prefix 0x5afe
```

This will output some result like:

```
address:     0x5AFE941f405085500803E7479cEcEC46cB50B70A
factory:     0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67
singleton:   0x41675C099F32341bf84BFc5382aF534df5C7461a
initializer: 0xb63e800d00000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000bd89a1ce4dde368ffab0ec35506eece0b1ffdc540000000000000000000000000000000000000000000000000000000000000160000000000000000000000000fd0732dc9e303f09fcef3a7388ad10a83459ec990000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000111111111111111111111111111111111111111100000000000000000000000022222222222222222222222222222222222222220000000000000000000000000000000000000000000000000000000000000024fe51f64300000000000000000000000029fcb43b46531bca003ddc8fcb67ffe91900c76200000000000000000000000000000000000000000000000000000000
salt nonce:  0xdc50d6fbe25fc5fe0cffb1a2fc5170b76ac620f74f22c7b247d8c73d83072302
---
owners:      0x1111111111111111111111111111111111111111
             0x2222222222222222222222222222222222222222
threshold:   1
to:          0xBD89A1CE4DDe368FFAB0eC35506eEcE0b1fFdc54
data:        0xfe51f64300000000000000000000000029fcb43b46531bca003ddc8fcb67ffe91900c762
fallback:    0xfd0732Dc9E303f09fCEf3a7388Ad10A83459Ec99
---
calldata:    0x1688f0b900000000000000000000000041675c099f32341bf84bfc5382af534df5c7461a0000000000000000000000000000000000000000000000000000000000000060dc50d6fbe25fc5fe0cffb1a2fc5170b76ac620f74f22c7b247d8c73d8307230200000000000000000000000000000000000000000000000000000000000001c4b63e800d00000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000bd89a1ce4dde368ffab0ec35506eece0b1ffdc540000000000000000000000000000000000000000000000000000000000000160000000000000000000000000fd0732dc9e303f09fcef3a7388ad10a83459ec990000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000111111111111111111111111111111111111111100000000000000000000000022222222222222222222222222222222222222220000000000000000000000000000000000000000000000000000000000000024fe51f64300000000000000000000000029fcb43b46531bca003ddc8fcb67ffe91900c7620000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

Note that the owner signature threshold defaults to 1 but can optionally be specified with:

```sh
deadbeef ... --threshold 2 ...
```

## GPU Mining

GPU mining is opt-in.

### Vocabulary

A handful of terms appear throughout this section and the GPU source files. Defining them up front keeps the rest of the section short:

- **wgpu** is the Rust binding for the WebGPU compute API. It lets one shader (a small program the GPU runs in parallel for every input it is given) run on Apple, NVIDIA, AMD, and Intel GPUs through whichever native driver is available.
- **Metal** (Apple), **Vulkan** (Linux/Windows on most GPUs), **DX12** (Windows), and **GL** (a fallback path) are the native driver choices wgpu can pick.
- **WSLg** is Microsoft's GPU bridge for the Windows Subsystem for Linux. **Mesa** is the open-source graphics stack that ships with most Linux distros, and **llvmpipe** is its software-only fallback — do not use it for mining, it runs on the CPU.
- A **dispatch** is one job the CLI hands the GPU. A **workgroup** is a tile of 256 shader invocations the GPU schedules together; a single **invocation** is one independent run of the shader on one candidate salt nonce.
- **Readback** means copying a buffer's bytes from the GPU back to the CPU. The miner reads back one tiny 36-byte result struct per dispatch, but the readback is synchronous, so we want each dispatch large enough that the readback is not the bottleneck.
- **Maddr/s** is "millions of candidate addresses per second."

### Usage

List available adapters:

```sh
deadbeef --list-gpus
```

The adapter list includes the device type, plus driver details when wgpu reports
them. If wgpu reports an adapter as `Cpu / Software Rendering`, the miner will
not select it by default.

Mine with the first available GPU adapter:

```sh
deadbeef ... --gpu --prefix 0x00
```

Select an adapter and backend explicitly:

```sh
deadbeef ... --gpu --gpu-adapter 0 --gpu-backend metal
deadbeef ... --gpu --gpu-adapter 0 --gpu-backend vulkan
deadbeef ... --gpu --gpu-adapter 0 --gpu-backend dx12
deadbeef ... --gpu --gpu-adapter 0 --gpu-backend gl
```

Use `metal` for Apple GPUs, `vulkan` for NVIDIA on Linux, and `dx12` or `vulkan` for NVIDIA on Windows.
On WSLg, Vulkan may only expose Mesa `llvmpipe`; if so, use Mesa's D3D12-backed GL path:

```sh
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  deadbeef ... --gpu --gpu-backend gl
```

`gl` is a fallback backend. In the current CLI implementation, a GL mining run exits without tearing down the GL miner before process exit.

`--gpu-batch-size` requests how many salt nonces are scanned per dispatch.
The miner rounds that request up to a whole workgroup and clamps very small or very large values to the supported range.
The default is `8 × 65,535 × 256 = 134,215,680` candidates per dispatch: WebGPU caps each dispatch dimension at 65,535 workgroups, each workgroup runs 256 invocations, and the host stacks 8 such rows to reduce readback overhead while staying backend-neutral.

How to think about the trade-off: each dispatch ends with a synchronous readback of one tiny result struct, which costs a fixed amount of time regardless of batch size. A larger batch amortises that fixed cost over more candidate addresses, so throughput climbs as the batch grows. The catch is that a larger batch also takes longer to complete, so a slow GPU spends more wall time per dispatch and the miner reports progress less often. Pick a value that fills your GPU but still finishes a dispatch in a few seconds. On high-throughput GPUs, larger explicit batch-size requests can stack more rows and reduce readback overhead further; on slower GPUs, the default is intentionally conservative.

GPU mining prints periodic progress to stderr while it runs; `--quiet` disables those progress messages.

### Reference Benchmarks

These numbers are a point-in-time reference, not a guaranteed result. GPU and CPU throughput can change with power state, thermal state, other GPU work, compiler versions, and batch size.
The CPU benchmark below measures one CPU core in the core mining loop. The normal CLI CPU miner uses multiple threads by default, so it can be faster than the single-core number shown here.
The benchmark results below were recorded on May 3, 2026 against branch commit `da8cf78`.

Each dispatch row covers `65,535 × 256 = 16,776,960` candidates, so larger `--gpu-batch-size` values use more rows: a single-row dispatch covers about 16.8 million candidates, the default eight-row dispatch covers about 134.2 million, and a 32-row dispatch covers about 536.9 million.

The `DEADBEEF_GPU_BENCH_*` environment variables in the commands below are read only by the ignored benchmark test, not by the normal `deadbeef --gpu` CLI; setting them on a regular mining run has no effect.

MacBook Pro environment:

- MacBook Pro `Mac15,10`
- Apple M3 Max, 14-core CPU (10 performance, 4 efficiency)
- Apple M3 Max GPU, 30 cores, Metal backend
- 36 GB memory
- macOS 26.4.1
- Rust 1.95.0, Cargo 1.95.0

MacBook Pro benchmark commands:

```sh
cargo bench -p deadbeef-bench
```

```sh
DEADBEEF_GPU_BENCH_BATCH_SIZE=134215680 \
cargo test --release -p deadbeef gpu_benchmark_fixed_batches -- --ignored --nocapture
```

| Miner | Result |
| --- | --- |
| CPU, single-thread core loop | 297 ns/address mean, about 3.37 Maddr/s |
| GPU, Metal, default 8-row dispatch batch | 17,179,607,040 candidates in 66.978 s, about 256.50 Maddr/s |

On this machine, the GPU benchmark is about 76x faster than one CPU core in the core mining loop.

WSL2 desktop environment:

- Ubuntu 24.04.4 LTS on WSL 2.4.12.0, WSLg 1.0.65
- Windows build 10.0.26200.8246
- AMD Ryzen 9 7900 12-core CPU, 24 logical CPUs
- NVIDIA GeForce RTX 5090, 32 GB VRAM, driver 596.21
- WSLg/Mesa D3D12 GL backend, selected with `GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA --gpu-backend gl`
- Rust 1.95.0, Cargo 1.95.0

WSL2 benchmark commands:

```sh
cargo bench -p deadbeef-bench
```

```sh
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
DEADBEEF_GPU_BENCH_BACKEND=gl \
DEADBEEF_GPU_BENCH_BATCH_SIZE=16776960 \
cargo test --release -p deadbeef gpu_benchmark_fixed_batches -- --ignored --nocapture
```

```sh
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
DEADBEEF_GPU_BENCH_BACKEND=gl \
DEADBEEF_GPU_BENCH_BATCH_SIZE=134215680 \
cargo test --release -p deadbeef gpu_benchmark_fixed_batches -- --ignored --nocapture
```

```sh
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
DEADBEEF_GPU_BENCH_BACKEND=gl \
DEADBEEF_GPU_BENCH_BATCH_SIZE=536862720 \
cargo test --release -p deadbeef gpu_benchmark_fixed_batches -- --ignored --nocapture
```

| Miner | Result |
| --- | --- |
| CPU, single-thread core loop | 573.3 ns/address mean, about 1.74 Maddr/s |
| GPU, GL over D3D12, single-row dispatch batch | 2,147,450,880 candidates in 1.112 s, about 1,930.93 Maddr/s |
| GPU, GL over D3D12, default 8-row dispatch batch | 17,179,607,040 candidates in 8.159 s, about 2,105.58 Maddr/s |
| GPU, GL over D3D12, tuned 32-row dispatch batch | 68,718,428,160 candidates in 30.886 s, about 2,224.92 Maddr/s |

On this machine, the fastest GPU benchmark above is about 1,280x faster than one CPU core in the core mining loop.

### GPU Implementation Notes

The GPU miner keeps Safe construction and final verification on the CPU.
The shader only scans candidate salt nonces and returns a matching nonce for the CPU to re-check.

For using Safe deployments on different chains can also be used:

```sh
deadbeef ... --chain 100 ...
```

As well as custom fallback handlers:

```sh
deadbeef ... --fallback-handler 0x4e305935b14627eA57CBDbCfF57e81fd9F240403 ...
```

By default, the generated initializer will use the `SafeToL2Setup` contract. This ensures that the Safe deployment transaction can be replayed to get the same address on all supported chains. In order to disable this behaviour (not recommended), set the `--safe-to-l2-setup` flag to the 0 address:

```sh
deadbeef ... --safe-to-l2-setup 0x0000000000000000000000000000000000000000 ...
```

## Creating the Safe

The above command will generate some [calldata](https://www.quicknode.com/guides/ethereum-development/transactions/ethereum-transaction-calldata) for creating a Safe with the specified owners and threshold.

To create the safe, simply execute a transaction to the [factory address](https://etherscan.io/address/0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67) with the generated calldata, or use the `createProxyWithNonce` function on Etherscan.
The transaction can be executed from any account (it can be done in MetaMask directly for example).

### Metamask Steps

Go to Settings -> Advanced and enable `Show hex data`. When you go to create a transaction you will have a new optional field labelled `Hex data`.

Send a 0Ξ transaction to the factory address, placing the generated calldata in the `Hex data` field.

Metamask will recognise it as a contract interaction in the confirmation step.

### Etherscan

Use the `--params` flag to output contract-ready inputs.

1. Visit the `factory` URL from the command output. The link should open up the explorer at the correct location, if not click on _Contract_ > _Write Contract_.
2. Click on _Connect to Web3_ to connect the account you wish to pay for the Safe creation.
3. Fill the fields for the function _3. createProxyWithNonce (0x1688f0b9)_ using the generated outputs.

## Unsupported Chains

Safe deployments on non-officially supported networks can also be used by overriding all contract addresses and the proxy init code:

```sh
deadbeef ... \
  --chain $UNSUPPORTED_CHAIN \
  --proxy-factory 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --proxy-init-code 0xbb \
  --singleton 0xcccccccccccccccccccccccccccccccccccccccc
```

**Use this with caution**, this assumes that the proxy address is computed in the exact same was as on Ethereum, which may not be the case for all networks.
This feature is not officially supported by the tool.

## Is This Vegan Friendly 🥦?

Of course!
No actual cows were harmed in the creation or continual use of this tool.

```sh
alias deadbeef=seedfeed
```
