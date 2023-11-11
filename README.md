# `0xdeadbeef`

Tool used for computing vanity Safe addresses.

This tool only officially supports the latest Safe deployment [`v1.4.1`](https://github.com/safe-global/safe-deployments/tree/9cf5d5f75819371b7b63fcc66f316bcd920f3c58/src/assets/v1.4.1).
For Ethereum:
- `SafeProxyFactory` [`0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67`](https://etherscan.io/address/0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67)
- `Safe` [`0x41675C099F32341bf84BFc5382aF534df5C7461a`](https://etherscan.io/address/0x41675C099F32341bf84BFc5382aF534df5C7461a)
- `CompatibilityFallbackHandler` [`0xfd0732Dc9E303f09fCEf3a7388Ad10A83459Ec99`](https://etherscan.io/address/0xfd0732Dc9E303f09fCEf3a7388Ad10A83459Ec99)

Since this version of the Safe proxy factory uses `CREATE2` op-code, we can change the final address by fiddling with the user-specified `saltNonce` parameter.
It works by randomly trying out different values for the `saltNonce` parameter until it find ones that creates an address matching the desired prefix.

*Commit [`24df00b`](https://github.com/nlordell/deadbeef/tree/24df00bfb1e7fdb594be97c017cd627e643c5318) supports Safe `v1.3.0`.* 

## Building

For longer prefixes, this can take a **very** long time, so be sure to build with release:

```
cargo build --release
```

## Usage

```
deadbeef --help
```

For example, to generate calldata for creating a Safe with initial owners of `0x1111111111111111111111111111111111111111` and `0x2222222222222222222222222222222222222222` and prefix `0xdeadbeef`:

```
deadbeef \
  --owner 0x1111111111111111111111111111111111111111 \
  --owner 0x2222222222222222222222222222222222222222 \
  --prefix 0xdeadbeef
```

Note that the owner signature threshold defaults to 1 but can optionally be specified with:

```
deadbeef ... --threshold 2 ...
```

This will output some result like:

```
address:   0xdEADBEefEAFbe3622E000Fda70bBF742dDDEbC71
factory:   0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67
singleton: 0x41675C099F32341bf84BFc5382aF534df5C7461a
fallback:  0xfd0732Dc9E303f09fCEf3a7388Ad10A83459Ec99
owners:    0x1111111111111111111111111111111111111111
           0x2222222222222222222222222222222222222222
threshold: 1
calldata:  0x1688f0b900000000000000000000000041675c099f32341bf84bfc5382af534df5c7461a0000000000000000000000000000000000000000000000000000000000000060e4cf27da52614adef0b22f15b975c956927faa0ab22fa1f1a82db0760a9ddddd0000000000000000000000000000000000000000000000000000000000000184b63e800d0000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000160000000000000000000000000fd0732dc9e303f09fcef3a7388ad10a83459ec99000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200000000000000000000000011111111111111111111111111111111111111110000000000000000000000002222222222222222222222222222222222222222000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

For using Safe deployments on different chains can also be used:

```
deadbeef ... --chain 100 ...
```

As well as custom fallback handlers:

```
deadbeef ... --fallback-handler 0x4e305935b14627eA57CBDbCfF57e81fd9F240403 ...
```

## Creating the Safe

The above command will generate some [calldata](https://www.quicknode.com/guides/ethereum-development/transactions/ethereum-transaction-calldata) for creating a Safe with the specified owners and threshold.

To create the safe, simply execute a transaction to the [factory address](https://etherscan.io/address/0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67) with the generated calldata, or use the `createProxyWithNonce` function on Etherscan.
The transaction can be executed from any account (it can be done in MetaMask directly for example).

### Metamask Steps

Go to Settings -> Advanced and enable `Show hex data`. When you go to create a transaction you will have a new optional field labelled `Hex data`.

Send a `0eth` transaction to the factory address, placing the generated calldata in the `Hex data` field.

Metamask will recognise it as a contract interaction in the confirmation step.

### Etherscan

Use the `--params` flag to output contract-ready inputs.

1. Visit the [factory address](https://etherscan.io/address/0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67).
2. Click Contract -> Write Contract -> Connect to Web3.
3. Connect the account you wish to pay for the Safe creation.

Fill the fields in function `3. createProxyWithNonce` using the generated outputs.

## Unsupported Chains

Safe deployments on non-officially supported networks can also be used by overriding all contract addresses and the proxy init code:

```
deadbeef ... \
  --chain $UNSUPPORTED_CHAIN \
  --proxy-factory 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --proxy-init-code 0xbb \
  --singleton 0xcccccccccccccccccccccccccccccccccccccccc \
  --fallback-handler 0xdddddddddddddddddddddddddddddddddddddddd
```

**Use this with caution**, this assumes that the proxy address is computed in the exact same was as on Ethereum, which may not be the case for all networks.
This feature is not officially supported by the tool.


## Is This Vegan Friendly 🥦?

Of course!
No actual cows were harmed in the creation or continual use of this tool.

```
% alias deadbeef=seedfeed
% seedfeed \
  --owner 0x1111111111111111111111111111111111111111 \
  --owner 0x2222222222222222222222222222222222222222 \
  --owner 0x3333333333333333333333333333333333333333 \
  --threshold 2 \
  --prefix 0x5eedfeed
address:   0x5EedFeED446B211419EBac9253FbB8b9556781D1
factory:   0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67
singleton: 0x41675C099F32341bf84BFc5382aF534df5C7461a
fallback:  0xfd0732Dc9E303f09fCEf3a7388Ad10A83459Ec99
owners:    0x1111111111111111111111111111111111111111
           0x2222222222222222222222222222222222222222
           0x3333333333333333333333333333333333333333
threshold: 2
calldata:  0x1688f0b900000000000000000000000041675C099F32341bf84BFc5382aF534df5C7461a00000000000000000000000000000000000000000000000000000000000000605b237da2310f4948260ac5661d88f5150e5e62ca1f4faa76e3598bd427f212e800000000000000000000000000000000000000000000000000000000000001a4b63e800d0000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000180000000000000000000000000fd0732Dc9E303f09fCEf3a7388Ad10A83459Ec990000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000000000000111111111111111111111111111111111111111100000000000000000000000022222222222222222222222222222222222222220000000000000000000000003333333333333333333333333333333333333333000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```
