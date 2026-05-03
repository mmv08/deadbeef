mod chain;
mod deployment;
mod gpu;

use self::chain::{Chain, Singleton};
use chain::Details;
use clap::Parser;
use deadbeef_core::{config, Address, Configuration, NonZeroAddress, Safe};
use hex::FromHexError;
use std::{num::NonZeroUsize, process, str::FromStr, sync::mpsc, thread};

/// Generate vanity addresses for Safe deployments.
#[derive(Clone, Parser)]
#[command(group(clap::ArgGroup::new("gpu_mode").args(["gpu", "list_gpus"])))]
struct Args {
    /// Number of CPU mining threads to use. Ignored when `--gpu` is set.
    #[arg(short = 'n', long, default_value_t = num_cpus::get())]
    threads: usize,

    /// Safe owners.
    ///
    /// Can be specified multiple times in order to specify multiple owners.
    /// They will be included in the provided order.
    #[arg(
        short,
        long = "owner",
        required_unless_present = "list_gpus",
        num_args = 1..
    )]
    owners: Vec<NonZeroAddress>,

    /// Owner signature threshold.
    #[arg(short, long, default_value_t = 1)]
    threshold: usize,

    /// The prefix to look for.
    #[arg(short, long, required_unless_present = "list_gpus")]
    prefix: Option<Hex>,

    /// The chain ID to find a vanity Safe address for. If the chain is not
    /// supported, then all of '--proxy-factory', '--proxy-init-code', and
    /// '--singleton' must be specified.
    #[arg(short, long, default_value_t = Chain::ethereum())]
    chain: Chain,

    /// Override for the `SafeProxyFactory` address.
    #[arg(long)]
    proxy_factory: Option<NonZeroAddress>,

    /// Override for the `SafeProxy` init code.
    #[arg(long)]
    proxy_init_code: Option<Hex>,

    /// Override for the `Safe` singleton address.
    #[arg(long)]
    singleton: Option<NonZeroAddress>,

    /// Override for the `SafeL2` singleton address.
    ///
    /// For unsupported chains, if this is specified then `--safe-to-l2-setup`
    /// must also be specified.
    #[arg(long)]
    l2_singleton: Option<NonZeroAddress>,

    /// Override for the `SafeToL2Setup` address.
    ///
    /// Specifying the 0 address (or not specifying the contract address for
    /// unknown chains) will disable this feature (which is not recommended).
    ///
    /// For unsupported chains, if this is specified then `--l2-singleton` must
    /// also be specified.
    #[arg(long)]
    safe_to_l2_setup: Option<Address>,

    /// Override for the fallback handler address.
    #[arg(long)]
    fallback_handler: Option<NonZeroAddress>,

    /// Mine using a GPU.
    #[arg(long)]
    gpu: bool,

    /// List available GPU adapters and exit.
    #[arg(long)]
    list_gpus: bool,

    /// Select the GPU backend to use. `gl` is a fallback path; the current CLI
    /// exits without tearing down a GL miner before process exit.
    #[arg(
        long,
        value_enum,
        default_value_t = gpu::Backend::Primary,
        requires = "gpu_mode"
    )]
    gpu_backend: gpu::Backend,

    /// Select the GPU adapter index from `--list-gpus`.
    #[arg(long, requires = "gpu", conflicts_with = "list_gpus")]
    gpu_adapter: Option<usize>,

    /// Requested GPU candidates per dispatch. Rounded to a supported size.
    #[arg(
        long,
        default_value_t = gpu::DEFAULT_BATCH_SIZE,
        requires = "gpu",
        conflicts_with = "list_gpus"
    )]
    gpu_batch_size: u32,

    /// Allow software-rendered GPU adapters. Intended for tests and diagnostics.
    #[arg(long, hide = true, requires = "gpu", conflicts_with = "list_gpus")]
    allow_software_gpu: bool,

    /// Quiet mode.
    ///
    /// Only output the transaction calldata without any extra information.
    #[arg(short, long, conflicts_with = "params")]
    quiet: bool,

    /// Parameters mode.
    ///
    /// Only output the parameters for the calling the `createProxyWithNonce`
    /// function on the `SafeProxyFactory`.
    #[arg(short = 'P', long, conflicts_with = "quiet")]
    params: bool,
}

/// Helper type for parsing hexadecimal byte input from the command line.
#[derive(Clone)]
struct Hex(Vec<u8>);

impl Hex {
    fn cloned(&self) -> Vec<u8> {
        self.0.clone()
    }
}

impl FromStr for Hex {
    type Err = FromHexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        hex::decode(s.strip_prefix("0x").unwrap_or(s)).map(Hex)
    }
}

fn validate_prefix(prefix: &[u8]) -> Result<(), &'static str> {
    if prefix.len() > 20 {
        Err("prefix cannot be longer than an Ethereum address")
    } else {
        Ok(())
    }
}

fn main() {
    let args = Args::parse();

    if args.list_gpus {
        if let Err(err) = gpu::list_adapters(args.gpu_backend) {
            eprintln!("error: {err}");
            process::exit(1);
        }
        process::exit(0);
    }

    let prefix = args.prefix.expect("missing prefix");
    if let Err(err) = validate_prefix(&prefix.0) {
        eprintln!("error: {err}");
        process::exit(2);
    }

    let threads = NonZeroUsize::new(args.threads);
    let chain = args.chain.details();
    let config = chain
        .as_ref()
        .map(|details| {
            let contracts = details.deployment();
            let setup = args
                .safe_to_l2_setup
                .unwrap_or(contracts.safe_to_l2_setup)
                .non_zero()
                .map(|address| config::SafeToL2Setup {
                    address,
                    l2_singleton: args.l2_singleton.unwrap_or(contracts.safe_l2),
                });

            Configuration {
                proxy: config::Proxy {
                    factory: args.proxy_factory.unwrap_or(contracts.safe_proxy_factory),
                    init_code: args
                        .proxy_init_code
                        .as_ref()
                        .map(Hex::cloned)
                        .unwrap_or(contracts.safe_proxy_init_code.to_vec()),
                    singleton: args.singleton.unwrap_or_else(|| {
                        match (&setup, details.singleton()) {
                            // If we are using the `SafeToL2Setup`, then always
                            // use the `Safe` singleton.
                            (Some(_), _) => contracts.safe,
                            (_, Singleton::Safe) => contracts.safe,
                            (_, Singleton::SafeL2) => contracts.safe_l2,
                        }
                    }),
                },
                account: config::Account {
                    owners: args.owners.clone(),
                    threshold: args.threshold,
                    setup,
                    fallback_handler: args
                        .fallback_handler
                        .or_else(|| contracts.compatibility_fallback_handler.non_zero()),
                    identifier: None,
                },
            }
        })
        .or_else(|| {
            Some(Configuration {
                proxy: config::Proxy {
                    factory: args.proxy_factory?,
                    init_code: args.proxy_init_code?.cloned(),
                    singleton: args.singleton?,
                },
                account: config::Account {
                    owners: args.owners.clone(),
                    threshold: args.threshold,
                    setup: match (
                        args.safe_to_l2_setup.and_then(Address::non_zero),
                        args.l2_singleton,
                    ) {
                        (None, None) => None,
                        // For unsupported chains, if either `SafeToL2Setup` or
                        // `SafeL2` is specified, then both must be specified.
                        (safe_to_l2_setup, l2_singleton) => Some(config::SafeToL2Setup {
                            address: safe_to_l2_setup?,
                            l2_singleton: l2_singleton?,
                        }),
                    },
                    fallback_handler: args.fallback_handler,
                    identifier: None,
                },
            })
        })
        .expect("unsupported chain");
    let explorer = chain.as_ref().map(Details::explorer);

    let setup = || (Safe::new(config.clone()), prefix.0.clone());
    let safe = if args.gpu {
        let mut safe = Safe::new(config.clone());
        if let Err(err) = gpu::search(
            &mut safe,
            &prefix.0,
            gpu::Options {
                backend: args.gpu_backend,
                adapter: args.gpu_adapter,
                batch_size: args.gpu_batch_size,
                allow_software_adapter: args.allow_software_gpu,
                progress: !args.quiet,
            },
        ) {
            eprintln!("GPU mining failed: {err}");
            process::exit(1);
        }
        safe
    } else if let Some(threads) = threads {
        let (sender, receiver) = mpsc::channel();
        let _threads = (0..threads.get())
            .map(|_| {
                thread::spawn({
                    let (mut safe, prefix) = setup();
                    let result = sender.clone();
                    move || {
                        deadbeef_core::search(&mut safe, &prefix);
                        let _ = result.send(safe);
                    }
                })
            })
            .collect::<Vec<_>>();
        receiver.recv().expect("missing result")
    } else {
        let (mut safe, prefix) = setup();
        deadbeef_core::search(&mut safe, &prefix);
        safe
    };

    let transaction = safe.transaction();

    if args.quiet {
        println!("0x{}", hex::encode(&transaction.calldata));
    } else if args.params {
        let factory = explorer
            .map(|explorer| explorer.create_proxy_with_nonce_url(config.proxy.factory.get()))
            .unwrap_or_else(|| config.proxy.factory.to_string());

        println!("address:     {}", safe.creation_address());
        println!("factory:     {}", factory);
        println!("singleton:   {}", config.proxy.singleton);
        println!("initializer: 0x{}", hex::encode(safe.initializer()));
        println!("salt nonce:  0x{}", hex::encode(safe.salt_nonce()));
    } else {
        let (to, data) = config
            .account
            .setup
            .as_ref()
            .map(|setup| (setup.address.get(), setup.encode()))
            .unwrap_or_default();
        let fallback = config
            .account
            .fallback_handler
            .map(NonZeroAddress::get)
            .unwrap_or_default();

        println!("address:     {}", safe.creation_address());
        println!("factory:     {}", config.proxy.factory);
        println!("singleton:   {}", config.proxy.singleton);
        println!("initializer: 0x{}", hex::encode(safe.initializer()));
        println!("salt nonce:  0x{}", hex::encode(safe.salt_nonce()));
        println!("---");
        println!("owners:      {}", config.account.owners[0]);
        for owner in &args.owners[1..] {
            println!("             {}", owner);
        }
        println!("threshold:   {}", config.account.threshold);
        println!("to:          {}", to);
        println!("data:        0x{}", hex::encode(&data));
        println!("fallback:    {}", fallback);
        println!("---");
        println!("calldata:    0x{}", hex::encode(&transaction.calldata));
    }

    process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_gpus_does_not_require_safe_config() {
        let args = Args::try_parse_from(["deadbeef", "--list-gpus"]).unwrap();
        assert!(args.list_gpus);
    }

    #[test]
    fn parses_gpu_options() {
        let args = Args::try_parse_from([
            "deadbeef",
            "--owner",
            "0x1111111111111111111111111111111111111111",
            "--prefix",
            "0x00",
            "--gpu",
            "--gpu-backend",
            "metal",
            "--gpu-adapter",
            "1",
            "--gpu-batch-size",
            "257",
        ])
        .unwrap();

        assert!(args.gpu);
        assert_eq!(args.gpu_backend, gpu::Backend::Metal);
        assert_eq!(args.gpu_adapter, Some(1));
        assert_eq!(args.gpu_batch_size, 257);
    }

    #[test]
    fn gpu_options_require_gpu_mode() {
        for args in [
            &[
                "deadbeef",
                "--owner",
                "0x1111111111111111111111111111111111111111",
                "--prefix",
                "0x00",
                "--gpu-backend",
                "metal",
            ][..],
            &[
                "deadbeef",
                "--owner",
                "0x1111111111111111111111111111111111111111",
                "--prefix",
                "0x00",
                "--gpu-adapter",
                "0",
            ],
            &[
                "deadbeef",
                "--owner",
                "0x1111111111111111111111111111111111111111",
                "--prefix",
                "0x00",
                "--gpu-batch-size",
                "257",
            ],
        ] {
            assert!(Args::try_parse_from(args).is_err());
        }
    }

    #[test]
    fn list_gpus_accepts_gpu_backend() {
        let args =
            Args::try_parse_from(["deadbeef", "--list-gpus", "--gpu-backend", "gl"]).unwrap();

        assert!(args.list_gpus);
        assert_eq!(args.gpu_backend, gpu::Backend::Gl);
    }

    #[test]
    fn list_gpus_rejects_unused_gpu_batch_size() {
        assert!(
            Args::try_parse_from(["deadbeef", "--list-gpus", "--gpu-batch-size", "257"]).is_err()
        );
    }

    #[test]
    fn list_gpus_rejects_unused_gpu_adapter() {
        assert!(Args::try_parse_from(["deadbeef", "--list-gpus", "--gpu-adapter", "0"]).is_err());
    }

    #[test]
    fn rejects_overlong_prefixes() {
        assert!(validate_prefix(&[0; 21]).is_err());
    }
}
