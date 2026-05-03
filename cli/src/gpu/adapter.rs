use super::{err, Backend, Result};

pub(super) const NO_ADAPTERS: &str = "no matching GPU adapters found";
const NO_HARDWARE_ADAPTERS: &str = "only software-rendered GPU adapters were found";

pub(super) fn adapters(backend: Backend) -> Vec<wgpu::Adapter> {
    let backends = backend.to_wgpu_bitset();
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    pollster::block_on(instance.enumerate_adapters(backends))
}

pub(super) fn select_adapter(
    backend: Backend,
    adapter_index: Option<usize>,
    allow_software_adapter: bool,
) -> Result<wgpu::Adapter> {
    let adapters = adapters(backend);
    if adapters.is_empty() {
        return Err(err(NO_ADAPTERS));
    }

    if let Some(index) = adapter_index {
        let adapter = adapters
            .into_iter()
            .nth(index)
            .ok_or_else(|| err(format!("GPU adapter {index} was not found")))?;
        ensure_adapter_can_mine(&adapter, allow_software_adapter)?;
        return Ok(adapter);
    }

    adapters
        .into_iter()
        .find(|adapter| adapter_can_mine(adapter, allow_software_adapter))
        .ok_or_else(|| {
            err(format!(
                "{NO_HARDWARE_ADAPTERS}; use CPU mining or pass --allow-software-gpu for diagnostics"
            ))
        })
}

fn ensure_adapter_can_mine(adapter: &wgpu::Adapter, allow_software_adapter: bool) -> Result<()> {
    if adapter_can_mine(adapter, allow_software_adapter) {
        return Ok(());
    }

    let info = adapter.get_info();
    Err(err(format!(
        "GPU adapter is a software renderer, not a hardware GPU: {}. Use CPU mining or pass \
         --allow-software-gpu for diagnostics",
        adapter_description(&info)
    )))
}

fn adapter_can_mine(adapter: &wgpu::Adapter, allow_software_adapter: bool) -> bool {
    allow_software_adapter || !is_software_adapter(adapter)
}

fn is_software_adapter(adapter: &wgpu::Adapter) -> bool {
    // Per `wgpu::DeviceType`, `Cpu` means "Cpu / Software Rendering".
    adapter.get_info().device_type == wgpu::DeviceType::Cpu
}

pub(super) fn adapter_description(info: &wgpu::AdapterInfo) -> String {
    let mut details = vec![
        format!("{:?}", info.device_type),
        format!("backend: {:?}", info.backend),
        format!("vendor: 0x{:04x}", info.vendor),
        format!("device: 0x{:04x}", info.device),
    ];
    if !info.driver.is_empty() {
        details.push(format!("driver: {}", info.driver));
    }
    if !info.driver_info.is_empty() {
        details.push(format!("driver info: {}", info.driver_info));
    }
    if info.device_type == wgpu::DeviceType::Cpu {
        details.push("software renderer".to_owned());
    }

    format!("{} ({})", info.name, details.join(", "))
}
