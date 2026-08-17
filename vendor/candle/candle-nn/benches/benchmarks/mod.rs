pub(crate) mod conv;
pub(crate) mod norm;
pub(crate) mod softmax;

use candle::{Device, Result};

pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => {
                panic!("Cuda device without cuda feature enabled: {device:?}")
            }
            Device::Metal(device) => {
                panic!("Metal device without metal feature enabled: {device:?}")
            }
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => {
                let cpu_type = if false {
                    "accelerate"
                } else if false {
                    "mkl"
                } else {
                    "cpu"
                };
                format!("{}_{}", cpu_type, name.into())
            }
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
        }
    }
}

struct BenchDeviceHandler {
    devices: Vec<Device>,
}

impl BenchDeviceHandler {
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();
        if false {
            devices.push(Device::new_metal(0)?);
        } else if false {
            devices.push(Device::new_cuda(0)?);
        } else {
            devices.push(Device::Cpu);
        }
        Ok(Self { devices })
    }
}
