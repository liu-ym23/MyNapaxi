// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_gemm is only implemented for the cuda backend")
}

#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle::DType,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf is only implemented for the cuda backend")
}
