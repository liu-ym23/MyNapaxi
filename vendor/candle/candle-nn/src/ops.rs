//! Tensor ops.
//!

use candle::{CpuStorage, DType, Layout, Module, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.silu()
}

pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

struct Sigmoid;

impl candle::CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        fn fwd<T: num_traits::Float>(v: T) -> T {
            (v.neg().exp() + T::one()).recip()
        }

        // FIXME: using `candle::map_dtype` causes compilation errors.
        let storage = match storage {
            CpuStorage::BF16(slice) => {
                CpuStorage::BF16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F16(slice) => {
                CpuStorage::F16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F32(slice) => {
                CpuStorage::F32(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F64(slice) => {
                CpuStorage::F64(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            _ => Err(candle::Error::UnsupportedDTypeForOp(
                storage.dtype(),
                self.name(),
            ))?,
        };
        Ok((storage, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = (1 - sigmoid(x)) * sigmoid(x)
        let d_dx_sigmoid = res.ones_like()?.sub(res)?.mul(res)?;
        Ok(Some(grad_res.mul(&d_dx_sigmoid)?))
    }
}

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1(Sigmoid)
}

pub fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

pub fn mish(xs: &Tensor) -> Result<Tensor> {
    xs * (1.0 + xs.exp()?)?.log()?.tanh()
}

pub fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
}

pub fn selu(xs: &Tensor, alpha: f32, gamma: f32) -> Result<Tensor> {
    let is_pos = xs.gt(0f32)?;
    let alpha_t = Tensor::full(alpha, xs.dims(), xs.device())?;
    let neg = xs.exp()?.mul(&alpha_t)?.sub(&alpha_t)?;
    let selu = is_pos.where_cond(xs, &neg)?;
    let gamma_t = Tensor::full(gamma, xs.dims(), xs.device())?;
    selu.broadcast_mul(&gamma_t)
}

pub fn dropout(xs: &Tensor, drop_p: f32) -> Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(xs.dtype())? * scale)?;
    xs * mask
}

#[derive(Clone, Debug)]
pub struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl candle::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.forward(xs, train)
    }
}

struct SoftmaxLastDim;

impl candle::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut max = T::neg_infinity();
                    unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                    for (s, d) in src.iter().zip(dst.iter_mut()) {
                        *d = (*s - max).exp();
                    }
                    let mut sum_exp = T::zero();
                    unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                    for d in dst.iter_mut() {
                        *d /= sum_exp
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        match storage {
            CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
            CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
            CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
            CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
            _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
        }
    }
}

pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1_no_bwd(&SoftmaxLastDim)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    eps: f32,
}

impl candle::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let n_rows = el_count / dim_m1;
            let mut dst = vec![T::zero(); el_count];

            fn rms_row<
                T: candle::WithDType
                    + num_traits::Float
                    + num_traits::AsPrimitive<f32>
                    + num_traits::FromPrimitive,
            >(
                src: &[T],
                alpha: &[T],
                n: usize,
                eps: f32,
                dst: &mut [T],
            ) {
                let sum2 = src
                    .iter()
                    .map(|&v| {
                        let v = v.as_();
                        v * v
                    })
                    .sum::<f32>();
                let m = (sum2 / n as f32 + eps).sqrt();
                let m = T::from_f32(m).unwrap_or_else(T::nan);
                for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                    *d = *s / m * *alpha
                }
            }

            if n_rows <= 32 {
                let n = dim_m1;
                for row in 0..n_rows {
                    let src = &src[row * n..(row + 1) * n];
                    let dst = &mut dst[row * n..(row + 1) * n];
                    rms_row(src, alpha, n, eps, dst);
                }
            } else {
                src.par_chunks(dim_m1)
                    .zip(dst.par_chunks_mut(dim_m1))
                    .for_each(|(src, dst)| {
                        let n = src.len();
                        rms_row(src, alpha, n, eps, dst);
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }
}

pub fn rms_norm_slow(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps })
}

#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f32,
}

impl candle::CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            beta: &[T],
            beta_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let beta = match beta_layout.contiguous_offsets() {
                None => candle::bail!("beta has to be contiguous"),
                Some((o1, o2)) => &beta[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), (alpha, beta)) in
                        dst.iter_mut().zip(src.iter()).zip(alpha.iter().zip(beta))
                    {
                        let alpha = alpha.as_();
                        let beta = beta.as_();
                        let d_ = (s.as_() - mean) * inv_std * alpha + beta;
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }
}

pub fn layer_norm_slow(x: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed
        .to_dtype(x_dtype)?
        .broadcast_mul(alpha)?
        .broadcast_add(beta)
}

pub fn layer_norm(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    let hidden_size_beta = beta.dims1()?;
    if hidden_size_xs != hidden_size_alpha || hidden_size_xs != hidden_size_beta {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?} beta: {:?}",
            xs.shape(),
            alpha.shape(),
            beta.shape()
        )
    }
    xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })
}

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
pub fn pixel_shuffle(xs: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c / upscale_factor / upscale_factor;
    xs.reshape((b_size, out_c, upscale_factor, upscale_factor, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b_size, out_c, h * upscale_factor, w * upscale_factor))
}

pub fn pixel_unshuffle(xs: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c * downscale_factor * downscale_factor;
    xs.reshape((
        b_size,
        c,
        h / downscale_factor,
        downscale_factor,
        w / downscale_factor,
        downscale_factor,
    ))?
    .permute((0, 1, 3, 5, 2, 4))?
    .reshape((b_size, out_c, h / downscale_factor, w / downscale_factor))
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    match pad {
        0 => Ok(xs.clone()),
        1 => {
            let (_b_size, _c, h, w) = xs.dims4()?;
            let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
            let xs = Tensor::cat(&[&first, xs, &last], 3)?;
            let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
            Tensor::cat(&[&first, &xs, &last], 2)
        }
        n => candle::bail!("replication-pad with a size of {n} is not supported"),
    }
}

#[derive(Clone, Debug)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
    mask: Option<Tensor>,
    do_causal: bool,
}

impl candle::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("SDPA has no cpu impl")
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `mask`: (bs, qhead, seq, kv_seq)
/// - `do_causal`: Apply causal masking. If this is true, the mask does not need to be provided.
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// Note: For Grouped Query Attention and Multi-Query Attention, the k and v inputs should not be pre-tiled to match q.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Masking is supported
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
///     - Softcapping is not supported.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Sdpa {
            scale,
            softcapping,
            mask: mask.cloned(),
            do_causal,
        },
    )
}
