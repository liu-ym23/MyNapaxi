//! Rotary Embeddings
//!
use candle::{CpuStorage, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Interleaved variant of rotary embeddings.
/// The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
/// The resulting y0 and y1 are also interleaved with:
///   y0 = x0*cos - x1*sin
///   y1 = x0*sin + x1*cos
#[derive(Debug, Clone)]
struct RotaryEmbI;

impl candle::CustomOp3 for RotaryEmbI {
    fn name(&self) -> &'static str {
        "rotary-emb-int"
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
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            if t == 1 {
                for bh_i in 0..b * h {
                    let off = bh_i * d;
                    for i_over_2 in 0..d / 2 {
                        let i = off + 2 * i_over_2;
                        let rope_i = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_over_2 + b_i * d / 2
                        } else {
                            i_over_2
                        };
                        dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                        dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                    }
                }
            } else {
                src.par_chunks(t * d)
                    .zip(dst.par_chunks_mut(t * d))
                    .enumerate()
                    .for_each(|(bh_i, (src, dst))| {
                        for i_over_2 in 0..t * d / 2 {
                            let i = 2 * i_over_2;
                            let rope_i = if unbatched_rope {
                                let b_i = bh_i / h;
                                i_over_2 + b_i * t * d / 2
                            } else {
                                i_over_2
                            };
                            dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                            dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                        }
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }
}

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle::bail!("inconsistent batch size in rope {b_sz} {cs:?}",)
            }
            Ok((t, d))
        }
        _ => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

pub fn rope_i(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbI)
}

pub fn rope_i_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    let rope = rope.flatten_from(D::Minus2)?;
    Ok(rope)
}

/// Contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmb;

impl candle::CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "rotary-emb"
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
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            if t == 1 {
                for bh_i in 0..b * h {
                    let off = bh_i * d;
                    for i_d in 0..d / 2 {
                        let i1 = off + i_d;
                        let i2 = i1 + d / 2;
                        let i_cs = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_d + b_i * d / 2
                        } else {
                            i_d
                        };
                        dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                        dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                    }
                }
            } else {
                src.par_chunks(t * d)
                    .zip(dst.par_chunks_mut(t * d))
                    .enumerate()
                    .for_each(|(bh_i, (src, dst))| {
                        for i_t in 0..t {
                            for i_d in 0..d / 2 {
                                let i1 = i_t * d + i_d;
                                let i2 = i1 + d / 2;
                                let i_cs = i_t * (d / 2) + i_d;
                                let i_cs = if unbatched_rope {
                                    let b_i = bh_i / h;
                                    i_cs + b_i * t * d / 2
                                } else {
                                    i_cs
                                };
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }
}

pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmb)
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

pub fn rope_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _h, seq_len, _n_embd) = x.dims4()?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// T (seqlen)/H (num-heads)/D (head-dim) contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmbThd;

impl candle::CustomOp3 for RotaryEmbThd {
    fn name(&self) -> &'static str {
        "rotary-emb"
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
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * h * d)
                .zip(dst.par_chunks_mut(t * h * d))
                .enumerate()
                .for_each(|(b_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            for i_h in 0..h {
                                let i1 = i_t * h * d + i_h * d + i_d;
                                let i2 = i1 + d / 2;
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, t, h, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }
}

pub fn rope_thd(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len, _n_head, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbThd)
}
