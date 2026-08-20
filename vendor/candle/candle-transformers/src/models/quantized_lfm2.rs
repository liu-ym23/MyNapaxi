//! Quantized LFM2 / LFM2.5 GGUF inference (hybrid attention + short-conv).

use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    DType, Device, IndexOp, Result, Tensor,
};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Module};
use std::collections::HashMap;

fn get_qtensor<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[&str],
) -> Result<QTensor> {
    let mut last_err = None;
    for name in names {
        match ct.tensor(reader, name, device) {
            Ok(t) => return Ok(t),
            Err(e) => last_err = Some(e),
        }
    }
    candle::bail!(
        "cannot find tensor info for {} ({last_err:?})",
        names.join(" | ")
    )
}

fn get_dequantized<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[&str],
) -> Result<Tensor> {
    get_qtensor(ct, reader, device, names)?.dequantize(device)
}

#[derive(Debug, Clone)]
struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.w1.forward(xs)?;
        let w3 = self.w3.forward(xs)?;
        self.w2.forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
}

#[derive(Debug, Clone)]
struct ShortConvLayer {
    in_proj: QMatMul,
    out_proj: QMatMul,
    conv: Tensor,
    l_cache: usize,
    cache: Option<Tensor>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    span_mlp: tracing::Span,
}

/// Per-layer cache snapshot used for prefix KV / short-conv reuse.
#[derive(Debug, Clone)]
pub struct LayerCache {
    pub kv: Option<(Tensor, Tensor)>,
    pub conv: Option<Tensor>,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b, _n, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&y)
    }
}

impl ShortConvLayer {
    fn forward(&mut self, xs: &Tensor, _index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, hidden)?;
        let c = bcx.narrow(1, hidden, hidden)?;
        let x = bcx.narrow(1, 2 * hidden, hidden)?;
        let bx = (b * &x)?.contiguous()?;

        let mut conv_weight = self.conv.clone();
        if conv_weight.dims().len() == 3 {
            conv_weight = conv_weight.squeeze(1)?;
        } else if conv_weight.dims().len() == 2 && conv_weight.dims2()? == (self.l_cache, hidden) {
            conv_weight = conv_weight.t()?.contiguous()?;
        }
        let conv_weight = conv_weight.contiguous()?;

        let mut conv_out = if seq_len == 1 {
            let mut state = if let Some(cache) = &self.cache {
                cache.clone()
            } else {
                Tensor::zeros((b_sz, hidden, self.l_cache), bx.dtype(), bx.device())?
            };

            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }
            self.cache = Some(state.clone());

            (state * &conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let conv = Conv1d::new(
                conv_weight
                    .reshape((hidden, 1, self.l_cache))?
                    .contiguous()?,
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: hidden,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx.contiguous()?)?;
            out = out.narrow(2, 0, seq_len)?;

            if self.l_cache > 0 {
                let (_, _, cur_len) = bx.dims3()?;
                let start = cur_len.saturating_sub(self.l_cache);
                let mut cache_src = bx.narrow(2, start, cur_len - start)?;
                if cache_src.dims3()?.2 < self.l_cache {
                    let pad = self.l_cache - cache_src.dims3()?.2;
                    let zeros =
                        Tensor::zeros((b_sz, hidden, pad), cache_src.dtype(), cache_src.device())?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                self.cache = Some(cache_src);
            }

            out
        };

        conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn value_to_usize(v: &gguf_file::Value) -> Result<usize> {
    use gguf_file::Value::*;
    match v {
        U8(x) => Ok(*x as usize),
        I8(x) => Ok(*x as usize),
        U16(x) => Ok(*x as usize),
        I16(x) => Ok(*x as usize),
        U32(x) => Ok(*x as usize),
        I32(x) => Ok(*x as usize),
        U64(x) => Ok(*x as usize),
        I64(x) => Ok(*x as usize),
        F32(x) => Ok(*x as usize),
        F64(x) => Ok(*x as usize),
        Bool(x) => Ok(usize::from(*x)),
        String(_) => candle::bail!("unexpected string metadata"),
        Array(_) => candle::bail!("array should be handled separately"),
    }
}

fn read_usize_list(v: &gguf_file::Value, len: usize) -> Result<Vec<usize>> {
    match v {
        gguf_file::Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                out.push(value_to_usize(item)?);
            }
            if out.len() == len {
                Ok(out)
            } else if out.len() == 1 {
                Ok(vec![out[0]; len])
            } else {
                candle::bail!(
                    "unexpected array length in metadata, expected {len} got {}",
                    out.len()
                )
            }
        }
        _ => Ok(vec![value_to_usize(v)?; len]),
    }
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("lfm2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv_meta = md_get("lfm2.attention.head_count_kv")?;
        let embedding_length = md_get("lfm2.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("lfm2.context_length")?.to_u32()? as usize;
        let block_count = md_get("lfm2.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("lfm2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("lfm2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(1_000_000f32);
        let l_cache = md_get("lfm2.shortconv.l_cache")?.to_u32()? as usize;

        let head_count_kv = read_usize_list(head_count_kv_meta, block_count)?;
        let head_dim = embedding_length / head_count;
        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = get_qtensor(
            &ct,
            reader,
            device,
            &[
                "token_embd.weight",
                "tok_embeddings.weight",
                "model.embed_tokens.weight",
            ],
        )?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;

        let norm = RmsNorm::from_qtensor(
            get_qtensor(
                &ct,
                reader,
                device,
                &[
                    "output_norm.weight",
                    "embedding_norm.weight",
                    "model.embedding_norm.weight",
                    "model.embedding_norm",
                    "token_embd_norm.weight",
                ],
            )?,
            rms_norm_eps,
        )?;
        let output_q = get_qtensor(
            &ct,
            reader,
            device,
            &[
                "output.weight",
                "lm_head.weight",
                "model.output.weight",
                "model.lm_head.weight",
            ],
        )
        .or_else(|_| {
            get_qtensor(
                &ct,
                reader,
                device,
                &[
                    "token_embd.weight",
                    "tok_embeddings.weight",
                    "model.embed_tokens.weight",
                ],
            )
        })?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let is_attention = head_count_kv.get(layer_idx).copied().unwrap_or(head_count) > 0;

            let operator_norm = get_qtensor(
                &ct,
                reader,
                device,
                &[
                    &format!("{prefix}.attn_norm.weight"),
                    &format!("{prefix}.operator_norm.weight"),
                    &format!("{prefix}.attention_norm.weight"),
                ],
            )?;
            let ffn_norm = get_qtensor(
                &ct,
                reader,
                device,
                &[
                    &format!("{prefix}.ffn_norm.weight"),
                    &format!("{prefix}.ffn_norm"),
                ],
            )?;
            let mlp = {
                let w1 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.ffn_gate.weight"),
                        &format!("{prefix}.feed_forward.w1.weight"),
                        &format!("{prefix}.mlp.gate_proj.weight"),
                    ],
                )?;
                let w2 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.ffn_down.weight"),
                        &format!("{prefix}.feed_forward.w2.weight"),
                        &format!("{prefix}.mlp.down_proj.weight"),
                    ],
                )?;
                let w3 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.ffn_up.weight"),
                        &format!("{prefix}.feed_forward.w3.weight"),
                        &format!("{prefix}.mlp.up_proj.weight"),
                    ],
                )?;
                Mlp {
                    w1: QMatMul::from_qtensor(w1)?,
                    w2: QMatMul::from_qtensor(w2)?,
                    w3: QMatMul::from_qtensor(w3)?,
                }
            };

            let kind = if is_attention {
                let n_kv_head = head_count_kv[layer_idx];
                let wq = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.attn_q.weight"),
                        &format!("{prefix}.self_attn.q_proj.weight"),
                    ],
                )?;
                let wk = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.attn_k.weight"),
                        &format!("{prefix}.self_attn.k_proj.weight"),
                    ],
                )?;
                let wv = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.attn_v.weight"),
                        &format!("{prefix}.self_attn.v_proj.weight"),
                    ],
                )?;
                let wo = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.attn_output.weight"),
                        &format!("{prefix}.self_attn.out_proj.weight"),
                    ],
                )?;
                let q_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.attn_q_norm.weight"),
                        &format!("{prefix}.self_attn.q_layernorm.weight"),
                        &format!("{prefix}.attention.q_norm.weight"),
                    ],
                )?;
                let k_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.attn_k_norm.weight"),
                        &format!("{prefix}.self_attn.k_layernorm.weight"),
                        &format!("{prefix}.attention.k_norm.weight"),
                    ],
                )?;

                LayerKind::Attention(AttentionLayer {
                    wq: QMatMul::from_qtensor(wq)?,
                    wk: QMatMul::from_qtensor(wk)?,
                    wv: QMatMul::from_qtensor(wv)?,
                    wo: QMatMul::from_qtensor(wo)?,
                    q_norm: RmsNorm::from_qtensor(q_norm, rms_norm_eps)?,
                    k_norm: RmsNorm::from_qtensor(k_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    kv_cache: None,
                    span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                })
            } else {
                let in_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.shortconv.in_proj.weight"),
                        &format!("{prefix}.conv.in_proj.weight"),
                    ],
                )?;
                let out_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.shortconv.out_proj.weight"),
                        &format!("{prefix}.conv.out_proj.weight"),
                    ],
                )?;
                let conv = get_dequantized(
                    &ct,
                    reader,
                    device,
                    &[
                        &format!("{prefix}.shortconv.conv.weight"),
                        &format!("{prefix}.conv.conv.weight"),
                        &format!("{prefix}.shortconv.conv"),
                    ],
                )?;
                LayerKind::ShortConv(ShortConvLayer {
                    in_proj: QMatMul::from_qtensor(in_proj)?,
                    out_proj: QMatMul::from_qtensor(out_proj)?,
                    conv,
                    l_cache,
                    cache: None,
                })
            };

            layers.push(LayerWeights {
                operator_norm: RmsNorm::from_qtensor(operator_norm, rms_norm_eps)?,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                mlp,
                kind,
                span_mlp: tracing::span!(tracing::Level::TRACE, "ffn"),
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output_q)?,
            masks: HashMap::new(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            match &mut layer.kind {
                LayerKind::Attention(attn) => attn.kv_cache = None,
                LayerKind::ShortConv(conv) => conv.cache = None,
            }
        }
    }

    pub fn snapshot_kv_cache(&self) -> Vec<LayerCache> {
        self.layers
            .iter()
            .map(|layer| match &layer.kind {
                LayerKind::Attention(attn) => LayerCache {
                    kv: attn.kv_cache.clone(),
                    conv: None,
                },
                LayerKind::ShortConv(conv) => LayerCache {
                    kv: None,
                    conv: conv.cache.clone(),
                },
            })
            .collect()
    }

    pub fn restore_kv_cache(&mut self, snapshot: &[LayerCache]) {
        for (layer, cached) in self.layers.iter_mut().zip(snapshot.iter()) {
            match &mut layer.kind {
                LayerKind::Attention(attn) => attn.kv_cache = cached.kv.clone(),
                LayerKind::ShortConv(conv) => conv.cache = cached.conv.clone(),
            }
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let _enter = self.span.enter();
        let mut hidden = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            hidden = match &mut layer.kind {
                LayerKind::Attention(attn) => attn.forward(&normed, mask.as_ref(), index_pos)?,
                LayerKind::ShortConv(conv) => conv.forward(&normed, index_pos)?,
            };
            hidden = (hidden + residual)?;

            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            let _enter = layer.span_mlp.enter();
            let ff = layer.mlp.forward(&ff)?;
            hidden = (ff + residual)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&hidden)
    }
}
