use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, Axis, Ix2, Ix3, Ix4};
use ::ort::value::{Tensor, TensorRef};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::case::CaseSpec;
use crate::io::npz::{write_npz_f32, NpzData};
use crate::ort::session::OrtSessionManager;
use crate::scheduler::{ode_step, resolve_timesteps, x0_from_noise};

const DEFAULT_DIT_INSTRUCTION: &str = "Fill the audio semantic mask based on the given conditions:";

#[derive(Debug, Deserialize)]
struct IoContract {
    #[serde(default)]
    inputs: HashMap<String, Vec<String>>,
    #[serde(default)]
    outputs: HashMap<String, Vec<String>>,
}

pub struct CorePipeline {
    sessions: OrtSessionManager,
    contract: IoContract,
    qwen_tokenizer: Option<Tokenizer>,
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub online_qwen_embed: bool,
    pub prefer_precomputed_condition: bool,
    pub qwen_tokenizer: PathBuf,
    pub text_override: Option<String>,
    pub lyrics_override: Option<String>,
    pub text_max_tokens: usize,
    pub lyric_max_tokens: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            online_qwen_embed: false,
            prefer_precomputed_condition: true,
            qwen_tokenizer: PathBuf::from("checkpoints/Qwen3-Embedding-0.6B/tokenizer.json"),
            text_override: None,
            lyrics_override: None,
            text_max_tokens: 256,
            lyric_max_tokens: 2048,
        }
    }
}

fn format_instruction(instruction: &str) -> String {
    if instruction.ends_with(':') {
        instruction.to_string()
    } else {
        format!("{instruction}:")
    }
}

fn value_to_duration_text(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                format!("{i} seconds")
            } else if let Some(u) = n.as_u64() {
                format!("{u} seconds")
            } else if let Some(f) = n.as_f64() {
                format!("{} seconds", f as i64)
            } else {
                "30 seconds".to_string()
            }
        }
        Some(serde_json::Value::String(s)) => s.clone(),
        _ => "30 seconds".to_string(),
    }
}

fn meta_to_text(meta: &HashMap<String, serde_json::Value>) -> String {
    let bpm = meta
        .get("bpm")
        .or_else(|| meta.get("tempo"))
        .map(|v| v.to_string().trim_matches('"').to_string())
        .unwrap_or_else(|| "N/A".to_string());
    let timesig = meta
        .get("timesignature")
        .or_else(|| meta.get("time_signature"))
        .map(|v| v.to_string().trim_matches('"').to_string())
        .unwrap_or_else(|| "N/A".to_string());
    let keyscale = meta
        .get("keyscale")
        .or_else(|| meta.get("key"))
        .or_else(|| meta.get("scale"))
        .map(|v| v.to_string().trim_matches('"').to_string())
        .unwrap_or_else(|| "N/A".to_string());
    let duration = value_to_duration_text(meta.get("duration").or_else(|| meta.get("length")));
    format!("- bpm: {bpm}\n- timesignature: {timesig}\n- keyscale: {keyscale}\n- duration: {duration}\n")
}

fn format_lyrics(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
}

impl CorePipeline {
    pub fn new(onnx_dir: PathBuf) -> Result<Self> {
        let contract_path = onnx_dir.join("io_contract_core.json");
        let contract_text =
            std::fs::read_to_string(&contract_path).with_context(|| format!("read contract: {}", contract_path.display()))?;
        let contract: IoContract = serde_json::from_str(&contract_text).context("parse io_contract_core.json")?;
        Ok(Self {
            sessions: OrtSessionManager::new(onnx_dir),
            contract,
            qwen_tokenizer: None,
        })
    }

    pub fn generate(
        &mut self,
        case: &CaseSpec,
        inputs_npz: &Path,
        out_npz: &Path,
        out_wav: Option<&Path>,
        opts: &GenerateOptions,
    ) -> Result<()> {
        let mut required_f32 = vec![
            "refer_audio_acoustic_hidden_states_packed",
            "refer_audio_order_mask",
            "src_latents",
            "chunk_masks",
            "is_covers",
            "precomputed_lm_hints_25hz",
        ];
        if !opts.online_qwen_embed {
            required_f32.extend([
                "text_hidden_states",
                "text_attention_mask",
                "lyric_hidden_states",
                "lyric_attention_mask",
            ]);
        }
        let optional_f32 = [
            "xt_steps",
            "latent_masks",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "context_latents",
        ];

        let npz = NpzData::load(inputs_npz, &required_f32, &optional_f32)?;

        let refer_audio_acoustic_hidden_states_packed =
            npz.require_f32("refer_audio_acoustic_hidden_states_packed")?.to_owned();
        let src_latents = npz.require_f32("src_latents")?.to_owned();
        let chunk_masks = npz.require_f32("chunk_masks")?.to_owned();
        let precomputed_lm_hints_25hz = npz.require_f32("precomputed_lm_hints_25hz")?.to_owned();

        let refer_audio_order_mask_f = npz.require_f32("refer_audio_order_mask")?.to_owned();
        let refer_audio_order_mask = refer_audio_order_mask_f.mapv(|v| v as i64);
        let is_covers_f = npz.require_f32("is_covers")?.to_owned();
        let is_covers = is_covers_f.mapv(|v| v != 0.0);

        let (text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask) =
            if opts.online_qwen_embed {
                self.build_text_features(case, &npz, opts)?
            } else {
                (
                    npz.require_f32("text_hidden_states")?.to_owned(),
                    npz.require_f32("text_attention_mask")?.to_owned(),
                    npz.require_f32("lyric_hidden_states")?.to_owned(),
                    npz.require_f32("lyric_attention_mask")?.to_owned(),
                )
            };

        let (encoder_hidden_states, encoder_attention_mask, context_latents) =
            if opts.prefer_precomputed_condition {
                match (
                    npz.get_f32("encoder_hidden_states"),
                    npz.get_f32("encoder_attention_mask"),
                    npz.get_f32("context_latents"),
                ) {
                    (Some(h), Some(m), Some(c)) => (
                        h.to_owned()
                            .into_dimensionality::<Ix3>()
                            .context("encoder_hidden_states shape")?,
                        m.to_owned(),
                        c.to_owned()
                            .into_dimensionality::<Ix3>()
                            .context("context_latents shape")?,
                    ),
                    _ => self.run_condition_encoder(
                        &text_hidden_states,
                        &text_attention_mask,
                        &lyric_hidden_states,
                        &lyric_attention_mask,
                        &refer_audio_acoustic_hidden_states_packed,
                        &refer_audio_order_mask,
                        &src_latents,
                        &chunk_masks,
                        &is_covers,
                        &precomputed_lm_hints_25hz,
                    )?,
                }
            } else {
                self.run_condition_encoder(
                    &text_hidden_states,
                    &text_attention_mask,
                    &lyric_hidden_states,
                    &lyric_attention_mask,
                    &refer_audio_acoustic_hidden_states_packed,
                    &refer_audio_order_mask,
                    &src_latents,
                    &chunk_masks,
                    &is_covers,
                    &precomputed_lm_hints_25hz,
                )?
            };

        let xt = if let Some(xt_steps) = npz.get_f32("xt_steps") {
            let xt_steps = xt_steps.to_owned().into_dimensionality::<Ix4>().context("xt_steps shape")?;
            xt_steps.index_axis(Axis(0), 0).to_owned()
        } else {
            src_latents.clone()
                .into_dimensionality::<Ix3>()
                .context("src_latents shape")?
        };

        let latent_masks = npz.get_f32("latent_masks").map(|m| {
            m.to_owned()
                .into_dimensionality::<Ix2>()
                .context("latent_masks shape")
        });
        let latent_masks = match latent_masks {
            Some(Ok(v)) => Some(v),
            Some(Err(e)) => return Err(e),
            None => None,
        };

        let (pred_latents, xt_steps, vt_steps) = self.run_dit_loop(
            case,
            xt,
            encoder_hidden_states.clone(),
            encoder_attention_mask.clone(),
            context_latents.clone(),
            latent_masks,
        )?;

        let xt_stack = stack_steps(&xt_steps, "xt_steps")?;
        let vt_stack = stack_steps(&vt_steps, "vt_steps")?;

        let entries: Vec<(&str, ArrayD<f32>)> = vec![
            ("pred_latents", pred_latents.clone().into_dyn()),
            ("xt_steps", xt_stack.into_dyn()),
            ("vt_steps", vt_stack.into_dyn()),
            ("encoder_hidden_states", encoder_hidden_states.clone().into_dyn()),
            ("encoder_attention_mask", encoder_attention_mask.clone()),
            ("context_latents", context_latents.clone().into_dyn()),
        ];
        write_npz_f32(out_npz, &entries)?;

        if let Some(wav_path) = out_wav {
            let audio = self.decode_audio(&pred_latents)?;
            let audio_0 = audio.index_axis(Axis(0), 0).to_owned();
            crate::audio::write_wav(wav_path, &audio_0, 48000)?;
        }

        Ok(())
    }

    fn build_text_features(
        &mut self,
        case: &CaseSpec,
        npz: &NpzData,
        opts: &GenerateOptions,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        let text = opts
            .text_override
            .as_ref()
            .map(|v| v.as_str())
            .unwrap_or(case.caption.as_str())
            .trim()
            .to_string();
        let lyrics = opts
            .lyrics_override
            .as_ref()
            .map(|v| v.as_str())
            .unwrap_or(case.lyrics.as_str())
            .trim()
            .to_string();
        let instruction = format_instruction(DEFAULT_DIT_INSTRUCTION);
        let meta_text = meta_to_text(&case.metadata);
        let language = case
            .metadata
            .get("language")
            .or_else(|| case.metadata.get("vocal_language"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let text_prompt = format!(
            "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n",
            instruction, text, meta_text
        );
        let lyrics_prompt = format_lyrics(&lyrics, &language);

        let text_pair = if text.is_empty() {
            match (npz.get_f32("text_hidden_states"), npz.get_f32("text_attention_mask")) {
                (Some(h), Some(m)) => (h.to_owned(), m.to_owned()),
                _ => self.encode_qwen_text(" ", opts.text_max_tokens, &opts.qwen_tokenizer, false)?,
            }
        } else {
            self.encode_qwen_text(&text_prompt, opts.text_max_tokens, &opts.qwen_tokenizer, false)?
        };

        let lyric_pair = if lyrics.is_empty() {
            match (npz.get_f32("lyric_hidden_states"), npz.get_f32("lyric_attention_mask")) {
                (Some(h), Some(m)) => (h.to_owned(), m.to_owned()),
                _ => self.encode_qwen_text(" ", opts.lyric_max_tokens, &opts.qwen_tokenizer, true)?,
            }
        } else {
            self.encode_qwen_text(&lyrics_prompt, opts.lyric_max_tokens, &opts.qwen_tokenizer, true)?
        };

        Ok((text_pair.0, text_pair.1, lyric_pair.0, lyric_pair.1))
    }

    fn encode_qwen_text(
        &mut self,
        text: &str,
        max_tokens: usize,
        tokenizer_path: &Path,
        use_token_embedding: bool,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        if self.qwen_tokenizer.is_none() {
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("load tokenizer {}: {}", tokenizer_path.display(), e))?;
            self.qwen_tokenizer = Some(tokenizer);
        }
        let tokenizer = self.qwen_tokenizer.as_ref().expect("tokenizer initialized");
        let normalized = if text.trim().is_empty() { " " } else { text };
        let encoding = tokenizer
            .encode(normalized, true)
            .map_err(|e| anyhow::anyhow!("tokenize text for qwen embed: {}", e))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().copied().map(|v| v as i64).collect();
        if ids.is_empty() {
            ids.push(0);
        }
        if max_tokens > 0 && ids.len() > max_tokens {
            ids.truncate(max_tokens);
        }
        let t = ids.len();
        let input_ids = Array2::<i64>::from_shape_vec((1, t), ids).context("qwen input_ids shape")?;
        let attention_mask = Array2::<i64>::ones((1, t));

        let (hidden, _) = if use_token_embedding {
            let token_model_name = "qwen3_token_embedding_0p6.onnx";
            match self.sessions.get_mut(token_model_name) {
                Ok(sess) => {
                    let outputs = sess.run(::ort::inputs![
                        "input_ids" => TensorRef::from_array_view(&input_ids)?,
                    ])?;
                    let hidden = outputs
                        .get("token_embeddings")
                        .context("missing token_embeddings")?
                        .try_extract_array::<f32>()?
                        .to_owned()
                        .into_dyn();
                    (hidden, "token_embeddings")
                }
                Err(_) => {
                    let sess = self.sessions.get_mut("qwen3_embedding_0p6.onnx")?;
                    let outputs = sess.run(::ort::inputs![
                        "input_ids" => TensorRef::from_array_view(&input_ids)?,
                        "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
                    ])?;
                    let hidden = outputs
                        .get("last_hidden_state")
                        .context("missing last_hidden_state")?
                        .try_extract_array::<f32>()?
                        .to_owned()
                        .into_dyn();
                    (hidden, "last_hidden_state")
                }
            }
        } else {
            let sess = self.sessions.get_mut("qwen3_embedding_0p6.onnx")?;
            let outputs = sess.run(::ort::inputs![
                "input_ids" => TensorRef::from_array_view(&input_ids)?,
                "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
            ])?;
            let hidden = outputs
                .get("last_hidden_state")
                .context("missing last_hidden_state")?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dyn();
            (hidden, "last_hidden_state")
        };
        let mask_f = attention_mask.mapv(|v| v as f32).into_dyn();
        Ok((hidden, mask_f))
    }

    fn run_condition_encoder(
        &mut self,
        text_hidden_states: &ArrayD<f32>,
        text_attention_mask: &ArrayD<f32>,
        lyric_hidden_states: &ArrayD<f32>,
        lyric_attention_mask: &ArrayD<f32>,
        refer_audio_acoustic_hidden_states_packed: &ArrayD<f32>,
        refer_audio_order_mask: &ArrayD<i64>,
        src_latents: &ArrayD<f32>,
        chunk_masks: &ArrayD<f32>,
        is_covers: &ArrayD<bool>,
        precomputed_lm_hints_25hz: &ArrayD<f32>,
    ) -> Result<(Array3<f32>, ArrayD<f32>, Array3<f32>)> {
        let sess = self.sessions.get_mut("condition_encoder.onnx")?;

        let outputs = sess.run(::ort::inputs![
            "text_hidden_states" => TensorRef::from_array_view(text_hidden_states)?,
            "text_attention_mask" => TensorRef::from_array_view(text_attention_mask)?,
            "lyric_hidden_states" => TensorRef::from_array_view(lyric_hidden_states)?,
            "lyric_attention_mask" => TensorRef::from_array_view(lyric_attention_mask)?,
            "refer_audio_acoustic_hidden_states_packed" => TensorRef::from_array_view(refer_audio_acoustic_hidden_states_packed)?,
            "refer_audio_order_mask" => TensorRef::from_array_view(refer_audio_order_mask)?,
            "src_latents" => TensorRef::from_array_view(src_latents)?,
            "chunk_masks" => TensorRef::from_array_view(chunk_masks)?,
            "is_covers" => TensorRef::from_array_view(is_covers)?,
            "precomputed_lm_hints_25hz" => TensorRef::from_array_view(precomputed_lm_hints_25hz)?,
        ])?;

        let encoder_hidden_states = outputs
            .get("encoder_hidden_states")
            .context("missing encoder_hidden_states")?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .context("encoder_hidden_states shape")?;
        let encoder_attention_mask_bool = outputs
            .get("encoder_attention_mask")
            .context("missing encoder_attention_mask")?
            .try_extract_array::<bool>()?
            .to_owned();
        let encoder_attention_mask = bool_to_f32(&encoder_attention_mask_bool);
        let context_latents = outputs
            .get("context_latents")
            .context("missing context_latents")?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .context("context_latents shape")?;

        Ok((encoder_hidden_states, encoder_attention_mask, context_latents))
    }

    fn run_dit_loop(
        &mut self,
        case: &CaseSpec,
        mut xt: Array3<f32>,
        encoder_hidden_states: Array3<f32>,
        encoder_attention_mask: ArrayD<f32>,
        context_latents: Array3<f32>,
        latent_masks: Option<Array2<f32>>,
    ) -> Result<(Array3<f32>, Vec<Array3<f32>>, Vec<Array3<f32>>)> {
        let dit_inputs = self.contract.inputs.get("dit_decoder").cloned().unwrap_or_default();
        let use_attention_mask = dit_inputs.iter().any(|v| v == "attention_mask");
        let dit_prefill_inputs = self
            .contract
            .inputs
            .get("dit_prefill_kv")
            .cloned()
            .unwrap_or_default();
        let dit_decode_inputs = self
            .contract
            .inputs
            .get("dit_decode_kv")
            .cloned()
            .unwrap_or_default();
        let dit_prefill_outputs = self
            .contract
            .outputs
            .get("dit_prefill_kv")
            .cloned()
            .unwrap_or_default();
        let dit_decode_outputs = self
            .contract
            .outputs
            .get("dit_decode_kv")
            .cloned()
            .unwrap_or_default();
        let use_dit_kv = !dit_prefill_inputs.is_empty()
            && !dit_decode_inputs.is_empty()
            && !dit_prefill_outputs.is_empty()
            && !dit_decode_outputs.is_empty()
            && self.sessions.exists("dit_prefill_kv.onnx")
            && self.sessions.exists("dit_decode_kv.onnx");
        if !use_dit_kv {
            eprintln!("[acestep_ort] dit kv models unavailable; fallback to dit_decoder.onnx");
        }

        let timesteps = resolve_timesteps(case.shift, None, case.inference_steps as usize);

        let orig_len = xt.shape()[1];
        let pad_len = (2 - (orig_len % 2)) % 2;
        let context_latents_padded = if pad_len > 0 {
            pad_3d(&context_latents, pad_len)
        } else {
            context_latents.clone()
        };
        let attention_mask_padded = if use_attention_mask {
            let mask = latent_masks.unwrap_or_else(|| Array2::<f32>::ones((xt.shape()[0], xt.shape()[1])));
            if pad_len > 0 {
                pad_2d(&mask, pad_len)
            } else {
                mask
            }
        } else {
            Array2::<f32>::zeros((1, 1))
        };

        let mut xt_steps = Vec::new();
        let mut vt_steps = Vec::new();
        let mut cache_map: HashMap<String, ArrayD<f32>> = HashMap::new();
        let prefill_present_names = dit_prefill_outputs
            .iter()
            .filter(|n| n.starts_with("present_"))
            .cloned()
            .collect::<Vec<_>>();
        let decode_present_names = dit_decode_outputs
            .iter()
            .filter(|n| n.starts_with("present_"))
            .cloned()
            .collect::<Vec<_>>();

        for (idx, t) in timesteps.iter().enumerate() {
            let t_vec = Array1::<f32>::from_elem(xt.shape()[0], *t);
            let xt_in = if pad_len > 0 { pad_3d(&xt, pad_len) } else { xt.clone() };
            let outputs = if use_dit_kv {
                let mut inputs = ::ort::inputs![
                    "hidden_states" => TensorRef::from_array_view(&xt_in)?,
                    "timestep" => Tensor::from_array(t_vec.clone())?,
                    "timestep_r" => Tensor::from_array(t_vec.clone())?,
                    "encoder_hidden_states" => TensorRef::from_array_view(&encoder_hidden_states)?,
                    "encoder_attention_mask" => TensorRef::from_array_view(&encoder_attention_mask)?,
                    "context_latents" => TensorRef::from_array_view(&context_latents_padded)?,
                    "attention_mask" => TensorRef::from_array_view(&attention_mask_padded)?,
                ];
                if idx > 0 {
                    for name in &dit_decode_inputs {
                        if let Some(suffix) = name.strip_prefix("past_") {
                            let present_name = format!("present_{suffix}");
                            let value = cache_map
                                .get(present_name.as_str())
                                .with_context(|| format!("missing cache tensor: {present_name}"))?;
                            inputs.push((name.clone().into(), TensorRef::from_array_view(value)?.into()));
                        }
                    }
                    let sess = self.sessions.get_mut("dit_decode_kv.onnx")?;
                    let outputs = sess.run(inputs)?;
                    cache_map.clear();
                    for name in &decode_present_names {
                        let value = outputs
                            .get(name.as_str())
                            .with_context(|| format!("missing output {name}"))?
                            .try_extract_array::<f32>()?
                            .to_owned()
                            .into_dyn();
                        cache_map.insert(name.clone(), value);
                    }
                    outputs
                } else {
                    let sess = self.sessions.get_mut("dit_prefill_kv.onnx")?;
                    let outputs = sess.run(inputs)?;
                    cache_map.clear();
                    for name in &prefill_present_names {
                        let value = outputs
                            .get(name.as_str())
                            .with_context(|| format!("missing output {name}"))?
                            .try_extract_array::<f32>()?
                            .to_owned()
                            .into_dyn();
                        cache_map.insert(name.clone(), value);
                    }
                    outputs
                }
            } else {
                let mut inputs = ::ort::inputs![
                    "hidden_states" => TensorRef::from_array_view(&xt_in)?,
                    "timestep" => Tensor::from_array(t_vec.clone())?,
                    "timestep_r" => Tensor::from_array(t_vec.clone())?,
                    "encoder_hidden_states" => TensorRef::from_array_view(&encoder_hidden_states)?,
                    "encoder_attention_mask" => TensorRef::from_array_view(&encoder_attention_mask)?,
                    "context_latents" => TensorRef::from_array_view(&context_latents_padded)?,
                ];
                if use_attention_mask {
                    inputs.push((
                        "attention_mask".into(),
                        TensorRef::from_array_view(&attention_mask_padded)?.into(),
                    ));
                }
                let sess = self.sessions.get_mut("dit_decoder.onnx")?;
                sess.run(inputs)?
            };
            let vt = outputs
                .get("vt")
                .context("missing vt output")?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality::<Ix3>()
                .context("vt shape")?;
            let vt = if pad_len > 0 {
                vt.slice(s![.., 0..orig_len, ..]).to_owned()
            } else {
                vt
            };

            xt_steps.push(xt.clone());
            vt_steps.push(vt.clone());

            if idx == timesteps.len() - 1 {
                xt = x0_from_noise(&xt, &vt, *t);
                break;
            }
            xt = ode_step(&xt, &vt, *t, timesteps[idx + 1]);
        }

        Ok((xt, xt_steps, vt_steps))
    }

    fn decode_audio(&mut self, pred_latents: &Array3<f32>) -> Result<Array3<f32>> {
        let latents_bct = pred_latents
            .view()
            .permuted_axes([0, 2, 1])
            .to_owned()
            .as_standard_layout()
            .to_owned();
        let sess = self.sessions.get_mut("vae_decoder.onnx")?;
        let outputs = sess.run(::ort::inputs!["latents" => TensorRef::from_array_view(&latents_bct)?])?;
        let audio = outputs
            .get("audio")
            .context("missing audio output")?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .context("audio shape")?;
        Ok(audio)
    }
}

fn bool_to_f32(mask: &ArrayD<bool>) -> ArrayD<f32> {
    mask.mapv(|v| if v { 1.0 } else { 0.0 })
}

fn pad_3d(arr: &Array3<f32>, pad_len: usize) -> Array3<f32> {
    let (b, t, c) = arr.dim();
    let mut out = Array3::<f32>::zeros((b, t + pad_len, c));
    out.slice_mut(s![.., 0..t, ..]).assign(arr);
    out
}

fn pad_2d(arr: &Array2<f32>, pad_len: usize) -> Array2<f32> {
    let (b, t) = arr.dim();
    let mut out = Array2::<f32>::zeros((b, t + pad_len));
    out.slice_mut(s![.., 0..t]).assign(arr);
    out
}

fn stack_steps(steps: &[Array3<f32>], name: &str) -> Result<Array4<f32>> {
    if steps.is_empty() {
        return Err(anyhow::anyhow!("no steps recorded for {}", name));
    }
    let views: Vec<_> = steps.iter().map(|a| a.view()).collect();
    ndarray::stack(Axis(0), &views).context(format!("stack {name}"))
}
