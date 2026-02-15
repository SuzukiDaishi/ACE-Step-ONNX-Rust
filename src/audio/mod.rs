use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use ndarray::{Array2, Array3, ArrayD, Ix3};
use ndarray_npy::NpzReader;
use ::ort::value::TensorRef;
use serde::Serialize;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::audio::metrics::{audio_metrics, AudioMetrics};
use crate::case::CaseSpec;
use crate::ort::session::OrtSessionManager;

pub mod metrics;

#[derive(Debug, Serialize)]
pub struct AudioReport {
    pub baseline_path: String,
    pub candidate_path: String,
    pub sample_rate: u32,
    pub baseline_shape: Vec<usize>,
    pub candidate_shape: Vec<usize>,
    pub metrics: AudioMetrics,
    pub thresholds: AudioThresholds,
    pub verdict: AudioVerdict,
}

#[derive(Debug, Serialize)]
pub struct AudioThresholds {
    pub min_snr: f64,
    pub min_corr: f64,
}

#[derive(Debug, Serialize)]
pub struct AudioVerdict {
    pub pass: bool,
    pub failures: Vec<String>,
}

impl AudioReport {
    pub fn from_case(case: &CaseSpec, candidate_wav: &Path, min_snr: f64, min_corr: f64) -> Result<Self> {
        let baseline = resolve_baseline_audio(&case.case_id)?;
        let (baseline_audio, baseline_sr) = read_wav(&baseline)?;
        let (candidate_audio, candidate_sr) = read_wav(candidate_wav)?;

        if baseline_sr != candidate_sr {
            return Err(anyhow::anyhow!(
                "sample rate mismatch: baseline {} vs candidate {}",
                baseline_sr,
                candidate_sr
            ));
        }

        let metrics = audio_metrics(&baseline_audio, &candidate_audio);
        let mut failures = Vec::new();
        if metrics.snr_db < min_snr {
            failures.push(format!("snr {:.3} < {:.3}", metrics.snr_db, min_snr));
        }
        if metrics.corr < min_corr {
            failures.push(format!("corr {:.6} < {:.6}", metrics.corr, min_corr));
        }
        let verdict = AudioVerdict {
            pass: failures.is_empty(),
            failures,
        };
        Ok(Self {
            baseline_path: baseline.display().to_string(),
            candidate_path: candidate_wav.display().to_string(),
            sample_rate: baseline_sr,
            baseline_shape: baseline_audio.shape().iter().copied().collect(),
            candidate_shape: candidate_audio.shape().iter().copied().collect(),
            metrics,
            thresholds: AudioThresholds { min_snr, min_corr },
            verdict,
        })
    }

    pub fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("create report dir: {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self).context("serialize audio report")?;
        std::fs::write(path, text).with_context(|| format!("write report: {}", path.display()))?;
        Ok(())
    }
}

pub fn decode_audio_from_npz(onnx_dir: &Path, candidate_npz: &Path) -> Result<Array3<f32>> {
    let f = File::open(candidate_npz).with_context(|| format!("open npz: {}", candidate_npz.display()))?;
    let mut npz = NpzReader::new(f).with_context(|| format!("read npz: {}", candidate_npz.display()))?;
    let latents: ArrayD<f32> = npz
        .by_name("pred_latents.npy")
        .with_context(|| format!("read pred_latents from {}", candidate_npz.display()))?;
    let latents = latents
        .into_dimensionality::<Ix3>()
        .context("pred_latents shape")?;
    let latents_bct = latents
        .permuted_axes([0, 2, 1])
        .to_owned()
        .as_standard_layout()
        .to_owned();

    let mut sessions = OrtSessionManager::new(onnx_dir.to_path_buf());
    let sess = sessions.get_mut("vae_decoder.onnx")?;
    let latents_tensor = TensorRef::from_array_view(&latents_bct)?;
    let outputs = sess.run(::ort::inputs!["latents" => latents_tensor])?;
    let audio = outputs
        .get("audio")
        .context("missing output audio")?
        .try_extract_array::<f32>()?
        .to_owned();
    let audio = audio.into_dimensionality::<Ix3>().context("audio shape")?;
    Ok(audio)
}

pub fn write_wav(path: &Path, audio: &Array2<f32>, sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create wav dir: {}", parent.display()))?;
    }
    let channels = audio.shape()[0] as u16;
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).with_context(|| format!("create wav: {}", path.display()))?;
    let frames = audio.shape()[1];
    for i in 0..frames {
        for c in 0..channels as usize {
            let v = audio[(c, i)].clamp(-1.0, 1.0);
            writer.write_sample(v).context("write wav sample")?;
        }
    }
    writer.finalize().context("finalize wav")?;
    Ok(())
}

fn resolve_baseline_audio(case_id: &str) -> Result<PathBuf> {
    let summary_path = Path::new("fixtures/tensors").join(format!("{case_id}.json"));
    if summary_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&summary_path) {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(path) = data.get("audio_path").and_then(|v| v.as_str()) {
                    let candidate = PathBuf::from(path);
                    if candidate.is_absolute() && candidate.exists() {
                        return Ok(candidate);
                    }
                    let rel = Path::new(path);
                    if rel.exists() {
                        return Ok(rel.to_path_buf());
                    }
                }
            }
        }
    }
    let fallback = Path::new("fixtures/audio").join(format!("{case_id}.wav"));
    if fallback.exists() {
        return Ok(fallback);
    }
    Err(anyhow::anyhow!("baseline audio not found for case_id={case_id}"))
}

fn read_wav(path: &Path) -> Result<(Array2<f32>, u32)> {
    let mut reader = WavReader::open(path).with_context(|| format!("open wav: {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut data: Vec<f32> = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                data.push(sample.context("read wav sample")?);
            }
        }
        SampleFormat::Int => {
            let max = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            for sample in reader.samples::<i32>() {
                let s = sample.context("read wav sample")? as f32 / max;
                data.push(s);
            }
        }
    }
    let frames = data.len() / channels.max(1);
    let mut arr = Array2::<f32>::zeros((channels.max(1), frames));
    for i in 0..frames {
        for c in 0..channels {
            arr[(c, i)] = data[i * channels + c];
        }
    }
    Ok((arr, spec.sample_rate))
}
