mod audio;
mod case;
mod io;
mod ort;
mod parity;
mod pipeline;
mod scheduler;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;
use std::path::PathBuf;

use crate::audio::{decode_audio_from_npz, write_wav, AudioReport};
use crate::case::CaseSpec;
use crate::parity::{compare_npz, write_parity_report, ParityReport};
use crate::pipeline::core::{CorePipeline, GenerateOptions};
use crate::pipeline::lm::{LmGenerateOptions, LmPipeline};

#[derive(Parser)]
#[command(name = "acestep_ort")]
#[command(about = "ACE-Step Rust ORT runtime (core + condition)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(long)]
        case: PathBuf,
        #[arg(long)]
        inputs_npz: PathBuf,
        #[arg(long)]
        out_npz: PathBuf,
        #[arg(long)]
        out_wav: PathBuf,
        #[arg(long, default_value = "artifacts/onnx_runtime")]
        onnx_dir: PathBuf,
        #[arg(long, default_value_t = false)]
        online_qwen_embed: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        prefer_precomputed_condition: bool,
        #[arg(long, default_value = "checkpoints/Qwen3-Embedding-0.6B/tokenizer.json")]
        qwen_tokenizer: PathBuf,
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        lyrics: Option<String>,
        #[arg(long, default_value_t = 256)]
        text_max_tokens: usize,
        #[arg(long, default_value_t = 2048)]
        lyric_max_tokens: usize,
        #[arg(long, default_value_t = false)]
        online_lm_simple_mode: bool,
        #[arg(long, default_value = "auto")]
        lm_model: String,
        #[arg(long)]
        lm_tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 768)]
        lm_max_new_tokens: usize,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        lm_constrained: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        deterministic: bool,
    },
    Parity {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        out_json: PathBuf,
        #[arg(long, default_value_t = 1e-1)]
        max_abs: f64,
        #[arg(long, default_value_t = 5e-4)]
        rmse: f64,
    },
    AudioParity {
        #[arg(long)]
        case: PathBuf,
        #[arg(long)]
        candidate_npz: PathBuf,
        #[arg(long)]
        out_wav: PathBuf,
        #[arg(long)]
        out_json: PathBuf,
        #[arg(long, default_value = "artifacts/onnx_runtime")]
        onnx_dir: PathBuf,
        #[arg(long, default_value_t = 48000)]
        sample_rate: u32,
        #[arg(long, default_value_t = 35.0)]
        min_snr: f64,
        #[arg(long, default_value_t = 0.98)]
        min_corr: f64,
    },
    LmSample {
        #[arg(long)]
        case: PathBuf,
        #[arg(long, default_value = "artifacts/onnx_runtime")]
        onnx_dir: PathBuf,
        #[arg(long, default_value = "auto")]
        lm_model: String,
        #[arg(long)]
        lm_tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 768)]
        lm_max_new_tokens: usize,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        lm_constrained: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        deterministic: bool,
        #[arg(long)]
        out_json: PathBuf,
    },
}

fn cmd_generate(
    case_path: PathBuf,
    inputs_npz: PathBuf,
    out_npz: PathBuf,
    out_wav: PathBuf,
    onnx_dir: PathBuf,
    mut opts: GenerateOptions,
    lm_opts: Option<(String, Option<PathBuf>, LmGenerateOptions)>,
) -> Result<()> {
    let mut case = CaseSpec::from_path(&case_path).context("load case spec")?;

    if let Some((lm_model, lm_tokenizer, lm_generate_opts)) = lm_opts {
        if case.mode == "simple_mode" {
            let instrumental = case
                .metadata
                .get("instrumental")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let vocal_language = case
                .metadata
                .get("vocal_language")
                .or_else(|| case.metadata.get("language"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();

            let mut lm = LmPipeline::new(onnx_dir.clone(), &lm_model, lm_tokenizer)?;
            let sample = lm.generate_sample_from_query(
                case.simple_mode_query.as_str(),
                instrumental,
                vocal_language.as_str(),
                &lm_generate_opts,
            )?;

            case.caption = sample.caption.clone();
            case.lyrics = sample.lyrics.clone();
            for (k, v) in sample.metadata {
                case.metadata.insert(k, v);
            }
            if let Some(language) = case
                .metadata
                .get("language")
                .and_then(Value::as_str)
                .map(|v| v.to_string())
            {
                case.metadata
                    .entry("vocal_language".to_string())
                    .or_insert(Value::String(language));
            }

            opts.online_qwen_embed = true;
            if opts.text_override.is_none() {
                opts.text_override = Some(case.caption.clone());
            }
            if opts.lyrics_override.is_none() {
                opts.lyrics_override = Some(case.lyrics.clone());
            }

            let out_dir = out_npz
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            let lm_sample_path = out_dir.join(format!("{}_rust_lm_sample.json", case.case_id));
            let lm_sample_json = serde_json::to_string_pretty(&serde_json::json!({
                "caption": case.caption.clone(),
                "lyrics": case.lyrics.clone(),
                "metadata": case.metadata.clone(),
                "token_ids": sample.token_ids,
                "raw_text": sample.raw_text,
                "lm_model": lm_model,
            }))?;
            std::fs::write(&lm_sample_path, lm_sample_json)
                .with_context(|| format!("write rust lm sample: {}", lm_sample_path.display()))?;
            println!("Saved LM sample: {}", lm_sample_path.display());
        }
    }

    let mut pipeline = CorePipeline::new(onnx_dir)?;
    pipeline.generate(&case, &inputs_npz, &out_npz, Some(&out_wav), &opts)?;
    println!("Saved tensor: {}", out_npz.display());
    println!("Saved audio : {}", out_wav.display());
    Ok(())
}

fn cmd_parity(baseline: PathBuf, candidate: PathBuf, out_json: PathBuf, max_abs: f64, rmse: f64) -> Result<()> {
    let report: ParityReport = compare_npz(&baseline, &candidate, max_abs, rmse)?;
    write_parity_report(&out_json, &report)?;
    println!("Wrote report: {}", out_json.display());
    if !report.verdict.pass {
        for f in &report.verdict.failures {
            println!("  - {}", f);
        }
    }
    Ok(())
}

fn cmd_audio_parity(
    case_path: PathBuf,
    candidate_npz: PathBuf,
    out_wav: PathBuf,
    out_json: PathBuf,
    onnx_dir: PathBuf,
    sample_rate: u32,
    min_snr: f64,
    min_corr: f64,
) -> Result<()> {
    let case = CaseSpec::from_path(&case_path).context("load case spec")?;
    let audio = decode_audio_from_npz(&onnx_dir, &candidate_npz)?;
    let audio_0 = audio.index_axis(ndarray::Axis(0), 0).to_owned();
    write_wav(&out_wav, &audio_0, sample_rate)?;
    let report: AudioReport = AudioReport::from_case(&case, &out_wav, min_snr, min_corr)?;
    report.write_json(&out_json)?;
    println!("Wrote report: {}", out_json.display());
    Ok(())
}

fn cmd_lm_sample(
    case_path: PathBuf,
    onnx_dir: PathBuf,
    lm_model: String,
    lm_tokenizer: Option<PathBuf>,
    lm_generate_opts: LmGenerateOptions,
    out_json: PathBuf,
) -> Result<()> {
    let case = CaseSpec::from_path(&case_path).context("load case spec")?;
    let resolved_lm_model = if lm_model == "auto" {
        case.lm_model_variant.clone()
    } else {
        lm_model
    };

    let instrumental = case
        .metadata
        .get("instrumental")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let vocal_language = case
        .metadata
        .get("vocal_language")
        .or_else(|| case.metadata.get("language"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let mut lm = LmPipeline::new(onnx_dir, &resolved_lm_model, lm_tokenizer)?;
    let sample = lm.generate_sample_from_query(
        case.simple_mode_query.as_str(),
        instrumental,
        vocal_language.as_str(),
        &lm_generate_opts,
    )?;

    let out_obj = serde_json::json!({
        "case_id": case.case_id,
        "lm_model": resolved_lm_model,
        "caption": sample.caption,
        "lyrics": sample.lyrics,
        "metadata": sample.metadata,
        "token_ids": sample.token_ids,
        "raw_text": sample.raw_text,
    });
    std::fs::write(&out_json, serde_json::to_string_pretty(&out_obj)?)
        .with_context(|| format!("write lm sample json: {}", out_json.display()))?;
    println!("Wrote LM sample: {}", out_json.display());
    Ok(())
}

fn main() -> Result<()> {
    let _ = ::ort::init().commit();

    let cli = Cli::parse();
    match cli.command {
        Commands::Generate {
            case,
            inputs_npz,
            out_npz,
            out_wav,
            onnx_dir,
            online_qwen_embed,
            prefer_precomputed_condition,
            qwen_tokenizer,
            text,
            lyrics,
            text_max_tokens,
            lyric_max_tokens,
            online_lm_simple_mode,
            lm_model,
            lm_tokenizer,
            lm_max_new_tokens,
            lm_constrained,
            deterministic,
        } => {
            let resolved_lm_model = if lm_model == "auto" {
                CaseSpec::from_path(&case)
                    .map(|c| c.lm_model_variant)
                    .unwrap_or_else(|_| "1.7B".to_string())
            } else {
                lm_model
            };
            cmd_generate(
                case,
                inputs_npz,
                out_npz,
                out_wav,
                onnx_dir,
                GenerateOptions {
                    online_qwen_embed,
                    prefer_precomputed_condition,
                    qwen_tokenizer,
                    text_override: text,
                    lyrics_override: lyrics,
                    text_max_tokens,
                    lyric_max_tokens,
                },
                if online_lm_simple_mode {
                    Some((
                        resolved_lm_model,
                        lm_tokenizer,
                        LmGenerateOptions {
                            max_new_tokens: lm_max_new_tokens,
                            constrained: lm_constrained,
                            deterministic,
                        },
                    ))
                } else {
                    None
                },
            )
        }
        Commands::Parity {
            baseline,
            candidate,
            out_json,
            max_abs,
            rmse,
        } => cmd_parity(baseline, candidate, out_json, max_abs, rmse),
        Commands::AudioParity {
            case,
            candidate_npz,
            out_wav,
            out_json,
            onnx_dir,
            sample_rate,
            min_snr,
            min_corr,
        } => cmd_audio_parity(case, candidate_npz, out_wav, out_json, onnx_dir, sample_rate, min_snr, min_corr),
        Commands::LmSample {
            case,
            onnx_dir,
            lm_model,
            lm_tokenizer,
            lm_max_new_tokens,
            lm_constrained,
            deterministic,
            out_json,
        } => cmd_lm_sample(
            case,
            onnx_dir,
            lm_model,
            lm_tokenizer,
            LmGenerateOptions {
                max_new_tokens: lm_max_new_tokens,
                constrained: lm_constrained,
                deterministic,
            },
            out_json,
        ),
    }
}
