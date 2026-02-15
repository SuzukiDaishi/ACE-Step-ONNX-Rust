mod case;
mod metrics;
mod scheduler;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "acestep_ort")]
#[command(about = "ACE-Step Rust ORT runtime scaffold", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(long)]
        case: PathBuf,
    },
    Parity {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        out_json: PathBuf,
    },
}

fn read_npz_tensor(path: &PathBuf, key: &str) -> Result<ArrayD<f32>> {
    let f = File::open(path).with_context(|| format!("open npz: {}", path.display()))?;
    let mut npz = NpzReader::new(f).with_context(|| format!("read npz: {}", path.display()))?;
    let arr: ArrayD<f32> = npz
        .by_name(key)
        .with_context(|| format!("read key {} from {}", key, path.display()))?;
    Ok(arr)
}

fn cmd_generate(case_path: PathBuf) -> Result<()> {
    let case = case::CaseSpec::from_path(&case_path)?;
    println!("Generate scaffold invoked for case: {}", case.case_id);
    println!(
        "Mode: {} | seed: {} | steps: {} | shift: {} | method: {}",
        case.mode, case.seed, case.inference_steps, case.shift, case.infer_method
    );
    println!("This scaffold reserves ORT wiring for the next implementation step.");
    Ok(())
}

fn cmd_parity(baseline: PathBuf, candidate: PathBuf, out_json: PathBuf) -> Result<()> {
    let a = read_npz_tensor(&baseline, "pred_latents")?;
    let b = read_npz_tensor(&candidate, "pred_latents")?;

    if a.shape() != b.shape() {
        bail!(
            "Shape mismatch for pred_latents: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }

    let m = metrics::tensor_metrics(&a, &b);
    let report = json!({
        "tensor": "pred_latents",
        "shape": a.shape(),
        "max_abs": m.max_abs,
        "rmse": m.rmse,
        "cos_sim": m.cos_sim
    });

    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create report dir: {}", parent.display()))?;
    }

    let mut f = File::create(&out_json)
        .with_context(|| format!("create report file: {}", out_json.display()))?;
    f.write_all(serde_json::to_string_pretty(&report)?.as_bytes())
        .with_context(|| format!("write report file: {}", out_json.display()))?;

    println!("Wrote report: {}", out_json.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate { case } => cmd_generate(case),
        Commands::Parity {
            baseline,
            candidate,
            out_json,
        } => cmd_parity(baseline, candidate, out_json),
    }
}
