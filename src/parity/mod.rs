use anyhow::{Context, Result};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::parity::metrics::{tensor_metrics, TensorMetrics};

pub mod metrics;

#[derive(Debug, Serialize)]
pub struct ParityReport {
    pub baseline: String,
    pub candidate: String,
    pub per_tensor: HashMap<String, TensorMetrics>,
    pub thresholds: Thresholds,
    pub verdict: Verdict,
}

#[derive(Debug, Serialize)]
pub struct Thresholds {
    pub max_abs: f64,
    pub rmse: f64,
}

#[derive(Debug, Serialize)]
pub struct Verdict {
    pub pass: bool,
    pub failures: Vec<String>,
}

pub fn compare_npz(baseline: &Path, candidate: &Path, max_abs: f64, rmse: f64) -> Result<ParityReport> {
    let keys = ["pred_latents", "xt_steps", "vt_steps"];
    let baseline_map = read_npz_f32(baseline, &keys)?;
    let candidate_map = read_npz_f32(candidate, &keys)?;

    let mut per_tensor = HashMap::new();
    let mut failures = Vec::new();

    for key in keys.iter() {
        let a = baseline_map.get(*key).context(format!("missing {} in baseline", key))?;
        let b = candidate_map.get(*key).context(format!("missing {} in candidate", key))?;
        let metrics = tensor_metrics(a, b);
        if metrics.max_abs > max_abs {
            failures.push(format!("tensor {key}: max_abs {:.3e} > {:.3e}", metrics.max_abs, max_abs));
        }
        if metrics.rmse > rmse {
            failures.push(format!("tensor {key}: rmse {:.3e} > {:.3e}", metrics.rmse, rmse));
        }
        per_tensor.insert((*key).to_string(), metrics);
    }

    let verdict = Verdict {
        pass: failures.is_empty(),
        failures,
    };

    Ok(ParityReport {
        baseline: baseline.display().to_string(),
        candidate: candidate.display().to_string(),
        per_tensor,
        thresholds: Thresholds { max_abs, rmse },
        verdict,
    })
}

pub fn write_parity_report(path: &Path, report: &ParityReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create report dir: {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(report).context("serialize parity report")?;
    std::fs::write(path, text).with_context(|| format!("write report: {}", path.display()))?;
    Ok(())
}

fn read_npz_f32(path: &Path, keys: &[&str]) -> Result<HashMap<String, ArrayD<f32>>> {
    let f = File::open(path).with_context(|| format!("open npz: {}", path.display()))?;
    let mut npz = NpzReader::new(f).with_context(|| format!("read npz: {}", path.display()))?;
    let mut map = HashMap::new();
    for key in keys {
        let name = format!("{key}.npy");
        let arr: ArrayD<f32> = npz.by_name(&name).with_context(|| format!("read {} from {}", key, path.display()))?;
        map.insert((*key).to_string(), arr);
    }
    Ok(map)
}
