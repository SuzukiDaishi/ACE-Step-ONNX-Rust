use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct CaseSpec {
    pub case_id: String,
    pub mode: String,
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default = "default_steps")]
    pub inference_steps: usize,
    #[serde(default = "default_shift")]
    pub shift: f32,
    #[serde(default = "default_method")]
    pub infer_method: String,
}

fn default_seed() -> i64 {
    42
}
fn default_steps() -> usize {
    8
}
fn default_shift() -> f32 {
    3.0
}
fn default_method() -> String {
    "ode".to_string()
}

impl CaseSpec {
    pub fn from_path(path: &Path) -> Result<Self> {
        let txt = fs::read_to_string(path).with_context(|| format!("read case file: {}", path.display()))?;
        let c: CaseSpec = serde_json::from_str(&txt).with_context(|| format!("parse case file: {}", path.display()))?;
        Ok(c)
    }
}
