use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaseSpec {
    pub case_id: String,
    pub mode: String,
    pub seed: i64,
    pub inference_steps: i64,
    pub shift: f32,
    pub infer_method: String,
    pub thinking: bool,
    pub audio_format: String,
    pub caption: String,
    pub lyrics: String,
    pub simple_mode_query: String,
    pub lm_model_variant: String,
    pub deterministic: bool,
    pub metadata: HashMap<String, serde_json::Value>,
    pub expected: HashMap<String, serde_json::Value>,
}

impl Default for CaseSpec {
    fn default() -> Self {
        Self {
            case_id: String::new(),
            mode: "text2music".to_string(),
            seed: 42,
            inference_steps: 8,
            shift: 3.0,
            infer_method: "ode".to_string(),
            thinking: false,
            audio_format: "wav".to_string(),
            caption: String::new(),
            lyrics: String::new(),
            simple_mode_query: String::new(),
            lm_model_variant: "1.7B".to_string(),
            deterministic: true,
            metadata: HashMap::new(),
            expected: HashMap::new(),
        }
    }
}

impl CaseSpec {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read case: {}", path.display()))?;
        let mut spec: CaseSpec =
            serde_json::from_str(&text).with_context(|| format!("parse case: {}", path.display()))?;
        if spec.case_id.is_empty() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                spec.case_id = stem.to_string();
            }
        }
        if spec.mode.is_empty() {
            spec.mode = "text2music".to_string();
        }
        Ok(spec)
    }
}
