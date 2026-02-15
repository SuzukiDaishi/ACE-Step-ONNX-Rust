use anyhow::{Context, Result};
use ::ort::session::Session;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct OrtSessionManager {
    onnx_dir: PathBuf,
    sessions: HashMap<String, Session>,
}

impl OrtSessionManager {
    pub fn new(onnx_dir: PathBuf) -> Self {
        Self {
            onnx_dir,
            sessions: HashMap::new(),
        }
    }

    pub fn get_mut(&mut self, model_name: &str) -> Result<&mut Session> {
        if !self.sessions.contains_key(model_name) {
            let model_path = self.model_path(model_name);
            let session = Session::builder()
                .with_context(|| "create session builder")?
                .commit_from_file(model_path.as_path())
                .with_context(|| format!("load onnx: {}", model_path.display()))?;
            self.sessions.insert(model_name.to_string(), session);
        }
        Ok(self.sessions.get_mut(model_name).expect("session inserted"))
    }

    pub fn exists(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists()
    }

    fn model_path(&self, model_name: &str) -> PathBuf {
        let path = Path::new(model_name);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.onnx_dir.join(model_name)
        }
    }
}
