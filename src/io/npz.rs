use anyhow::{Context, Result};
use ndarray::ArrayD;
use ndarray_npy::{NpzReader, NpzWriter};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct NpzData {
    f32_map: HashMap<String, ArrayD<f32>>,
}

impl NpzData {
    pub fn load(path: &Path, required_f32: &[&str], optional_f32: &[&str]) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open npz: {}", path.display()))?;
        let mut npz = NpzReader::new(f).with_context(|| format!("read npz: {}", path.display()))?;

        let mut f32_map = HashMap::new();

        for key in required_f32 {
            let name = npy_name(key);
            let arr = read_f32(&mut npz, &name).with_context(|| format!("read {} from {}", key, path.display()))?;
            f32_map.insert((*key).to_string(), arr);
        }

        for key in optional_f32 {
            let name = npy_name(key);
            if let Some(arr) = try_read_f32(&mut npz, &name) {
                f32_map.insert((*key).to_string(), arr);
            }
        }

        Ok(Self { f32_map })
    }

    pub fn get_f32(&self, key: &str) -> Option<&ArrayD<f32>> {
        self.f32_map.get(key)
    }

    pub fn require_f32(&self, key: &str) -> Result<&ArrayD<f32>> {
        self.get_f32(key).ok_or_else(|| anyhow::anyhow!("missing f32 key {}", key))
    }

}

pub fn write_npz_f32(path: &Path, entries: &[(&str, ArrayD<f32>)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create dir: {}", parent.display()))?;
    }
    let f = File::create(path).with_context(|| format!("create npz: {}", path.display()))?;
    let mut npz = NpzWriter::new(f);
    for (name, arr) in entries {
        npz.add_array(*name, arr)
            .with_context(|| format!("write {} to {}", name, path.display()))?;
    }
    npz.finish().with_context(|| format!("finalize npz: {}", path.display()))?;
    Ok(())
}

fn npy_name(key: &str) -> String {
    format!("{key}.npy")
}

fn read_f32<R: std::io::Read + std::io::Seek>(npz: &mut NpzReader<R>, name: &str) -> Result<ArrayD<f32>> {
    let arr: ArrayD<f32> = npz.by_name(name).with_context(|| format!("read {name}"))?;
    Ok(arr)
}

fn try_read_f32<R: std::io::Read + std::io::Seek>(npz: &mut NpzReader<R>, name: &str) -> Option<ArrayD<f32>> {
    npz.by_name(name).ok()
}
