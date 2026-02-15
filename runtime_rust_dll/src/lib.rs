use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::sync::Mutex;
use tokenizers::Tokenizer;

static LAST_ERROR: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContextConfig {
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default)]
    pub blocked_token_ids: Vec<usize>,
    #[serde(default)]
    pub forced_token_id: Option<usize>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
}

fn default_seed() -> i64 {
    42
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            blocked_token_ids: Vec::new(),
            forced_token_id: None,
            tokenizer_path: None,
        }
    }
}

pub struct AceContext {
    pub cfg: ContextConfig,
    pub tokenizer: Option<Tokenizer>,
}

#[derive(Debug, Deserialize)]
struct PrepareState {
    #[serde(default)]
    shift: Option<f32>,
    #[serde(default)]
    inference_steps: Option<usize>,
    #[serde(default)]
    current_step: Option<usize>,
}

fn set_last_error(msg: impl Into<String>) {
    if let Ok(mut g) = LAST_ERROR.lock() {
        *g = msg.into();
    }
}

fn ok() -> i32 {
    0
}

fn err(msg: impl Into<String>) -> i32 {
    set_last_error(msg);
    -1
}

fn cstr_to_string(ptr_in: *const c_char) -> Result<String, String> {
    if ptr_in.is_null() {
        return Ok(String::new());
    }
    let c = unsafe { CStr::from_ptr(ptr_in) };
    c.to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("invalid utf-8 c string: {e}"))
}

fn make_c_string(s: &str) -> Result<*mut c_char, String> {
    CString::new(s)
        .map(|c| c.into_raw())
        .map_err(|e| format!("failed to allocate c string: {e}"))
}

fn resolve_timesteps(shift: f32, steps: usize) -> Vec<f32> {
    if steps == 0 {
        return vec![0.0];
    }
    let mut ts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = 1.0f32 - (i as f32 / steps as f32);
        ts.push(t);
    }
    if (shift - 1.0).abs() < 1e-6 {
        return ts;
    }
    if (shift - 2.0).abs() < 1e-6 {
        return ts.into_iter().map(|t| t * t).collect();
    }
    if (shift - 3.0).abs() < 1e-6 {
        return ts.into_iter().map(|t| t * t * t).collect();
    }
    ts
}

fn parse_metadata_from_text(raw_text: &str) -> HashMap<String, Value> {
    let mut metadata = HashMap::<String, Value>::new();
    let think_re = Regex::new(r"(?s)<think>(.*?)</think>").expect("valid regex");
    let reasoning_text = think_re
        .captures(raw_text)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .unwrap_or_else(|| raw_text.to_string());

    let mut current_key = String::new();
    let mut current_lines: Vec<String> = Vec::new();
    let save_field = |key: &str, lines: &[String], dst: &mut HashMap<String, Value>| {
        if key.is_empty() || lines.is_empty() {
            return;
        }
        let joined = lines.join("\n");
        match key {
            "bpm" | "duration" => {
                if let Ok(v) = joined.trim().parse::<i64>() {
                    dst.insert(key.to_string(), json!(v));
                } else {
                    dst.insert(key.to_string(), json!(joined.trim()));
                }
            }
            "caption" | "genres" | "keyscale" | "language" | "timesignature" => {
                dst.insert(key.to_string(), json!(joined.trim()));
            }
            _ => {}
        }
    };

    for line in reasoning_text.lines() {
        if line.trim_start().starts_with('<') {
            continue;
        }
        let new_field = !line.starts_with(' ') && !line.starts_with('\t') && line.contains(':');
        if new_field {
            save_field(&current_key, &current_lines, &mut metadata);
            current_lines.clear();
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            current_key = parts[0].trim().to_lowercase();
            if parts.len() > 1 && !parts[1].trim().is_empty() {
                current_lines.push(parts[1].to_string());
            }
        } else if !current_key.is_empty() && (line.starts_with(' ') || line.starts_with('\t')) {
            current_lines.push(line.to_string());
        }
    }
    save_field(&current_key, &current_lines, &mut metadata);
    metadata
}

#[no_mangle]
pub extern "C" fn ace_create_context(config_json: *const c_char) -> *mut AceContext {
    let cfg_text = match cstr_to_string(config_json) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(e);
            return ptr::null_mut();
        }
    };
    let cfg = if cfg_text.trim().is_empty() {
        ContextConfig::default()
    } else {
        match serde_json::from_str::<ContextConfig>(&cfg_text) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("invalid config json: {e}"));
                return ptr::null_mut();
            }
        }
    };
    let tokenizer = cfg
        .tokenizer_path
        .as_ref()
        .and_then(|p| Tokenizer::from_file(p).ok());
    let ctx = AceContext { cfg, tokenizer };
    Box::into_raw(Box::new(ctx))
}

#[no_mangle]
pub extern "C" fn ace_free_context(ctx: *mut AceContext) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ctx));
    }
}

#[no_mangle]
pub extern "C" fn ace_string_free(ptr_in: *mut c_char) {
    if ptr_in.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(ptr_in));
    }
}

#[no_mangle]
pub extern "C" fn ace_last_error() -> *mut c_char {
    let msg = LAST_ERROR.lock().map(|g| g.clone()).unwrap_or_else(|_| "unknown".to_string());
    match make_c_string(&msg) {
        Ok(p) => p,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn ace_prepare_step_inputs(
    ctx: *mut AceContext,
    state_json: *const c_char,
    in_tensor_ptr: *const f32,
    in_tensor_len: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    if ctx.is_null() || out_json.is_null() {
        return err("null pointer in ace_prepare_step_inputs");
    }
    if in_tensor_ptr.is_null() && in_tensor_len > 0 {
        return err("null in_tensor_ptr with non-zero length");
    }

    let state_text = match cstr_to_string(state_json) {
        Ok(v) => v,
        Err(e) => return err(e),
    };
    let state = if state_text.trim().is_empty() {
        PrepareState {
            shift: Some(3.0),
            inference_steps: Some(8),
            current_step: Some(0),
        }
    } else {
        match serde_json::from_str::<PrepareState>(&state_text) {
            Ok(v) => v,
            Err(e) => return err(format!("invalid state json: {e}")),
        }
    };
    let shift = state.shift.unwrap_or(3.0);
    let steps = state.inference_steps.unwrap_or(8).max(1);
    let current = state.current_step.unwrap_or(0).min(steps.saturating_sub(1));
    let timesteps = resolve_timesteps(shift, steps);
    let t = timesteps[current];
    let t_next = timesteps.get(current + 1).copied().unwrap_or(0.0);

    let cfg = unsafe { &*ctx };
    let payload = json!({
        "seed": cfg.cfg.seed,
        "shift": shift,
        "inference_steps": steps,
        "current_step": current,
        "timestep": t,
        "next_timestep": t_next,
        "input_tensor_len": in_tensor_len,
    });
    let payload_s = match serde_json::to_string(&payload) {
        Ok(v) => v,
        Err(e) => return err(format!("serialize output json failed: {e}")),
    };
    let c = match make_c_string(&payload_s) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    unsafe {
        *out_json = c;
    }
    ok()
}

#[no_mangle]
pub extern "C" fn ace_scheduler_step(
    _ctx: *mut AceContext,
    xt_ptr: *const f32,
    vt_ptr: *const f32,
    len: usize,
    dt: f32,
    out_xt_ptr: *mut f32,
) -> i32 {
    if xt_ptr.is_null() || vt_ptr.is_null() || out_xt_ptr.is_null() {
        return err("null pointer in ace_scheduler_step");
    }
    if len == 0 {
        return err("len must be > 0");
    }
    let xt = unsafe { slice::from_raw_parts(xt_ptr, len) };
    let vt = unsafe { slice::from_raw_parts(vt_ptr, len) };
    let out = unsafe { slice::from_raw_parts_mut(out_xt_ptr, len) };
    for i in 0..len {
        out[i] = xt[i] - vt[i] * dt;
    }
    ok()
}

#[no_mangle]
pub extern "C" fn ace_apply_lm_constraints(
    ctx: *mut AceContext,
    logits_ptr: *const f32,
    vocab_size: usize,
    out_masked_logits_ptr: *mut f32,
) -> i32 {
    if ctx.is_null() || logits_ptr.is_null() || out_masked_logits_ptr.is_null() {
        return err("null pointer in ace_apply_lm_constraints");
    }
    if vocab_size == 0 {
        return err("vocab_size must be > 0");
    }
    let logits = unsafe { slice::from_raw_parts(logits_ptr, vocab_size) };
    let out = unsafe { slice::from_raw_parts_mut(out_masked_logits_ptr, vocab_size) };
    out.copy_from_slice(logits);

    let cfg = unsafe { &*ctx };
    for &tok in &cfg.cfg.blocked_token_ids {
        if tok < vocab_size {
            out[tok] = -1.0e30_f32;
        }
    }
    if let Some(forced) = cfg.cfg.forced_token_id {
        if forced < vocab_size {
            for (idx, v) in out.iter_mut().enumerate() {
                if idx != forced {
                    *v = -1.0e30_f32;
                }
            }
        }
    }
    ok()
}

#[no_mangle]
pub extern "C" fn ace_finalize_metadata(
    ctx: *mut AceContext,
    token_ids_ptr: *const i64,
    len: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    if ctx.is_null() || token_ids_ptr.is_null() || out_json.is_null() {
        return err("null pointer in ace_finalize_metadata");
    }
    let token_ids = unsafe { slice::from_raw_parts(token_ids_ptr, len) };
    let ctx_ref = unsafe { &mut *ctx };

    let raw_text = if let Some(tok) = ctx_ref.tokenizer.as_ref() {
        let ids_u32 = token_ids
            .iter()
            .filter_map(|&v| if v >= 0 { Some(v as u32) } else { None })
            .collect::<Vec<_>>();
        tok.decode(&ids_u32, false).unwrap_or_default()
    } else {
        token_ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
    };

    let metadata = parse_metadata_from_text(&raw_text);
    let payload = json!({
        "token_count": len,
        "raw_text": raw_text,
        "metadata": metadata,
    });
    let payload_s = match serde_json::to_string(&payload) {
        Ok(v) => v,
        Err(e) => return err(format!("serialize output json failed: {e}")),
    };
    let c = match make_c_string(&payload_s) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    unsafe {
        *out_json = c;
    }
    ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    #[test]
    fn resolve_timesteps_shift3_matches_expected() {
        let ts = resolve_timesteps(3.0, 8);
        assert_eq!(ts.len(), 9);
        assert!((ts[0] - 1.0).abs() < 1e-7);
        assert!((ts[1] - 0.669_921_9).abs() < 1e-6);
        assert!((ts[8] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn scheduler_step_computes_xt_minus_vt_dt() {
        let xt = [1.0_f32, 1.0, 1.0, 1.0];
        let vt = [0.1_f32, 0.2, 0.3, 0.4];
        let mut out = [0.0_f32; 4];
        let rc = ace_scheduler_step(
            ptr::null_mut(),
            xt.as_ptr(),
            vt.as_ptr(),
            xt.len(),
            0.5,
            out.as_mut_ptr(),
        );
        assert_eq!(rc, 0);
        assert!((out[0] - 0.95).abs() < 1e-7);
        assert!((out[1] - 0.9).abs() < 1e-7);
        assert!((out[2] - 0.85).abs() < 1e-7);
        assert!((out[3] - 0.8).abs() < 1e-7);
    }

    #[test]
    fn lm_constraints_apply_blocklist_and_forced_token() {
        let cfg = CString::new(r#"{"blocked_token_ids":[1,3],"forced_token_id":2}"#).unwrap();
        let ctx = ace_create_context(cfg.as_ptr());
        assert!(!ctx.is_null());

        let logits = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 5];
        let rc = ace_apply_lm_constraints(ctx, logits.as_ptr(), logits.len(), out.as_mut_ptr());
        assert_eq!(rc, 0);
        // forced token wins over all others.
        assert!((out[2] - 2.0).abs() < 1e-7);
        for (idx, v) in out.iter().enumerate() {
            if idx != 2 {
                assert!(*v < -1.0e29_f32);
            }
        }
        ace_free_context(ctx);
    }

    #[test]
    fn metadata_parser_extracts_known_fields() {
        let raw = r#"
caption: A bright synth-pop song
genres: synthpop, dance
bpm: 128
duration: 30
timesignature: 4/4
"#;
        let meta = parse_metadata_from_text(raw);
        assert_eq!(meta.get("caption"), Some(&json!("A bright synth-pop song")));
        assert_eq!(meta.get("genres"), Some(&json!("synthpop, dance")));
        assert_eq!(meta.get("bpm"), Some(&json!(128)));
        assert_eq!(meta.get("duration"), Some(&json!(30)));
        assert_eq!(meta.get("timesignature"), Some(&json!("4/4")));
    }
}
