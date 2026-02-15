use anyhow::{Context, Result};
use ndarray::{s, Array2, ArrayD, Ix3};
use ::ort::value::TensorRef;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::ort::session::OrtSessionManager;

const DEFAULT_LM_INSPIRED_INSTRUCTION: &str =
    "Expand the user's input into a more detailed and specific musical description:";

#[derive(Debug, Clone, Deserialize)]
struct LmIoContract {
    num_layers: usize,
    prefill_path: String,
    decode_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmSimpleSample {
    pub token_ids: Vec<i64>,
    pub raw_text: String,
    pub metadata: HashMap<String, Value>,
    pub caption: String,
    pub lyrics: String,
    pub instrumental: bool,
}

#[derive(Debug, Clone)]
pub struct LmGenerateOptions {
    pub max_new_tokens: usize,
    pub constrained: bool,
    pub deterministic: bool,
}

impl Default for LmGenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 768,
            constrained: true,
            deterministic: true,
        }
    }
}

pub struct LmPipeline {
    sessions: OrtSessionManager,
    contract: LmIoContract,
    tokenizer: Tokenizer,
    audio_code_token_ids: Vec<usize>,
    eos_token_id: Option<usize>,
    pad_token_id: Option<usize>,
    newline_token_id: Option<usize>,
    im_end_token_id: Option<usize>,
    backtick_token_id: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstrainedState {
    ThinkTag,
    NewlineAfterThink,
    BpmName,
    BpmValue,
    CaptionName,
    CaptionValue,
    DurationName,
    DurationValue,
    GenresName,
    GenresValue,
    KeyscaleName,
    KeyscaleValue,
    LanguageName,
    LanguageValue,
    TimesigName,
    TimesigValue,
    ThinkEndTag,
    Completed,
}

struct ConstraintTracker {
    tokenizer: Tokenizer,
    state: ConstrainedState,
    position_in_state: usize,
    accumulated_token_ids: Vec<usize>,
    accumulated_text: String,
    caption_after_newline: bool,
    caption_token_count: usize,
    caption_ending: bool,
    pending_field_name: String,
    genres_trie: GenresTrieNode,
    has_genres_vocab: bool,
    char_to_tokens: HashMap<char, Vec<usize>>,
    token_to_text: HashMap<usize, String>,
    bpm_prefix_tree: HashMap<Vec<usize>, Vec<usize>>,
    duration_prefix_tree: HashMap<Vec<usize>, Vec<usize>>,
    timesig_prefix_tree: HashMap<Vec<usize>, Vec<usize>>,
    keyscale_prefix_tree: HashMap<Vec<usize>, Vec<usize>>,
    language_prefix_tree: HashMap<Vec<usize>, Vec<usize>>,
    language_forced_tokens: Vec<usize>,
    backtick_token_id: Option<usize>,
    newline_token_id: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct GenresTrieNode {
    end: bool,
    children: HashMap<char, GenresTrieNode>,
}

impl ConstraintTracker {
    fn new(
        tokenizer: &Tokenizer,
        vocal_language: &str,
        newline_token_id: Option<usize>,
        backtick_token_id: Option<usize>,
    ) -> Result<Self> {
        let (genres_trie, has_genres_vocab) = build_genres_trie_from_file("acestep/genres_vocab.txt");
        let (char_to_tokens, token_to_text) = precompute_char_token_mapping(tokenizer);

        let bpm_values: Vec<String> = (30..=300).map(|v| v.to_string()).collect();
        let duration_values: Vec<String> = (10..=600).map(|v| v.to_string()).collect();
        let timesig_values = vec!["2".to_string(), "3".to_string(), "4".to_string(), "6".to_string()];
        let keyscale_values = build_valid_keyscales();
        let language_values = vec![
            "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr",
            "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne",
            "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te",
            "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let bpm_prefix_tree =
            build_prefixed_value_tree(tokenizer, &bpm_values, "bpm:", "bpm: ", newline_token_id)?;
        let duration_prefix_tree =
            build_prefixed_value_tree(tokenizer, &duration_values, "duration:", "duration: ", newline_token_id)?;
        let timesig_prefix_tree = build_prefixed_value_tree(
            tokenizer,
            &timesig_values,
            "timesignature:",
            "timesignature: ",
            newline_token_id,
        )?;
        let keyscale_prefix_tree =
            build_prefixed_value_tree(tokenizer, &keyscale_values, "keyscale:", "keyscale: ", newline_token_id)?;
        let language_prefix_tree =
            build_prefixed_value_tree(tokenizer, &language_values, "language:", "language: ", newline_token_id)?;

        let language_forced_tokens = {
            let lang = vocal_language.trim().to_lowercase();
            if lang.is_empty() || lang == "unknown" {
                Vec::new()
            } else {
                encode_ids(tokenizer, &format!(" {lang}"))
                    .with_context(|| format!("encode forced language token: {lang}"))?
            }
        };

        Ok(Self {
            tokenizer: tokenizer.clone(),
            state: ConstrainedState::ThinkTag,
            position_in_state: 0,
            accumulated_token_ids: Vec::new(),
            accumulated_text: String::new(),
            caption_after_newline: false,
            caption_token_count: 0,
            caption_ending: false,
            pending_field_name: String::new(),
            genres_trie,
            has_genres_vocab,
            char_to_tokens,
            token_to_text,
            bpm_prefix_tree,
            duration_prefix_tree,
            timesig_prefix_tree,
            keyscale_prefix_tree,
            language_prefix_tree,
            language_forced_tokens,
            backtick_token_id,
            newline_token_id,
        })
    }

    fn in_metadata_phase(&self) -> bool {
        self.state != ConstrainedState::Completed
    }

    fn fixed_string_for_state(&self) -> Option<&'static str> {
        match self.state {
            ConstrainedState::ThinkTag => Some("<think>"),
            ConstrainedState::NewlineAfterThink => Some("\n"),
            ConstrainedState::BpmName => Some("bpm:"),
            ConstrainedState::CaptionName => Some("caption:"),
            ConstrainedState::DurationName => Some("duration:"),
            ConstrainedState::GenresName => Some("genres:"),
            ConstrainedState::KeyscaleName => Some("keyscale:"),
            ConstrainedState::LanguageName => Some("language:"),
            ConstrainedState::TimesigName => Some("timesignature:"),
            ConstrainedState::ThinkEndTag => Some("</think>"),
            _ => None,
        }
    }

    fn transition_to_next_state(&mut self) {
        self.state = match self.state {
            ConstrainedState::ThinkTag => ConstrainedState::NewlineAfterThink,
            ConstrainedState::NewlineAfterThink => ConstrainedState::BpmName,
            ConstrainedState::BpmName => ConstrainedState::BpmValue,
            ConstrainedState::BpmValue => ConstrainedState::CaptionName,
            ConstrainedState::CaptionName => ConstrainedState::CaptionValue,
            ConstrainedState::CaptionValue => ConstrainedState::DurationName,
            ConstrainedState::DurationName => ConstrainedState::DurationValue,
            ConstrainedState::DurationValue => ConstrainedState::GenresName,
            ConstrainedState::GenresName => ConstrainedState::GenresValue,
            ConstrainedState::GenresValue => ConstrainedState::KeyscaleName,
            ConstrainedState::KeyscaleName => ConstrainedState::KeyscaleValue,
            ConstrainedState::KeyscaleValue => ConstrainedState::LanguageName,
            ConstrainedState::LanguageName => ConstrainedState::LanguageValue,
            ConstrainedState::LanguageValue => ConstrainedState::TimesigName,
            ConstrainedState::TimesigName => ConstrainedState::TimesigValue,
            ConstrainedState::TimesigValue => ConstrainedState::ThinkEndTag,
            ConstrainedState::ThinkEndTag => ConstrainedState::Completed,
            ConstrainedState::Completed => ConstrainedState::Completed,
        };
        self.position_in_state = 0;
        self.accumulated_token_ids.clear();
        self.accumulated_text.clear();
        self.caption_after_newline = false;
        self.caption_token_count = 0;
        self.caption_ending = false;
        self.pending_field_name.clear();
    }

    fn state_from_field_name(name: &str) -> Option<ConstrainedState> {
        match name {
            "duration" => Some(ConstrainedState::DurationValue),
            "genres" => Some(ConstrainedState::GenresValue),
            "keyscale" => Some(ConstrainedState::KeyscaleValue),
            "language" => Some(ConstrainedState::LanguageValue),
            "timesignature" => Some(ConstrainedState::TimesigValue),
            _ => None,
        }
    }

    fn decode_token(&self, token_id: usize) -> String {
        self.tokenizer
            .decode(&[token_id as u32], false)
            .unwrap_or_default()
    }

    fn allowed_tokens_for_fixed_string(&self, fixed_str: &str) -> Vec<usize> {
        if self.position_in_state >= fixed_str.len() {
            return Vec::new();
        }
        let remaining = &fixed_str[self.position_in_state..];
        if remaining.is_empty() {
            return Vec::new();
        }

        for end in (1..=remaining.len()).rev() {
            let prefix = &remaining[..end];
            if let Ok(enc) = self.tokenizer.encode(prefix, false) {
                let ids = enc.get_ids();
                if ids.len() == 1 {
                    return vec![ids[0] as usize];
                }
            }
        }

        let mut allowed_tokens: HashMap<usize, usize> = HashMap::new();
        let search_limit = remaining.len().min(20);
        for end in 1..=search_limit {
            let prefix = &remaining[..end];
            if let Ok(enc) = self.tokenizer.encode(prefix, false) {
                let ids = enc.get_ids();
                if let Some(first) = ids.first() {
                    let first_id = *first as usize;
                    let decoded = self.decode_token(first_id);
                    let normalized_prefix = prefix.trim_start().to_lowercase();
                    let normalized_decoded = decoded.trim_start().to_lowercase();
                    if normalized_decoded.starts_with(&normalized_prefix)
                        || normalized_prefix.starts_with(&normalized_decoded)
                    {
                        let entry = allowed_tokens.entry(first_id).or_insert(0);
                        if end > *entry {
                            *entry = end;
                        }
                    }
                }
            }
        }

        let mut sorted = allowed_tokens.into_iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().map(|(token, _)| token).collect()
    }

    fn get_trie_node_from_trie<'a>(trie: &'a GenresTrieNode, prefix: &str) -> Option<&'a GenresTrieNode> {
        let mut node = trie;
        for ch in prefix.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }

    fn get_allowed_genres_tokens(&self) -> Vec<usize> {
        if !self.has_genres_vocab {
            return Vec::new();
        }

        let accumulated = self.accumulated_text.to_lowercase();
        let current_prefix = accumulated.trim().to_string();
        let Some(current_node) = (if current_prefix.is_empty() {
            Some(&self.genres_trie)
        } else {
            Self::get_trie_node_from_trie(&self.genres_trie, &current_prefix)
        }) else {
            return self.newline_token_id.into_iter().collect();
        };

        let valid_next_chars = current_node.children.keys().copied().collect::<HashSet<char>>();
        let is_complete = current_node.end;

        if valid_next_chars.is_empty() {
            return if is_complete {
                self.newline_token_id.into_iter().collect()
            } else {
                Vec::new()
            };
        }

        let mut candidate_tokens = HashSet::<usize>::new();
        for ch in &valid_next_chars {
            if let Some(tokens) = self.char_to_tokens.get(ch) {
                for id in tokens {
                    candidate_tokens.insert(*id);
                }
            }
        }

        let mut allowed = HashSet::<usize>::new();
        for token_id in candidate_tokens {
            let Some(decoded_normalized) = self.token_to_text.get(&token_id) else {
                continue;
            };

            if decoded_normalized.trim().is_empty() {
                if valid_next_chars.contains(&' ') || valid_next_chars.contains(&',') {
                    allowed.insert(token_id);
                }
                continue;
            }

            let new_prefix = format!("{current_prefix}{decoded_normalized}");
            if Self::get_trie_node_from_trie(&self.genres_trie, &new_prefix).is_some() {
                allowed.insert(token_id);
            }
        }

        if is_complete {
            if let Some(newline_id) = self.newline_token_id {
                allowed.insert(newline_id);
            }
        }

        let mut out = allowed.into_iter().collect::<Vec<_>>();
        out.sort_unstable();
        out
    }

    fn apply_tree_constraint(
        &self,
        logits: &mut [f32],
        tree: &HashMap<Vec<usize>, Vec<usize>>,
    ) {
        let allowed = tree
            .get(&self.accumulated_token_ids)
            .cloned()
            .unwrap_or_else(|| self.newline_token_id.into_iter().collect());
        whitelist(logits, &allowed);
    }

    fn apply(
        &mut self,
        logits: &mut [f32],
        pad_token_id: Option<usize>,
        eos_token_id: Option<usize>,
        im_end_token_id: Option<usize>,
        audio_code_token_ids: &[usize],
    ) {
        mask_id(logits, pad_token_id);
        if self.in_metadata_phase() {
            mask_id(logits, eos_token_id);
            mask_id(logits, im_end_token_id);
        }

        loop {
            let Some(fixed) = self.fixed_string_for_state() else {
                break;
            };
            let allowed = self.allowed_tokens_for_fixed_string(fixed);
            if allowed.is_empty() {
                self.transition_to_next_state();
                continue;
            }
            whitelist(logits, &allowed);
            return;
        }

        match self.state {
            ConstrainedState::BpmValue => {
                self.apply_tree_constraint(logits, &self.bpm_prefix_tree);
            }
            ConstrainedState::DurationValue => {
                self.apply_tree_constraint(logits, &self.duration_prefix_tree);
            }
            ConstrainedState::TimesigValue => {
                self.apply_tree_constraint(logits, &self.timesig_prefix_tree);
            }
            ConstrainedState::KeyscaleValue => {
                self.apply_tree_constraint(logits, &self.keyscale_prefix_tree);
            }
            ConstrainedState::LanguageValue => {
                if !self.language_forced_tokens.is_empty() {
                    if self.accumulated_token_ids.len() < self.language_forced_tokens.len() {
                        keep_only(logits, self.language_forced_tokens[self.accumulated_token_ids.len()]);
                    } else if let Some(newline_id) = self.newline_token_id {
                        keep_only(logits, newline_id);
                    }
                    return;
                }

                if self.accumulated_token_ids.is_empty() {
                    let candidates = self
                        .language_prefix_tree
                        .get(&Vec::<usize>::new())
                        .cloned()
                        .unwrap_or_default();
                    if candidates.is_empty() {
                        if let Some(newline_id) = self.newline_token_id {
                            keep_only(logits, newline_id);
                        }
                        return;
                    }
                    let mut best_id = candidates[0];
                    let mut best_value = f32::NEG_INFINITY;
                    for &id in &candidates {
                        if id < logits.len() && logits[id] > best_value {
                            best_value = logits[id];
                            best_id = id;
                        }
                    }
                    keep_only(logits, best_id);
                    return;
                }

                self.apply_tree_constraint(logits, &self.language_prefix_tree);
            }
            ConstrainedState::CaptionValue => {
                if let Some(backtick_id) = self.backtick_token_id {
                    if backtick_id < logits.len() {
                        logits[backtick_id] = -1.0e30_f32;
                    }
                }
                for id in audio_code_token_ids {
                    if *id < logits.len() {
                        logits[*id] = -1.0e30_f32;
                    }
                }

                if self.caption_after_newline && !self.caption_ending {
                    let top_token_id = argmax(logits);
                    let top_token_text = self.decode_token(top_token_id);
                    if let Some(first) = top_token_text.chars().next() {
                        if first != ' ' && first != '\t' {
                            self.caption_ending = true;
                            self.caption_after_newline = false;
                            self.pending_field_name.clear();
                            return;
                        }
                    }
                    self.caption_after_newline = false;
                }

                if self.caption_token_count >= 512 {
                    if let Some(newline_id) = self.newline_token_id {
                        keep_only(logits, newline_id);
                    }
                }
            }
            ConstrainedState::GenresValue => {
                if self.has_genres_vocab {
                    let allowed = self.get_allowed_genres_tokens();
                    if !allowed.is_empty() {
                        whitelist(logits, &allowed);
                    } else if let Some(newline_id) = self.newline_token_id {
                        keep_only(logits, newline_id);
                    }
                    return;
                } else {
                    if let Some(backtick_id) = self.backtick_token_id {
                        if backtick_id < logits.len() {
                            logits[backtick_id] = -1.0e30_f32;
                        }
                    }
                    for id in audio_code_token_ids {
                        if *id < logits.len() {
                            logits[*id] = -1.0e30_f32;
                        }
                    }
                    if self.accumulated_text.trim().is_empty() {
                        mask_id(logits, self.newline_token_id);
                    }
                }
            }
            ConstrainedState::Completed => {
                for id in audio_code_token_ids {
                    if *id < logits.len() {
                        logits[*id] = -1.0e30_f32;
                    }
                }
            }
            _ => {}
        }
    }

    fn on_token(&mut self, token_id: usize) {
        if self.state == ConstrainedState::Completed {
            return;
        }

        if let Some(fixed) = self.fixed_string_for_state() {
            let token_text = self.decode_token(token_id);
            self.position_in_state += token_text.len();
            if self.position_in_state >= fixed.len() {
                self.transition_to_next_state();
            }
            return;
        }

        let token_text = self.decode_token(token_id);
        match self.state {
            ConstrainedState::BpmValue
            | ConstrainedState::DurationValue
            | ConstrainedState::TimesigValue
            | ConstrainedState::KeyscaleValue
            | ConstrainedState::LanguageValue => {
                if Some(token_id) == self.newline_token_id {
                    self.transition_to_next_state();
                } else {
                    self.accumulated_token_ids.push(token_id);
                    self.accumulated_text.push_str(&token_text);
                }
            }
            ConstrainedState::GenresValue => {
                if Some(token_id) == self.newline_token_id {
                    self.transition_to_next_state();
                } else {
                    self.accumulated_text.push_str(&token_text);
                }
            }
            ConstrainedState::CaptionValue => {
                self.caption_token_count += 1;
                self.accumulated_text.push_str(&token_text);

                if self.caption_ending {
                    self.pending_field_name.push_str(&token_text);
                    if token_text.contains(':') {
                        let field_name = self
                            .pending_field_name
                            .split(':')
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_lowercase();
                        if let Some(state) = Self::state_from_field_name(&field_name) {
                            self.state = state;
                            self.position_in_state = 0;
                            self.accumulated_token_ids.clear();
                            self.accumulated_text.clear();
                            self.caption_after_newline = false;
                            self.caption_ending = false;
                            self.pending_field_name.clear();
                        } else {
                            self.caption_ending = false;
                            self.pending_field_name.clear();
                            self.transition_to_next_state();
                        }
                    }
                    return;
                }

                if token_text.contains('\n') {
                    self.caption_after_newline = true;
                } else {
                    self.caption_after_newline = false;
                }
            }
            _ => {
                return;
            }
        }
    }
}

impl LmPipeline {
    pub fn new(
        onnx_dir: PathBuf,
        lm_model_variant: &str,
        tokenizer_path: Option<PathBuf>,
    ) -> Result<Self> {
        let tag = if lm_model_variant == "0.6B" { "0p6" } else { "1p7" };
        let contract_path = onnx_dir.join(format!("io_contract_lm_{tag}.json"));
        let contract_text = std::fs::read_to_string(&contract_path)
            .with_context(|| format!("read lm contract: {}", contract_path.display()))?;
        let contract: LmIoContract = serde_json::from_str(&contract_text).context("parse lm io contract")?;

        let tokenizer_path = match tokenizer_path {
            Some(v) => v,
            None => {
                let model_dir = if lm_model_variant == "0.6B" {
                    "acestep-5Hz-lm-0.6B"
                } else {
                    "acestep-5Hz-lm-1.7B"
                };
                PathBuf::from(format!("checkpoints/{model_dir}/tokenizer.json"))
            }
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {}", tokenizer_path.display(), e))?;

        let mut audio_code_token_ids = Vec::new();
        let audio_code_re = Regex::new(r"^<\|audio_code_\d+\|>$").expect("valid regex");
        for (token, id) in tokenizer.get_vocab(true) {
            if audio_code_re.is_match(&token) {
                audio_code_token_ids.push(id as usize);
            }
        }
        audio_code_token_ids.sort_unstable();

        let newline_token_id = tokenizer
            .token_to_id("\n")
            .map(|v| v as usize)
            .or_else(|| {
                tokenizer
                    .encode("\n", false)
                    .ok()
                    .and_then(|e| e.get_ids().first().copied().map(|v| v as usize))
            });

        Ok(Self {
            sessions: OrtSessionManager::new(onnx_dir),
            contract,
            eos_token_id: tokenizer.token_to_id("<|endoftext|>").map(|v| v as usize),
            pad_token_id: tokenizer.token_to_id("<|pad|>").map(|v| v as usize),
            newline_token_id,
            im_end_token_id: tokenizer.token_to_id("<|im_end|>").map(|v| v as usize),
            backtick_token_id: tokenizer.token_to_id("`").map(|v| v as usize),
            tokenizer,
            audio_code_token_ids,
        })
    }

    pub fn generate_sample_from_query(
        &mut self,
        query: &str,
        instrumental: bool,
        vocal_language: &str,
        opts: &LmGenerateOptions,
    ) -> Result<LmSimpleSample> {
        let prompt = self.build_inspiration_prompt(query, instrumental);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenize lm prompt: {}", e))?;
        let prompt_ids: Vec<i64> = encoding.get_ids().iter().copied().map(i64::from).collect();

        let (mut logits, mut cache) = self.run_prefill(&prompt_ids)?;
        let mut generated_u32: Vec<u32> = Vec::new();
        let mut generated_i64: Vec<i64> = Vec::new();
        let mut constraints = if opts.constrained {
            Some(self.build_constraint_tracker(vocal_language)?)
        } else {
            None
        };

        for _ in 0..opts.max_new_tokens {
            let mut logits_last = extract_last_logits(&logits)?;
            if let Some(state) = constraints.as_mut() {
                state.apply(
                    &mut logits_last,
                    self.pad_token_id,
                    self.eos_token_id,
                    self.im_end_token_id,
                    &self.audio_code_token_ids,
                );
            }
            let next_token_id = if opts.deterministic {
                argmax(&logits_last)
            } else {
                argmax(&logits_last)
            } as i64;

            generated_i64.push(next_token_id);
            generated_u32.push(next_token_id as u32);
            if let Some(state) = constraints.as_mut() {
                state.on_token(next_token_id as usize);
            }

            if self.should_stop(next_token_id as usize) {
                break;
            }

            let total_len = prompt_ids.len() + generated_i64.len();
            let (next_logits, next_cache) = self.run_decode(next_token_id, total_len, &cache)?;
            logits = next_logits;
            cache = next_cache;
        }

        let raw_text = self
            .tokenizer
            .decode(&generated_u32, false)
            .map_err(|e| anyhow::anyhow!("decode lm output: {}", e))?;
        let (mut metadata, _) = parse_lm_output(&raw_text)?;
        let mut lyrics = extract_lyrics_from_output(&raw_text)?;
        if lyrics.is_empty() && instrumental {
            lyrics = "[Instrumental]".to_string();
        }
        if !lyrics.is_empty() {
            metadata.insert("lyrics".to_string(), Value::String(lyrics.clone()));
        }
        metadata.insert("instrumental".to_string(), Value::Bool(instrumental));

        let language = vocal_language.trim().to_lowercase();
        if !language.is_empty() && language != "unknown" {
            metadata.insert("language".to_string(), Value::String(vocal_language.trim().to_string()));
        }

        let caption = metadata
            .get("caption")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if lyrics.is_empty() {
            lyrics = metadata
                .get("lyrics")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }

        Ok(LmSimpleSample {
            token_ids: generated_i64,
            raw_text,
            metadata,
            caption,
            lyrics,
            instrumental,
        })
    }

    fn build_inspiration_prompt(&self, query: &str, instrumental: bool) -> String {
        let query_text = if query.trim().is_empty() {
            "NO USER INPUT"
        } else {
            query.trim()
        };
        let inst = if instrumental { "true" } else { "false" };
        format!(
            "<|im_start|>system\n# Instruction\n{DEFAULT_LM_INSPIRED_INSTRUCTION}\n\n<|im_end|>\n<|im_start|>user\n{query_text}\n\ninstrumental: {inst}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    fn run_prefill(&mut self, prompt_ids: &[i64]) -> Result<(ArrayD<f32>, Vec<ArrayD<f32>>)> {
        let sess = self
            .sessions
            .get_mut(&self.contract.prefill_path)
            .with_context(|| format!("load {}", self.contract.prefill_path))?;
        let input_ids =
            Array2::<i64>::from_shape_vec((1, prompt_ids.len()), prompt_ids.to_vec()).context("prefill input_ids shape")?;
        let attention_mask = Array2::<i64>::ones((1, prompt_ids.len()));
        let outputs = sess.run(::ort::inputs![
            "input_ids" => TensorRef::from_array_view(&input_ids)?,
            "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
        ])?;
        extract_lm_outputs(outputs, self.contract.num_layers)
    }

    fn run_decode(
        &mut self,
        next_token_id: i64,
        total_len: usize,
        cache: &[ArrayD<f32>],
    ) -> Result<(ArrayD<f32>, Vec<ArrayD<f32>>)> {
        let sess = self
            .sessions
            .get_mut(&self.contract.decode_path)
            .with_context(|| format!("load {}", self.contract.decode_path))?;
        let input_ids = Array2::<i64>::from_shape_vec((1, 1), vec![next_token_id]).context("decode input_ids shape")?;
        let attention_mask = Array2::<i64>::ones((1, total_len));

        let mut inputs = ::ort::inputs![
            "input_ids" => TensorRef::from_array_view(&input_ids)?,
            "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
        ];
        for layer in 0..self.contract.num_layers {
            let k_name = format!("past_key_{layer}");
            let v_name = format!("past_value_{layer}");
            inputs.push((
                k_name.into(),
                TensorRef::from_array_view(&cache[2 * layer])?.into(),
            ));
            inputs.push((
                v_name.into(),
                TensorRef::from_array_view(&cache[2 * layer + 1])?.into(),
            ));
        }
        let outputs = sess.run(inputs)?;
        extract_lm_outputs(outputs, self.contract.num_layers)
    }

    fn build_constraint_tracker(&self, vocal_language: &str) -> Result<ConstraintTracker> {
        ConstraintTracker::new(
            &self.tokenizer,
            vocal_language,
            self.newline_token_id,
            self.backtick_token_id,
        )
    }

    fn should_stop(&self, token_id: usize) -> bool {
        Some(token_id) == self.eos_token_id || Some(token_id) == self.pad_token_id || Some(token_id) == self.im_end_token_id
    }
}

fn extract_lm_outputs(outputs: ::ort::session::SessionOutputs<'_>, num_layers: usize) -> Result<(ArrayD<f32>, Vec<ArrayD<f32>>)> {
    let logits = outputs
        .get("logits")
        .context("lm output missing logits")?
        .try_extract_array::<f32>()?
        .to_owned()
        .into_dyn();
    let mut cache = Vec::with_capacity(num_layers * 2);
    for layer in 0..num_layers {
        let k_name = format!("present_key_{layer}");
        let v_name = format!("present_value_{layer}");
        let key = outputs
            .get(k_name.as_str())
            .with_context(|| format!("lm output missing {k_name}"))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        let value = outputs
            .get(v_name.as_str())
            .with_context(|| format!("lm output missing {v_name}"))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        cache.push(key);
        cache.push(value);
    }
    Ok((logits, cache))
}

fn extract_last_logits(logits: &ArrayD<f32>) -> Result<Vec<f32>> {
    let logits3 = logits
        .view()
        .into_dimensionality::<Ix3>()
        .context("lm logits shape should be [B,T,V]")?;
    let t = logits3.shape()[1];
    if t == 0 {
        anyhow::bail!("lm logits has zero sequence length")
    }
    Ok(logits3.slice(s![0, t - 1, ..]).to_vec())
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if *value > best {
            best = *value;
            best_idx = idx;
        }
    }
    best_idx
}

fn keep_only(logits: &mut [f32], keep_id: usize) {
    for (idx, value) in logits.iter_mut().enumerate() {
        if idx != keep_id {
            *value = -1.0e30_f32;
        }
    }
}

fn mask_id(logits: &mut [f32], id: Option<usize>) {
    if let Some(idx) = id {
        if idx < logits.len() {
            logits[idx] = -1.0e30_f32;
        }
    }
}

fn whitelist(logits: &mut [f32], allowed_ids: &[usize]) {
    if allowed_ids.is_empty() {
        for v in logits.iter_mut() {
            *v = -1.0e30_f32;
        }
        return;
    }
    let mut masked = vec![-1.0e30_f32; logits.len()];
    for id in allowed_ids {
        if *id < logits.len() {
            masked[*id] = logits[*id];
        }
    }
    logits.copy_from_slice(&masked);
}

fn encode_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<usize>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize '{text}': {}", e))?;
    Ok(enc.get_ids().iter().copied().map(|v| v as usize).collect())
}

fn build_prefixed_value_tree(
    tokenizer: &Tokenizer,
    values: &[String],
    context_prefix_for_matching: &str,
    context_prefix_for_tokenization: &str,
    newline_token_id: Option<usize>,
) -> Result<HashMap<Vec<usize>, Vec<usize>>> {
    let context_token_ids = encode_ids(tokenizer, context_prefix_for_matching)?;
    let mut tree: HashMap<Vec<usize>, BTreeSet<usize>> = HashMap::new();

    for value in values {
        let full_text = format!("{context_prefix_for_tokenization}{value}");
        let full_tokens = encode_ids(tokenizer, &full_text)?;
        let value_tokens = if full_tokens.starts_with(&context_token_ids) {
            full_tokens[context_token_ids.len()..].to_vec()
        } else {
            encode_ids(tokenizer, &format!(" {value}"))?
        };
        if value_tokens.is_empty() {
            continue;
        }

        for idx in 0..value_tokens.len() {
            let prefix = value_tokens[..idx].to_vec();
            tree.entry(prefix).or_default().insert(value_tokens[idx]);
        }
        if let Some(newline_id) = newline_token_id {
            tree.entry(value_tokens.clone()).or_default().insert(newline_id);
        }
    }

    let out = tree
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect::<Vec<_>>()))
        .collect::<HashMap<_, _>>();
    Ok(out)
}

fn build_valid_keyscales() -> Vec<String> {
    let notes = ["A", "B", "C", "D", "E", "F", "G"];
    let accidentals = ["", "#", "b"];
    let modes = ["major", "minor"];
    let mut values = Vec::new();
    for note in notes {
        for accidental in accidentals {
            for mode in modes {
                values.push(format!("{note}{accidental} {mode}"));
            }
        }
    }
    values
}

fn build_genres_trie_from_file(path: &str) -> (GenresTrieNode, bool) {
    let Ok(text) = fs::read_to_string(path) else {
        return (GenresTrieNode::default(), false);
    };
    let mut root = GenresTrieNode::default();
    let mut loaded = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        loaded = true;
        let genre = line.to_lowercase();
        let mut node = &mut root;
        for ch in genre.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.end = true;
    }
    (root, loaded)
}

fn precompute_char_token_mapping(tokenizer: &Tokenizer) -> (HashMap<char, Vec<usize>>, HashMap<usize, String>) {
    let vocab_size = tokenizer.get_vocab_size(false);
    let mut char_to_tokens_set: HashMap<char, HashSet<usize>> = HashMap::new();
    let mut token_to_text: HashMap<usize, String> = HashMap::new();

    for token_id in 0..vocab_size {
        let Ok(decoded) = tokenizer.decode(&[token_id as u32], false) else {
            continue;
        };
        if decoded.is_empty() {
            continue;
        }

        let lower = decoded.to_lowercase();
        let normalized = if lower.trim().is_empty() {
            " ".to_string()
        } else {
            lower.trim_end().to_string()
        };
        token_to_text.insert(token_id, normalized);

        if let Some(first) = decoded.chars().next() {
            let c = first.to_ascii_lowercase();
            char_to_tokens_set.entry(c).or_default().insert(token_id);
        }
        let stripped = decoded.trim_start();
        if stripped.len() < decoded.len() {
            if let Some(first_nonspace) = stripped.chars().next() {
                let c = first_nonspace.to_ascii_lowercase();
                char_to_tokens_set.entry(c).or_default().insert(token_id);
            }
        }
    }

    let mut char_to_tokens = HashMap::<char, Vec<usize>>::new();
    for (ch, ids) in char_to_tokens_set {
        let mut v = ids.into_iter().collect::<Vec<_>>();
        v.sort_unstable();
        char_to_tokens.insert(ch, v);
    }
    (char_to_tokens, token_to_text)
}

fn parse_lm_output(output_text: &str) -> Result<(HashMap<String, Value>, String)> {
    let mut metadata = HashMap::new();

    let code_re = Regex::new(r"<\|audio_code_\d+\|>").context("compile audio_code regex")?;
    let audio_codes = code_re
        .find_iter(output_text)
        .map(|m| m.as_str().to_string())
        .collect::<Vec<_>>()
        .join("");

    let mut reasoning_text = None::<String>;
    let think_re = Regex::new(r"(?s)<think>(.*?)</think>").context("compile think regex")?;
    if let Some(cap) = think_re.captures(output_text) {
        if let Some(m) = cap.get(1) {
            reasoning_text = Some(m.as_str().trim().to_string());
        }
    }
    if reasoning_text.is_none() {
        let split_idx = output_text.find("<|audio_code_").unwrap_or(output_text.len());
        reasoning_text = Some(output_text[..split_idx].trim().to_string());
    }
    let reasoning_text = reasoning_text.unwrap_or_default();

    let mut current_key = String::new();
    let mut current_lines: Vec<String> = Vec::new();
    let save_field = |key: &str, lines: &[String], meta: &mut HashMap<String, Value>| {
        if key.is_empty() || lines.is_empty() {
            return;
        }
        let joined = lines.join("\n");
        match key {
            "bpm" => {
                if let Ok(v) = joined.trim().parse::<i64>() {
                    meta.insert("bpm".to_string(), json!(v));
                } else {
                    meta.insert("bpm".to_string(), Value::String(joined.trim().to_string()));
                }
            }
            "duration" => {
                if let Ok(v) = joined.trim().parse::<i64>() {
                    meta.insert("duration".to_string(), json!(v));
                } else {
                    meta.insert("duration".to_string(), Value::String(joined.trim().to_string()));
                }
            }
            "caption" => {
                let caption = joined
                    .split('\n')
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                meta.insert("caption".to_string(), Value::String(caption));
            }
            "genres" | "keyscale" | "language" | "timesignature" => {
                meta.insert(key.to_string(), Value::String(joined.trim().to_string()));
            }
            _ => {}
        }
    };

    for line in reasoning_text.lines() {
        if line.trim_start().starts_with('<') {
            continue;
        }
        let starts_new_key = !line.starts_with(' ') && !line.starts_with('\t') && line.contains(':');
        if starts_new_key {
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

    Ok((metadata, audio_codes))
}

fn extract_lyrics_from_output(output_text: &str) -> Result<String> {
    let think_end_re = Regex::new(r"</think>").context("compile think_end regex")?;
    let Some(m) = think_end_re.find(output_text) else {
        return Ok(String::new());
    };
    let mut tail = output_text[m.end()..].trim().to_string();
    if tail.is_empty() {
        return Ok(String::new());
    }
    let lyric_header_re = Regex::new(r"(?i)^#\s*lyri[c|cs]?\s*\n").context("compile lyric header regex")?;
    tail = lyric_header_re.replace(&tail, "").to_string();
    let im_end_re = Regex::new(r"<\|im_end\|>\s*$").context("compile im_end regex")?;
    tail = im_end_re.replace(&tail, "").to_string();
    Ok(tail.trim().to_string())
}
