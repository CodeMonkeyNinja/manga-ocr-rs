//! Japanese manga OCR — image-to-text for scanned manga and printed Japanese.
//!
//! Runs [mayocream/manga-ocr-onnx](https://huggingface.co/mayocream/manga-ocr-onnx)
//! (kha-white/manga-ocr-base exported to ONNX) via ONNX Runtime.
//! Returns raw Japanese text; no translation, no furigana stripping.
//!
//! Handles yokogumi (horizontal), tategaki (vertical), and tegaki
//! (handwritten) text.  Images are squish-resized to 224×224 matching the
//! original training pipeline.
//!
//! # Quick start
//!
//! ```no_run
//! use manga_ocr_rs::MangaOcr;
//!
//! let ocr = MangaOcr::new(manga_ocr_rs::default_model_dir()).unwrap();
//! let img = image::open("panel.png").unwrap();
//! println!("{}", ocr.recognize(&img).unwrap());
//! ```
//!
//! Models are downloaded automatically on first `cargo build` via `build.rs`.
//! Override the location by setting `MANGA_OCR_MODELS_DIR` before building.

use anyhow::{Context, Result};
use image::{imageops, DynamicImage};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use std::cmp::Ordering;
use std::path::Path;
use std::sync::Mutex;

// ── Preprocessing ─────────────────────────────────────────────────────────────
const IMG_SIZE: usize = 224;
const PIXEL_MEAN: f32 = 0.5;
const PIXEL_STD: f32  = 0.5;

// ── Generation (from generation_config.json) ──────────────────────────────────
const DECODER_START_TOKEN_ID: i64 = 2;
const EOS_TOKEN_ID: i64           = 3;
const MAX_DECODE_STEPS: usize     = 300;
const NUM_BEAMS: usize            = 4;
const LENGTH_PENALTY: f32         = 2.0;
const NO_REPEAT_NGRAM: usize      = 3;

// ── Default model directory (set by build.rs) ─────────────────────────────────

/// Returns the directory where `build.rs` downloaded (or expects) the model files.
///
/// Set `MANGA_OCR_MODELS_DIR` at build time to override:
/// ```bash
/// MANGA_OCR_MODELS_DIR=/my/models cargo build
/// ```
pub fn default_model_dir() -> &'static Path {
    Path::new(env!("MANGA_OCR_DEFAULT_MODEL_DIR"))
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

/// Preprocess matching kha-white/manga-ocr-base's ViTImageProcessor:
///
/// 1. Grayscale → RGB  (`convert("L").convert("RGB")` in the original)
/// 2. Resize directly to 224×224 with Bilinear  (squishes — no padding)
/// 3. Normalise to [-1, 1]  (mean=0.5, std=0.5)
///
/// The original model was trained on squished (non-aspect-preserving) resizes,
/// so we must NOT centre-pad to square — doing so degrades accuracy.
///
/// Returns shape `[1, 3, H, W]` + flat NCHW `Vec<f32>`.
fn preprocess(img: &DynamicImage) -> ([usize; 4], Vec<f32>) {
    // Bilinear matches preprocessor_config.json "resample": 2 (PIL.Image.BILINEAR)
    let resized = img
        .grayscale()
        .resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, imageops::FilterType::Triangle)
        .to_rgb8();

    let mut flat = vec![0.0f32; 3 * IMG_SIZE * IMG_SIZE];
    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let p = resized.get_pixel(x as u32, y as u32);
            flat[0 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (p[0] as f32 / 255.0 - PIXEL_MEAN) / PIXEL_STD;
            flat[1 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (p[1] as f32 / 255.0 - PIXEL_MEAN) / PIXEL_STD;
            flat[2 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (p[2] as f32 / 255.0 - PIXEL_MEAN) / PIXEL_STD;
        }
    }
    ([1, 3, IMG_SIZE, IMG_SIZE], flat)
}

// ── Vocabulary ────────────────────────────────────────────────────────────────
//
// vocab.txt is a line-indexed file: token ID = line number (0-based).
// Special tokens to skip when decoding:
//   0=[PAD]  1=[UNK]  2=[CLS]/BOS  3=[SEP]/EOS  4=[MASK]  5-14=<unused0-9>

struct VocabDecoder {
    tokens: Vec<String>,
}

impl VocabDecoder {
    fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let tokens = content.lines().map(str::to_owned).collect();
        Ok(Self { tokens })
    }

    fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|&id| {
                let uid = id as usize;
                if uid < 15 { return None; }
                self.tokens.get(uid)
            })
            .map(|tok| tok.trim_start_matches("##"))
            .collect()
    }
}

// ── Beam search helpers ───────────────────────────────────────────────────────

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_z = sum_exp.ln() + max;
    logits.iter().map(|&x| x - log_z).collect()
}

fn top_k(log_probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = log_probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    v.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v.truncate(k);
    v
}

fn apply_no_repeat_ngram(ids: &[i64], log_probs: &mut [f32]) {
    let n = NO_REPEAT_NGRAM;
    if ids.len() < n - 1 || n < 2 { return; }
    let prefix = &ids[ids.len() - (n - 1)..];
    for i in 0..ids.len().saturating_sub(n - 1) {
        if ids[i..i + n - 1] == *prefix {
            let tok = ids[i + n - 1] as usize;
            if tok < log_probs.len() {
                log_probs[tok] = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// OCR engine wrapping the encoder + decoder ONNX sessions and vocabulary.
///
/// Construct once (model load is expensive), then call [`recognize`] repeatedly.
/// `Session::run` requires `&mut Session`, so each session is wrapped in a
/// `Mutex` — this lets `MangaOcr` be shared as `Arc<MangaOcr>` across threads.
///
/// [`recognize`]: MangaOcr::recognize
pub struct MangaOcr {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    vocab:   VocabDecoder,
}

impl MangaOcr {
    /// Load models from `model_dir`.
    ///
    /// Expects these files inside the directory:
    /// - `encoder_model.onnx`
    /// - `decoder_model.onnx`
    /// - `vocab.txt`
    ///
    /// Use [`default_model_dir()`] to get the path that `build.rs` prepared.
    pub fn new(model_dir: &Path) -> Result<Self> {
        let enc_path = model_dir.join("encoder_model.onnx");
        let dec_path = model_dir.join("decoder_model.onnx");
        let tok_path = model_dir.join("vocab.txt");

        let encoder = Session::builder()
            .context("encoder: SessionBuilder")?
            .commit_from_file(&enc_path)
            .with_context(|| format!("encoder: open {}", enc_path.display()))?;

        let decoder = Session::builder()
            .context("decoder: SessionBuilder")?
            .commit_from_file(&dec_path)
            .with_context(|| format!("decoder: open {}", dec_path.display()))?;

        let vocab = VocabDecoder::from_file(&tok_path)?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            vocab,
        })
    }

    /// OCR one image crop.  Returns raw Japanese text; no translation.
    ///
    /// Works on any aspect ratio: tategaki (tall), yokogumi (wide), tegaki
    /// (handwritten).  Images are squish-resized to 224×224 (no padding),
    /// matching the original training pipeline.
    ///
    /// Uses beam search (4 beams) matching `generation_config.json`.
    pub fn recognize(&self, img: &DynamicImage) -> Result<String> {
        // ── 1. Encode ────────────────────────────────────────────────────────
        let (pv_shape, pv_data) = preprocess(img);
        let pv_tensor = OrtTensor::<f32>::from_array((pv_shape, pv_data))
            .context("pixel_values tensor")?;

        let (enc_seq_len, hidden_dim, enc_hidden) = {
            let mut enc = self.encoder.lock()
                .map_err(|e| anyhow::anyhow!("encoder lock poisoned: {e}"))?;
            let out = enc.run(ort::inputs!["pixel_values" => pv_tensor])
                .context("encoder run")?;
            let (shape, data) = out["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .context("encoder: extract last_hidden_state")?;
            (shape[1] as usize, shape[2] as usize, data.to_vec())
        };

        // ── 2. Beam search ───────────────────────────────────────────────────
        let seeds = {
            let mut lp = self.decoder_logprobs_single(
                &[DECODER_START_TOKEN_ID], enc_seq_len, hidden_dim, &enc_hidden,
            )?;
            apply_no_repeat_ngram(&[DECODER_START_TOKEN_ID], &mut lp);
            top_k(&lp, NUM_BEAMS)
        };

        let mut beams: Vec<(Vec<i64>, f32, bool)> = seeds.iter()
            .map(|&(tok, lp)| {
                let done = tok as i64 == EOS_TOKEN_ID;
                (vec![DECODER_START_TOKEN_ID, tok as i64], lp, done)
            })
            .collect();
        let mut completed: Vec<(Vec<i64>, f32)> = beams.iter()
            .filter(|(_, _, done)| *done)
            .map(|(ids, score, _)| (ids.clone(), *score))
            .collect();

        for _step in 1..MAX_DECODE_STEPS {
            let active: Vec<usize> = beams.iter().enumerate()
                .filter(|(_, (_, _, done))| !*done)
                .map(|(i, _)| i)
                .collect();
            if active.is_empty() { break; }

            let batch = active.len();
            let seq_len = beams[active[0]].0.len();

            let flat_ids: Vec<i64> = active.iter()
                .flat_map(|&i| beams[i].0.iter().copied())
                .collect();

            let flat_enc: Vec<f32> = enc_hidden.iter().copied()
                .cycle()
                .take(batch * enc_seq_len * hidden_dim)
                .collect();

            let batch_log_probs: Vec<Vec<f32>> = {
                let mut dec = self.decoder.lock()
                    .map_err(|e| anyhow::anyhow!("decoder lock poisoned: {e}"))?;
                let out = dec.run(ort::inputs![
                    "input_ids" =>
                        OrtTensor::<i64>::from_array(([batch, seq_len], flat_ids))
                            .context("input_ids tensor")?,
                    "encoder_hidden_states" =>
                        OrtTensor::<f32>::from_array(([batch, enc_seq_len, hidden_dim], flat_enc))
                            .context("encoder_hidden_states tensor")?
                ]).context("decoder batch run")?;

                let (logits_shape, logits_data) = out["logits"]
                    .try_extract_tensor::<f32>()
                    .context("logits")?;
                let vocab_size = logits_shape[2] as usize;

                (0..batch).map(|b| {
                    let offset = (b * seq_len + seq_len - 1) * vocab_size;
                    log_softmax(&logits_data[offset..offset + vocab_size])
                }).collect()
            };

            let mut candidates: Vec<(usize, i64, f32)> = Vec::with_capacity(batch * NUM_BEAMS);
            for (b, &beam_idx) in active.iter().enumerate() {
                let beam_score = beams[beam_idx].1;
                let mut lp = batch_log_probs[b].clone();
                apply_no_repeat_ngram(&beams[beam_idx].0, &mut lp);
                for (tok, lp) in top_k(&lp, NUM_BEAMS) {
                    candidates.push((beam_idx, tok as i64, beam_score + lp));
                }
            }
            candidates.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            candidates.truncate(NUM_BEAMS);

            beams = candidates.iter().map(|&(old_idx, tok, score)| {
                let mut ids = beams[old_idx].0.clone();
                let done = tok == EOS_TOKEN_ID;
                if !done { ids.push(tok); }
                (ids, score, done)
            }).collect();

            for (ids, score, done) in &beams {
                if *done { completed.push((ids.clone(), *score)); }
            }
        }

        for (ids, score, done) in &beams {
            if !done { completed.push((ids.clone(), *score)); }
        }

        // ── 3. Pick best beam (length-normalised score) ───────────────────────
        let best_ids = completed.iter()
            .max_by(|(ids_a, score_a), (ids_b, score_b)| {
                let norm = |ids: &[i64], s: f32| s / (ids.len() as f32).powf(LENGTH_PENALTY);
                norm(ids_a, *score_a).partial_cmp(&norm(ids_b, *score_b))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(ids, _)| ids.as_slice())
            .unwrap_or(&[]);

        // ── 4. Detokenise ─────────────────────────────────────────────────────
        let decode_ids = if best_ids.first() == Some(&DECODER_START_TOKEN_ID) {
            &best_ids[1..]
        } else {
            best_ids
        };
        Ok(self.vocab.decode(decode_ids))
    }

    fn decoder_logprobs_single(
        &self,
        ids: &[i64],
        enc_seq_len: usize,
        hidden_dim: usize,
        enc_hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let seq_len = ids.len();
        let mut dec = self.decoder.lock()
            .map_err(|e| anyhow::anyhow!("decoder lock poisoned: {e}"))?;
        let out = dec.run(ort::inputs![
            "input_ids" =>
                OrtTensor::<i64>::from_array(([1usize, seq_len], ids.to_vec()))
                    .context("input_ids tensor")?,
            "encoder_hidden_states" =>
                OrtTensor::<f32>::from_array(([1usize, enc_seq_len, hidden_dim], enc_hidden.to_vec()))
                    .context("encoder_hidden_states tensor")?
        ]).context("decoder bootstrap run")?;

        let (logits_shape, logits_data) = out["logits"]
            .try_extract_tensor::<f32>()
            .context("logits")?;
        let vocab_size = logits_shape[2] as usize;
        let last = &logits_data[(seq_len - 1) * vocab_size..seq_len * vocab_size];
        Ok(log_softmax(last))
    }
}
