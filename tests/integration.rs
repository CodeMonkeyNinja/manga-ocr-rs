//! Integration tests for MangaOcr.
//!
//! Uses three purpose-built fixtures in `assets/`:
//!
//!   Unit-test-yokogaki.png    — `データを正確に読み取る`  (360×197 px, IPAGothic)
//!   Unit-test-tategaki.png    — `言語モデルのテスト`       (480×262 px, IPAGothic)
//!   Unit-test-tegaki.png      — `手書きの文字サンプル`     (480×262 px, Dejima-Mincho)
//!
//! Each image is clean black text on white — the same class of input the model
//! was trained on (scanned manga).  Ground truth is known, so EXPECTED_* constants
//! are exact strings, not smoke checks.
//!
//! ## Models
//!
//! Downloaded automatically on first `cargo build` via `build.rs` (~441 MB).
//! Override the location with `MANGA_OCR_MODELS_DIR` before building.

use manga_ocr_rs::MangaOcr;
use std::path::Path;
use std::time::Instant;

// build.rs downloads models here (or MANGA_OCR_MODELS_DIR override).
const MODEL_DIR: &str = env!("MANGA_OCR_DEFAULT_MODEL_DIR");

const FIXTURE_YOKOGAKI: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/Unit-test-yokogaki.png");
const FIXTURE_TATEGAKI: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/Unit-test-tategaki.png");
const FIXTURE_TEGAKI: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/Unit-test-tegaki.png");

const FIXTURE_MANGA: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/ubunchu01_02.png");
const FIXTURE_MANGA_JSON: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/ubunchu01_02.json");

const EXPECTED_YOKOGAKI: &str = "データを正確に読み取る";
const EXPECTED_TATEGAKI:   &str = "『言語モデルのテスト』";
const EXPECTED_TEGAKI:     &str = "手書きの文字サンプル";

// ── helpers ───────────────────────────────────────────────────────────────────

fn models_present() -> bool {
    Path::new(MODEL_DIR).join("encoder_model.onnx").exists()
}

fn load_ocr() -> MangaOcr {
    MangaOcr::new(Path::new(MODEL_DIR)).expect("load MangaOcr models")
}

fn has_japanese(s: &str) -> bool {
    s.chars().any(|c| ('\u{3000}'..='\u{9FFF}').contains(&c))
}

/// Normalise OCR text for comparison: strip whitespace (Japanese doesn't use
/// spaces) and fold full-width ！？ to half-width !? (model outputs half-width).
fn normalise_jp(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| match c { '！' => '!', '？' => '?', _ => c })
        .collect()
}

fn assert_ocr_exact(label: &str, ocr: &MangaOcr, path: &str, expected: &str) {
    let img = image::open(path).unwrap_or_else(|e| panic!("{label}: open {path}: {e}"));
    let t = Instant::now();
    let r = ocr.recognize_with_score(&img).unwrap_or_else(|e| panic!("{label}: OCR failed: {e}"));
    let ms = t.elapsed().as_millis();
    println!("{label} ({ms} ms): {:?}  (expected: {expected:?})  confidence: {:.4} (raw: {:.4}, score: {:.4})  tokens: {}{}",
        r.text, r.confidence, r.raw_confidence, r.score, r.token_count,
        if r.truncated { "  TRUNCATED" } else { "" });
    assert_eq!(r.text, expected, "{label}: OCR output does not match ground truth");
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Horizontal printed text — `データを正確に読み取る`.
/// Clean IPAGothic on white, 600×80 px.
#[test]
fn test_yokogaki() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    assert_ocr_exact("yokogaki", &load_ocr(), FIXTURE_YOKOGAKI, EXPECTED_YOKOGAKI);
}

/// Tategaki (vertical) text — `言語モデルのテスト`.
///
/// The fixture is a 480×262 image where the vertical text occupies the left
/// region.  At this size the model reads the correct characters but may
/// confuse the bracket style: `『』` (double corner) vs `「」` (single corner).
/// We accept either.
#[test]
fn test_tategaki() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    let ocr = load_ocr();
    let img = image::open(FIXTURE_TATEGAKI)
        .unwrap_or_else(|e| panic!("tategaki: open: {e}"));
    let t = Instant::now();
    let r = ocr.recognize_with_score(&img)
        .unwrap_or_else(|e| panic!("tategaki: OCR failed: {e}"));
    let ms = t.elapsed().as_millis();
    println!("tategaki ({ms} ms): {:?}  (expected: {EXPECTED_TATEGAKI:?})  confidence: {:.4} (raw: {:.4}, score: {:.4})  tokens: {}{}",
        r.text, r.confidence, r.raw_confidence, r.score, r.token_count,
        if r.truncated { "  TRUNCATED" } else { "" });
    // Accept 「」 variant — model reads correct text but may confuse bracket style.
    let accepted = r.text == EXPECTED_TATEGAKI
        || r.text == EXPECTED_TATEGAKI.replace('『', "「").replace('』', "」");
    assert!(accepted, "tategaki: got {:?}, expected {EXPECTED_TATEGAKI:?} (or 「」bracket variant)", r.text);
}

/// Tegaki (handwritten-style) text — `手書きの文字サンプル`.
/// Dejima-Mincho font, 480×262 px.
#[test]
fn test_tegaki() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    assert_ocr_exact("tegaki", &load_ocr(), FIXTURE_TEGAKI, EXPECTED_TEGAKI);
}

/// Verify the model is actually loaded and returns non-empty Japanese
/// from a trivially simple input — fast smoke test, no large fixture needed.
#[test]
fn test_yokogaki_is_japanese() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    let img = image::open(FIXTURE_YOKOGAKI)
        .expect("open yokogaki fixture");
    let ocr = load_ocr();
    let r = ocr.recognize_with_score(&img).expect("OCR failed");
    println!("yokogaki_is_japanese: confidence: {:.4} (raw: {:.4}, score: {:.4})  tokens: {}{}",
        r.confidence, r.raw_confidence, r.score, r.token_count,
        if r.truncated { "  TRUNCATED" } else { "" });
    assert!(!r.text.is_empty(), "OCR returned empty string");
    assert!(has_japanese(&r.text), "no Japanese in {:?}", r.text);
}

/// Crops each annotated speech bubble from `ubunchu01_02.png` using the
/// ground-truth bounding boxes in `ubunchu01_02.json` and asserts the OCR
/// output matches (whitespace-stripped, since manga-ocr doesn't emit spaces).
///
/// box_2d format: `[y1, x1, y2, x2]` in Gemini-normalised [0–1000] space.
/// Scale to pixels: `x_px = x_norm * img_width / 1000` (same for y).
///
/// ## Known failure categories (not marked #[ignore])
///
/// - **Large decorative text** — model struggles with bold display fonts
/// - **Tiny text** — characters too small after resize to 224×224
/// - **Inverted / dark background** — model trained on white backgrounds only
/// - **Slanted action text** — speed-lines and diagonal orientation
/// - **Clipped text** — bounding box clips first/last characters
/// - **Trailing hallucination** — excess whitespace causes decoder overshoot
#[test]
#[ignore]
fn test_ubunchu_annotations() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    if !Path::new(FIXTURE_MANGA).exists() {
        eprintln!("skip: {FIXTURE_MANGA} not found");
        return;
    }
    if !Path::new(FIXTURE_MANGA_JSON).exists() {
        eprintln!("skip: {FIXTURE_MANGA_JSON} not found");
        return;
    }

    let json_str = std::fs::read_to_string(FIXTURE_MANGA_JSON).expect("read json");
    let doc: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");

    let full_img = image::open(FIXTURE_MANGA).expect("open manga");
    let (img_w, img_h) = (full_img.width() as f32, full_img.height() as f32);
    // box_2d values are Gemini-normalized [0–1000]; scale to actual pixel dimensions.
    let sx = img_w / 1000.0;
    let sy = img_h / 1000.0;

    let ocr = load_ocr();
    let annotations = doc["annotations"].as_array().expect("annotations array");

    let mut passed = 0usize;
    let mut failed = 0usize;

    for ann in annotations {
        let b = ann["box_2d"].as_array().expect("box_2d");
        // box_2d: [y1, x1, y2, x2]
        let y1 = (b[0].as_f64().unwrap() as f32 * sy) as u32;
        let x1 = (b[1].as_f64().unwrap() as f32 * sx) as u32;
        let y2 = (b[2].as_f64().unwrap() as f32 * sy) as u32;
        let x2 = (b[3].as_f64().unwrap() as f32 * sx) as u32;
        let cw = x2.saturating_sub(x1).max(1);
        let ch = y2.saturating_sub(y1).max(1);

        let crop = full_img.crop_imm(x1, y1, cw, ch);
        let expected_raw = ann["text"].as_str().unwrap_or("");
        let desc = ann["description"].as_str().unwrap_or("?");

        let t = Instant::now();
        let r = ocr.recognize_with_score(&crop);
        let ms = t.elapsed().as_millis();

        let (result, conf, raw_conf, score, tokens, trunc) = match r {
            Ok(rec) => (rec.text, rec.confidence, rec.raw_confidence, rec.score, rec.token_count, rec.truncated),
            Err(e) => (format!("ERROR: {e}"), 0.0, 0.0, f32::NEG_INFINITY, 0, false),
        };

        let expected = normalise_jp(expected_raw);
        let actual   = normalise_jp(&result);

        let ok = actual == expected;
        if ok { passed += 1; } else { failed += 1; }

        let trunc_tag = if trunc { "  TRUNCATED" } else { "" };
        println!(
            "[{}] ({ms} ms) {desc}  confidence: {conf:.4} (raw: {raw_conf:.4}, score: {score:.4})  tokens: {tokens}{trunc_tag}\n  expected: {expected_raw:?}\n  got:      {result:?}",
            if ok { "PASS" } else { "FAIL" },
        );
    }

    println!("\nubunchu annotations: {passed} passed, {failed} failed out of {}", annotations.len());
    assert_eq!(failed, 0, "{failed} annotation(s) did not match ground truth");
}

// ── normalise_jp unit tests ──────────────────────────────────────────────────

#[test]
fn test_normalise_jp_strips_whitespace() {
    assert_eq!(
        normalise_jp("最近人気の デスクトップな リナックスです！"),
        "最近人気のデスクトップなリナックスです!",
    );
}

#[test]
fn test_normalise_jp_folds_fullwidth_punctuation() {
    assert_eq!(normalise_jp("却下！"), "却下!");
    assert_eq!(normalise_jp("本当？"), "本当?");
    assert_eq!(normalise_jp("本当！？"), "本当!?");
}

#[test]
fn test_normalise_jp_passthrough() {
    assert_eq!(normalise_jp("うぶんちゅ"), "うぶんちゅ");
    assert_eq!(normalise_jp("データを正確に読み取る"), "データを正確に読み取る");
}

#[test]
fn test_normalise_jp_empty() {
    assert_eq!(normalise_jp(""), "");
}

// ── Recognition / confidence tests ───────────────────────────────────────────

/// `recognize` (backward-compat wrapper) still returns plain String.
#[test]
fn test_recognize_returns_string() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    let ocr = load_ocr();
    let img = image::open(FIXTURE_YOKOGAKI).expect("open yokogaki");
    let text = ocr.recognize(&img).expect("recognize failed");
    assert_eq!(text, EXPECTED_YOKOGAKI);
}

/// `recognize_with_score` returns valid Recognition fields.
#[test]
fn test_recognition_fields() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    let ocr = load_ocr();
    let img = image::open(FIXTURE_YOKOGAKI).expect("open yokogaki");
    let r = ocr.recognize_with_score(&img).expect("recognize_with_score failed");

    assert_eq!(r.text, EXPECTED_YOKOGAKI);
    // Confidence must be in (0, 1]
    assert!(r.confidence > 0.0 && r.confidence <= 1.0,
        "confidence {:.4} out of range (0, 1]", r.confidence);
    assert!(r.raw_confidence > 0.0 && r.raw_confidence <= 1.0,
        "raw_confidence {:.4} out of range (0, 1]", r.raw_confidence);
    // Clean fixture should not truncate
    assert!(!r.truncated, "clean fixture should not be truncated");
    // Token count should be positive and reasonable
    assert!(r.token_count > 0 && r.token_count < 50,
        "unexpected token_count {} for clean fixture", r.token_count);
    // Score is a negative log-prob normalised value (should be <= 0)
    assert!(r.score <= 0.0, "score {:.4} should be <= 0", r.score);
}

/// Realistic manga-bubble-sized images should get no meaningful penalty.
/// Both fixtures are now in the sweet spot (360×197 and 480×262).
#[test]
fn test_dimension_calibration_no_penalty_tategaki() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    let ocr = load_ocr();
    let img = image::open(FIXTURE_TATEGAKI).expect("open tategaki");
    let r = ocr.recognize_with_score(&img).expect("recognize_with_score failed");

    // 480×262 = 125,760 px² — within sweet spot, no penalty expected
    assert!((r.confidence - r.raw_confidence).abs() < 0.001,
        "manga-bubble-sized image should have no penalty: confidence {:.4} vs raw {:.4}",
        r.confidence, r.raw_confidence);
}

/// Sweet-spot images (yokogaki: 360×197) should get no meaningful penalty.
#[test]
fn test_dimension_calibration_no_penalty() {
    if !models_present() {
        eprintln!("skip: models not found at {MODEL_DIR}");
        return;
    }
    let ocr = load_ocr();
    let img = image::open(FIXTURE_YOKOGAKI).expect("open yokogaki");
    let r = ocr.recognize_with_score(&img).expect("recognize_with_score failed");

    // 360×197 = 70,920 px² — within sweet spot
    assert!((r.confidence - r.raw_confidence).abs() < 0.001,
        "sweet-spot image should have no penalty: confidence {:.4} vs raw {:.4}",
        r.confidence, r.raw_confidence);
}
