# Changelog

## [1.0.0] — 2026-07-09

- **Stable API milestone** — no breaking changes planned. OCR pipeline,
  confidence scoring, and decoder bailout logic are mature and
  production-tested within the Lenzu ecosystem.

## [0.1.5] — 2026-06-xx

- CI: add publish workflow to release on tag push.

## [0.1.4] — 2026-05-xx

- **Security**: replace `native-tls` with `rustls` in `ort` dependency to
  eliminate OpenSSL vulnerabilities.

## [0.1.3] — 2026-04-xx

- **Early decoder bailout** — emit truncated result when decoder
  confidence drops below threshold, preventing runaway hallucination loops.

## [0.1.2] — 2026-04-xx

- **Configurable `max_decode_steps`** — default 50 (was hard-coded at 300).

## [0.1.1] — 2026-04-xx

- **Confidence scores** — surface `confidence`, `score`, `rating` fields
  from beam search (dimension-adjusted geometric mean of per-token
  probabilities).
- **Truncation flag** — `truncated` field when decoder hits step limit
  without emitting EOS.
- Fix `yokogumi` → `yokogaki` typo.
- Add `serde_json` dev-dependency; optimize test fixture images.
- Exclude model and test assets from crate package.

## [0.1.0] — 2026-04-xx

- Initial release: Japanese manga OCR via ViT encoder + BERT decoder
  (ONNX Runtime). Runs `mayocream/manga-ocr-onnx` with beam search k=4,
  no-repeat ngram size 3, length penalty 2.0.
- Handles yokogaki (horizontal), tategaki (vertical), and tegaki
  (handwritten) text.
- Automatic model download (~441 MB) on first build via `build.rs`.
- CLI binary `manga-ocr` for single-file recognition.
