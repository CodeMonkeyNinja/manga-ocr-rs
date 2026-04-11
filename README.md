# manga-ocr-rs

Japanese manga OCR in pure Rust — no Python, no pip.

Runs [mayocream/manga-ocr-onnx](https://huggingface.co/mayocream/manga-ocr-onnx)
(the original [kha-white/manga-ocr-base](https://github.com/kha-white/manga-ocr) ONNX export)
via ONNX Runtime.  Returns raw Japanese text from an image crop; no translation,
no furigana stripping — pure image-to-text.

Handles yokogumi (horizontal), tategaki (vertical), and tegaki (handwritten) text.
Images are squish-resized to 224×224 matching the original training pipeline.

---

## Quick start

```toml
# Cargo.toml
[dependencies]
manga-ocr-rs = "0.1"
```

```rust
use manga_ocr_rs::MangaOcr;

// Models are downloaded automatically on first `cargo build` (~441 MB).
let ocr = MangaOcr::new(manga_ocr_rs::default_model_dir())?;
let img = image::open("panel.png")?;
println!("{}", ocr.recognize(&img)?);
```

The first `cargo build` downloads three files (~441 MB total) from HuggingFace
into `~/.cache/manga-ocr-rs/` via `curl`.  Subsequent builds are instant.

To use a pre-downloaded copy:

```bash
MANGA_OCR_MODELS_DIR=/path/to/models cargo build
```

---

## CLI

```bash
cargo install manga-ocr-rs

manga-ocr panel.png
manga-ocr inspect          # print model I/O names
```

---

## Test results

| Fixture | Size | Expected |
|---------|------|----------|
| `Unit-test-yokogaki.png` | 711×389 | `データを正確に読み取る` |
| `Unit-test-tategaki.png` | 2760×1504 | `『言語モデルのテスト』` |
| `Unit-test-tegaki.png`   | 2760×1504 | `手書きの文字サンプル` |

`tategaki` and `tegaki` fixtures are pending re-crop to tight text bounds;
the current images are near-full-size source exports and the tests are expected
to fail until properly cropped.  `yokogaki` passes.

Run with `cargo test --release` for representative timings (debug builds are
significantly slower due to unoptimised ONNX inference).

---

## Architecture

```
DynamicImage
    │
    ▼  preprocess()
    │  grayscale → RGB, squish-resize 224×224 Bilinear
    │  normalize mean=0.5 std=0.5
    │  shape: [1, 3, 224, 224]
    │
    ▼  encoder_model.onnx  (ViT, ~328 MB)
    │  last_hidden_state: [1, 196, 768]
    │
    ▼  decoder_model.onnx  (BERT, ~113 MB)
    │  beam search: 4 beams, batched per step
    │  no_repeat_ngram_size=3, length_penalty=2.0
    │  stops at EOS or 300 steps
    │
    ▼  vocab.txt  (~30 KB, line-indexed)
    │
    String  (raw Japanese)
```

Beam search parameters match `generation_config.json` from the original model.

---

## License

MIT — see [LICENSE](LICENSE).

Model: [mayocream/manga-ocr-onnx](https://huggingface.co/mayocream/manga-ocr-onnx)  
Original: [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr) (MIT)
