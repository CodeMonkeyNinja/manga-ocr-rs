# manga-ocr-rs

Japanese manga OCR in pure Rust — no Python, no pip.

Runs [mayocream/manga-ocr-onnx](https://huggingface.co/mayocream/manga-ocr-onnx)
(the original [kha-white/manga-ocr-base](https://github.com/kha-white/manga-ocr) ONNX export)
via ONNX Runtime.  Returns raw Japanese text from an image crop; no translation,
no furigana stripping — pure image-to-text.

Handles yokogumi (horizontal), tategaki (vertical), and tegaki (handwritten) text
by centre-padding crops to a square before resizing, preserving character proportions
regardless of aspect ratio.

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

## Test results (debug build, beam search k=4)

| Fixture | Size | Expected | Result | Time |
|---------|------|----------|--------|------|
| `Unit-test-tegaki.png` | 500×80 | `手書きの文字サンプル` | ✓ exact | ~1 400 ms |
| `Unit-test-tategaki.png` | 70×450 | `言語モデルのテスト` | ✓ exact | ~12 200 ms |
| `Unit-test-horizontal.png` | 600×80 | `データを正確に読み取る` | ✓ exact | ~34 000 ms |

Times are unoptimized debug builds.  Release builds (`cargo test --release`) are
significantly faster.

---

## Architecture

```
DynamicImage
    │
    ▼  preprocess()
    │  grayscale → RGB, centre-pad to square (white fill)
    │  resize 224×224 Lanczos3, normalize mean=0.5 std=0.5
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
