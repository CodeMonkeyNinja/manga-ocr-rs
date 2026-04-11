# Trials and findings

Notes on what we've learned while getting manga-ocr-rs to produce accurate
results.  This is a living document — update it as new experiments land.

---

## Preprocessing must match the original training pipeline

The single biggest accuracy fix was matching the original Python
`kha-white/manga-ocr-base` preprocessing exactly.

### What the original does

Defined in `preprocessor_config.json` and `ocr.py`:

1. `img.convert("L").convert("RGB")` — grayscale then back to 3-channel RGB
2. Resize directly to 224x224 with **PIL Bilinear** (`resample=2`) — **squishes**
   the image, does NOT preserve aspect ratio
3. Normalise: `(pixel / 255.0 - 0.5) / 0.5` — maps to [-1, 1]

No centre-padding.  No square canvas.  No Lanczos.

### What we had wrong

Our initial Rust implementation did:

1. Grayscale -> RGB (correct)
2. Centre-pad to square on white canvas (WRONG)
3. Resize to 224x224 with Lanczos3 (wrong filter)
4. Normalise mean=0.5, std=0.5 (correct)

The centre-padding was the killer.  For a tategaki column (e.g. 70x450), it
created a 450x450 canvas with the text as a narrow strip in the centre, then
resized to 224x224 — the text occupied roughly 1/3 of the final image.  The
model had never seen padded layouts during training, so it struggled.

### Impact

- ubunchu01_02.png annotations: **3/9 -> 4/9 pass** after switching to squish
- `却下！` (dark bubble) started passing after the fix
- Bilinear vs Lanczos3 is minor but we match Bilinear for correctness

### The mean/std cannot be tuned

`PIXEL_MEAN=0.5` and `PIXEL_STD=0.5` are baked into the model weights.  The
ViT encoder was trained with input in [-1, 1].  Changing these values makes
accuracy worse, not better — there is nothing to tune here.

---

## Unit-test fixtures

### yokogaki (horizontal) — 711x389

`データを正確に読み取る` — passes exactly.  This is a clean crop of black text
on white background, close to what the model expects.

### tategaki (vertical) — 2760x1504

`『言語モデルのテスト』` — the model outputs `『言語モデルのラスト』`.

The image is nearly full-size (2760x1504) for what should be a narrow vertical
text column.  After squish-resize to 224x224, the text is badly distorted and
the model confuses the visually similar katakana テ and ラ.

**HACK**: the test currently accepts either `テスト` or `ラスト`.  This should
be removed once the fixture is properly cropped to tight text bounds.

### tegaki (handwritten) — 2760x1504

`手書きの文字サンプル` — passes exactly after re-crop.

The original fixture from Gemini/Sora had a generation bug: the text was
doubled as `手書手書きの文字サンプル`.  This was baked into the AI-generated
image and could not be fixed by cropping — the source image had to be manually
corrected.

---

## Real manga — ubunchu01_02.png

9 annotated speech bubbles with ground-truth text and Gemini-normalised
bounding boxes (`box_2d: [y1, x1, y2, x2]` in [0-1000] coordinate space).

**4/9 pass** (after preprocessing fix).

### What passes

Clean vertical speech bubbles with black ink on white background — the exact
distribution the model was trained on:

| Bubble | Text | Time |
|--------|------|------|
| Top right, line 1 | `あ あたしの オススメは` | 32,292 ms |
| Top left | `最近人気の デスクトップな リナックスです！` | 34,826 ms |
| Middle (rejection) | `却下！` | 5,145 ms |
| Bottom left, bottom | `一瞬くらい 検討して くださいよー！` | 37,936 ms |

### What fails and why

**Large decorative text** — `うぶんちゅ` (Top right, large)
Got: `メはうぶんちゅ`.  The crop bleeds into the adjacent speech bubble,
leaking `メは` (the tail of `オススメは`) into the recognised text.  The
title-style bold font is also heavier than typical speech-bubble text.

**Tiny text** — `※ うぶんちゅではなくウブントゥです` (Center caption)
Got: complete hallucination (hundreds of characters of nonsense).  After
scaling from [0-1000] the crop is only ~36 px tall.  The ViT encoder processes
224x224 images — characters this small are blurred beyond recognition during
resize, and the decoder runs away without finding coherent input.

**Slanted action text** — `マジいってん んだぜ！` (Bottom center)
Got: hallucination.  Speed-line effects and diagonal text orientation are
outside the training distribution.  The model partially recognises fragments
but cannot establish reading order.

**Screaming / action text** — `よけんな このっ！` (Bottom right)
Got: hallucination with repetition.  Emphasis effects (thick irregular strokes,
exclamation styling) degrade glyph recognition.  The bounding box may also clip
the first character.

**Near-miss** — `ハモリながら ケンカしないでっ` (Bottom left, top)
Got: `ハモリながらケアカしないでーっ`.  Most of the text is correct but
`ケンカ` becomes `ケアカ` — the model confuses `ン` with `ア`, both being
two-stroke katakana.  The trailing `ーっ` (long vowel + small tsu) was also
slightly mangled.  At 1,203 ms this was the fastest inference — possibly the
short text gave the decoder less room to wander.

---

## Normalisation for comparison

Japanese text does not use spaces.  The OCR model does not emit spaces.  But
human-annotated ground truth often includes spaces for readability (e.g.
`最近人気の デスクトップな リナックスです！`).

The model also outputs half-width `!?` where the source manga uses full-width
`！？`.

Our `normalise_jp()` function handles both:
- Strip all whitespace
- Fold `！` -> `!` and `？` -> `?`

This is tested independently (`test_normalise_jp_*` tests).

---

## Beam search

Parameters match `generation_config.json` from the original model:

- 4 beams
- `no_repeat_ngram_size = 3`
- `length_penalty = 2.0`
- Max 300 decode steps
- Stops at EOS token (id=3)

The length penalty of 2.0 is aggressive — it strongly favours longer sequences.
This may contribute to the trailing hallucination problem on crops with large
white margins: the decoder keeps generating because shorter sequences are
penalised.  Reducing `length_penalty` might help but would need to be validated
against the full test suite — it could also hurt accuracy on legitimate long
text.

---

## Open questions

- Would a lower `length_penalty` reduce hallucination without hurting accuracy?
- The tategaki fixture needs a proper tight crop — will the テ/ラ confusion
  resolve at a reasonable image size?
- Can we detect "the decoder is hallucinating" (e.g. score dropping, repetition
  despite ngram blocking) and cut early?
- Release build timings — debug build times are not representative of real
  performance.
