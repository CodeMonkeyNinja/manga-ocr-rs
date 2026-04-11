# ubunchu01_02 annotation test analysis

Date: 2026-04-11

## Overview

`test_ubunchu_annotations` crops 9 speech bubbles from `assets/ubunchu01_02.png`
(1246x1635 grayscale) using bounding boxes defined in `assets/ubunchu01_02.json`,
runs each crop through the manga-ocr model, and compares the normalised output
against ground truth.

CI result: **4 passed, 5 failed** out of 9.

## Scale factors

The JSON uses Gemini-normalised [0-1000] coordinates. The test converts to
pixels via:

    sx = 1246 / 1000 = 1.246
    sy = 1635 / 1000 = 1.635

## Per-annotation breakdown

### 1. Top right bubble, line 1 -- PASS

- box_2d: `[75, 804, 216, 951]`
- pixels: x1=1002, y1=123, x2=1185, y2=353 -> 183x230 crop
- expected: `ああたしのオススメは` (normalised)
- got: `ああたしのオススメは`
- Notes: Clean vertical text in a speech bubble. No issues.

### 2. Top right bubble, large text -- FAIL

- box_2d: `[140, 686, 310, 831]`
- pixels: x1=855, y1=229, x2=1035, y2=507 -> 180x278 crop
- expected: `うぶんちゅ`
- got: `メはうぶんちゅ`
- **Root cause: bounding box overlap.** The crop captures the bottom of
  annotation #1's text ("オスス**メは**") in addition to the target "うぶんちゅ".
  The box y-ranges overlap: #1 ends at y=216, #2 starts at y=140.
- **Fix options:**
  - (a) Tighten box_2d[0] (y1) for this annotation from 140 to ~220 to exclude
    the overlapping text.
  - (b) Accept prefix in the expected text: `メはうぶんちゅ`. (Not ideal --
    tests should reflect actual ground truth.)
  - (c) Use a fuzzy/contains match for annotations where bounding boxes are
    known to bleed.

### 3. Top left bubble -- PASS

- box_2d: `[145, 95, 345, 252]`
- pixels: x1=118, y1=237, x2=314, y2=564 -> 196x327 crop
- expected: `最近人気のデスクトップなリナックスです!`
- got: `最近人気のデスクトップなリナックスです!`
- Notes: Clean vertical text. No issues.

### 4. Center caption (tiny text) -- FAIL

- box_2d: `[386, 344, 408, 654]`
- pixels: x1=429, y1=631, x2=815, y2=667 -> 386x36 crop (!)
- expected: `※うぶんちゅではなくウブントゥです`
- got: long hallucinated garbage
- **Root cause: crop is too small.** The crop is only 36 pixels tall. After
  resizing to 224x224 the characters are severely degraded, causing the decoder
  to hallucinate.
- **Fix options:**
  - (a) Add vertical padding to the bounding box (e.g. expand to 50-60px tall)
    so characters survive the resize.
  - (b) Mark this annotation as a known failure and skip it (test infra could
    support an `"expect_fail": true` field).
  - (c) In the OCR pipeline, add minimum-dimension padding before resize.

### 5. Middle bubble (rejection) -- PASS

- box_2d: `[431, 328, 542, 501]`
- pixels: x1=409, y1=705, x2=624, y2=886 -> 215x181 crop
- expected: `却下!`
- got: `却下!`
- Notes: Large, bold text. No issues.

### 6. Bottom center, aggressive text -- FAIL

- box_2d: `[632, 431, 712, 563]`
- pixels: x1=537, y1=1033, x2=701, y2=1164 -> 164x131 crop
- expected: `マジいてえんだぞ!`
- got: long hallucinated garbage
- **Root cause: decorative action text with speed lines.** The text is rendered
  in a bold/slanted style with motion-line artwork behind it. The model
  (trained on clean scanned manga speech bubbles) hallucinates when given noisy
  visual input.
- **Fix options:**
  - (a) Mark as known failure / skip.
  - (b) Pre-process crop: increase contrast, threshold to isolate text from
    speed-line art.
  - (c) Accept this as a model limitation for this class of text.

### 7. Bottom right, screaming -- FAIL

- box_2d: `[629, 742, 730, 915]`
- pixels: x1=925, y1=1028, x2=1140, y2=1194 -> 215x166 crop
- expected: `よけんなこのっ!`
- got: long hallucinated garbage
- **Root cause: same as #6.** Large decorative text with visual effects. The
  crop includes heavy screentone/action-line art that confuses the model.
- **Fix options:** Same as #6.

### 8. Bottom left bubble, top part -- FAIL

- box_2d: `[631, 41, 765, 239]`
- pixels: x1=51, y1=1032, x2=298, y2=1251 -> 247x219 crop
- expected: `ハモリながらケンカしないでーっ`
- got: `ハモリながらケアカしないでーっ`
- **Root cause: minor character misread.** `ン` misread as `ア` (visually
  similar at small size). The crop size is reasonable (247x219) so this may
  be a model accuracy limitation at this font/size, or the bounding box
  clips slightly.
- **Fix options:**
  - (a) Slightly adjust the bounding box to ensure full character visibility.
  - (b) Accept as a model limitation -- the output is close but not exact.
  - (c) Use a fuzzy match (edit-distance threshold) for this annotation.

### 9. Bottom left bubble, bottom part -- PASS

- box_2d: `[805, 75, 929, 169]`
- pixels: x1=93, y1=1316, x2=211, y2=1519 -> 118x203 crop
- expected: `一瞬くらい検討してくださいよー!`
- got: `一瞬くらい検討してくださいよー!`
- Notes: Clean vertical text. No issues.

## Failure summary

| # | Description | Category | Severity |
|---|-------------|----------|----------|
| 2 | Large text "うぶんちゅ" | Bounding box overlap | Easy fix -- tighten y1 |
| 4 | Center caption | Tiny crop (36px tall) | Hard -- model limitation |
| 6 | Aggressive text | Decorative/action text | Hard -- model limitation |
| 7 | Screaming text | Decorative/action text | Hard -- model limitation |
| 8 | "ケンカしないで" | Minor misread (ン→ア) | Medium -- bbox or model |

## Recommended approach

**Short term (unblock CI):**

1. Fix annotation #2 by tightening its y1 from 140 to ~220.
2. For #4, #6, #7: these are known model limitations (tiny text, decorative
   art). Options:
   - Add an `"expect_fail"` or `"skip"` flag to the JSON and have the test
     tolerate those.
   - Or loosen the assertion: report failures but don't fail the test (print
     a summary, assert only on annotations without known issues).
3. For #8: try tightening the bounding box first. If it still misreads,
   treat as a known model limitation.

**Long term:**

- Investigate pre-processing (contrast normalisation, binarisation) to improve
  robustness on noisy crops.
- Consider a "confidence" or "difficulty" field per annotation to set
  per-annotation pass criteria.
- The model (manga-ocr) was trained on clean speech-bubble crops; action text
  and tiny captions are out-of-distribution inputs.
