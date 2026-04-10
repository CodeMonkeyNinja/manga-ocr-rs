/// Build script: download manga-ocr ONNX models on first build.
///
/// Models (~441 MB total) are cached in `$MANGA_OCR_MODELS_DIR` if set,
/// otherwise `~/.cache/manga-ocr-rs/`.  Set that env var before `cargo build`
/// to point at an already-downloaded copy and skip the network entirely.
///
/// The resolved path is emitted as `MANGA_OCR_DEFAULT_MODEL_DIR` so library
/// code and tests can call `env!("MANGA_OCR_DEFAULT_MODEL_DIR")` to get the
/// default location without hard-coding it.
use std::path::PathBuf;
use std::process::Command;

const HF_BASE: &str =
    "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/main";
const MODEL_FILES: &[&str] = &["encoder_model.onnx", "decoder_model.onnx", "vocab.txt"];

fn main() {
    // Re-run this script whenever the override env var changes.
    println!("cargo:rerun-if-env-changed=MANGA_OCR_MODELS_DIR");

    let models_dir = if let Ok(dir) = std::env::var("MANGA_OCR_MODELS_DIR") {
        PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".cache").join("manga-ocr-rs")
    };

    let all_present = MODEL_FILES.iter().all(|f| models_dir.join(f).exists());

    if !all_present {
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            println!("cargo:warning=manga-ocr-rs: could not create {}: {e}", models_dir.display());
        } else {
            println!(
                "cargo:warning=manga-ocr-rs: downloading models to {} (~441 MB) …",
                models_dir.display()
            );
            for file in MODEL_FILES {
                let dest = models_dir.join(file);
                if dest.exists() {
                    continue;
                }
                let url = format!("{HF_BASE}/{file}");
                println!("cargo:warning=manga-ocr-rs:   {file}");
                match Command::new("curl").args(["-fL", &url, "-o"]).arg(&dest).status() {
                    Ok(s) if s.success() => {}
                    Ok(s) => println!("cargo:warning=manga-ocr-rs:   curl exited {s} for {file}"),
                    Err(e) => println!("cargo:warning=manga-ocr-rs:   curl failed: {e}"),
                }
            }
        }
    }

    // Always emit the resolved path so env!("MANGA_OCR_DEFAULT_MODEL_DIR") works.
    println!(
        "cargo:rustc-env=MANGA_OCR_DEFAULT_MODEL_DIR={}",
        models_dir.display()
    );
}
