use anyhow::{Context, Result};
use manga_ocr_rs::MangaOcr;
use ort::session::Session;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(String::as_str) == Some("inspect") {
        let model_dir = args.get(2).map(String::as_str)
            .unwrap_or_else(|| manga_ocr_rs::default_model_dir().to_str().unwrap());
        if let Err(e) = inspect(model_dir) {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
        return;
    }

    if args.len() < 2 {
        let default = manga_ocr_rs::default_model_dir().display();
        eprintln!("Usage: {} <image> [model_dir]", args[0]);
        eprintln!("       {} inspect [model_dir]", args[0]);
        eprintln!("       model_dir defaults to {default}");
        std::process::exit(1);
    }

    let model_dir = args.get(2).map(String::as_str)
        .unwrap_or_else(|| manga_ocr_rs::default_model_dir().to_str().unwrap());

    if let Err(e) = run(&args[1], model_dir) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn inspect(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);
    for (label, filename) in [
        ("encoder", "encoder_model.onnx"),
        ("decoder", "decoder_model.onnx"),
    ] {
        let path = dir.join(filename);
        let session = Session::builder()
            .context("SessionBuilder")?
            .commit_from_file(&path)
            .with_context(|| format!("open {}", path.display()))?;

        println!("── {label} ({filename}) ──");
        print!("  inputs : ");
        println!("{}", session.inputs().iter().map(|i| i.name().to_string()).collect::<Vec<_>>().join(", "));
        print!("  outputs: ");
        println!("{}", session.outputs().iter().map(|o| o.name().to_string()).collect::<Vec<_>>().join(", "));
    }
    Ok(())
}

fn run(image_path: &str, model_dir: &str) -> Result<()> {
    let model_dir = Path::new(model_dir);
    for f in ["encoder_model.onnx", "decoder_model.onnx", "vocab.txt"] {
        let p = model_dir.join(f);
        if !p.exists() {
            anyhow::bail!(
                "{} not found — run `cargo build` to download, or set MANGA_OCR_MODELS_DIR",
                p.display()
            );
        }
    }

    let img = image::open(image_path)
        .with_context(|| format!("open image: {image_path}"))?;
    println!("image : {image_path}  ({}×{})", img.width(), img.height());

    let ocr = MangaOcr::new(model_dir).context("load models")?;

    let t = Instant::now();
    let text = ocr.recognize(&img).context("recognize")?;
    println!("time  : {:?}", t.elapsed());
    println!("text  : {text}");
    Ok(())
}
