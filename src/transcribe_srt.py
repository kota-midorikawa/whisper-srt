import argparse, shutil, sys
from pathlib import Path
from faster_whisper import WhisperModel

def ts(t: float) -> str:
    ms_total = int(round(max(0.0, t) * 1000))
    s, ms = divmod(ms_total, 1000); h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def load_model(device: str, model_name: str):
    if device == "cuda":
        return WhisperModel(model_name, device="cuda", compute_type="float16")
    # CPU
    return WhisperModel(model_name, device="cpu", compute_type="int8")

def main():
    p = argparse.ArgumentParser(description="Transcribe media to .srt using faster-whisper")
    p.add_argument("--media", required=True, help="input media file (audio/video)")
    p.add_argument("--language", default="ja")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--model", default="large-v3", help="e.g., tiny, base, small, medium, large-v3")
    p.add_argument("--no-vad", action="store_true", help="disable VAD")
    args = p.parse_args()

    assert shutil.which("ffmpeg"), "ffmpeg not found in PATH"
    media = Path(args.media); assert media.is_file(), f"Media not found: {media}"

    # device auto → try CUDA then CPU
    dev = args.device
    try:
        if dev == "auto":
            try:
                model = load_model("cuda", args.model)
            except Exception:
                print("[info] no CUDA; falling back to CPU int8", file=sys.stderr)
                model = load_model("cpu", args.model)
        else:
            model = load_model(dev, args.model)
    except Exception as e:
        # cuDNN等で失敗 → CPUへ
        if "cudnn" in str(e).lower():
            print("[info] cuDNN error; retry on CPU int8", file=sys.stderr)
            model = load_model("cpu", args.model)
        else:
            raise

    segments, info = model.transcribe(
        str(media),
        language=args.language,
        vad_filter=(not args.no_vad),
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=5,
    )

    base = media.stem
    with open(f"{base}.txt","w",encoding="utf-8") as f_txt, open(f"{base}.srt","w",encoding="utf-8") as f_srt:
        for i, seg in enumerate(segments, 1):
            text = (seg.text or "").replace("\n"," ").strip()
            f_txt.write(f"[{ts(seg.start)}] {text}\n")
            f_srt.write(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{text}\n\n")

if __name__ == "__main__":
    main()
