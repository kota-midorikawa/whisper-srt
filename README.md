# whisper-srt
Audio/Video → SubRip (.srt) & time-tagged .txt via faster-whisper.

## Usage
pip install faster-whisper
# ffmpeg must be in PATH

python src/transcribe_srt.py --media sample.mp4 --device auto --language ja
