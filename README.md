# Audio Deepfake Detector

This project is an audio deepfake detector built around a fine-tuned Wav2Vec2 model, `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`. It pairs a Flask backend with a static HTML frontend so you can upload an audio file, run it through the model, and get back a JSON result indicating whether the audio is likely fake along with a confidence score.

## Credit and Attribution

This project is based on [mrunalkanpillewar/deepfake_audio_detection](https://github.com/mrunalkanpillewar/deepfake_audio_detection), originally created by `mrunalkanpillewar`.

Modifications in this version include fixed CPU-only Torch dependencies, removal of dead code, added input validation, and a Windows encoding crash fix.

I checked the upstream repository for a `LICENSE` file at the standard `main` and `master` paths and did not find one, so no new license file has been added here.

## Architecture

The frontend HTML uploads an audio file to the Flask backend using `POST /analyze`. The backend resamples the audio to 16 kHz mono, feeds it into the Wav2Vec2 model, and returns JSON with `is_fake` and `confidence`.

## Setup and Installation

```powershell
git clone <your-repo-url>
cd deepfake_audio_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/app.py
```

Then open `frontend/"deepfake audio detector.html"` in a browser.

## API Reference

`GET /health`

Returns:

```json
{"status": "healthy", "model_loaded": true}
```

`POST /analyze`

Accepts multipart form-data with an `audio` field. Allowed extensions are `wav`, `flac`, `mp3`, `ogg`, and `m4a`.

Returns:

```json
{"is_fake": true, "confidence": 0.98}
```

## Known Limitations

- Flask dev server only; it is not production-ready as-is.
- `mp3` support depends on system `ffmpeg` being installed.
- `.opus` files are not in the allowed extensions list.
- The first run downloads the model from Hugging Face, which is roughly 1 GB or more and can take time.
- Windows users may see degraded Hugging Face cache symlink behavior unless Developer Mode is enabled.
