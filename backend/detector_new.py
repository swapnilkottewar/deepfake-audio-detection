import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# New model specifically for spoof / deepfake detection
MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
DEVICE = torch.device("cpu")  # keep CPU for stability

print(f"Using device: {DEVICE}")
print(f"Using model : {MODEL_ID}")


def load_model():
    """
    Load model and feature extractor for spoof/deepfake detection (NEW model).
    """
    print(f"Loading model: {MODEL_ID}")
    print(f"Using device: {DEVICE}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID,
        return_attention_mask=True
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID
    ).to(DEVICE)
    model.eval()

    print(f"✓ New model loaded. Labels: {model.config.id2label}")
    return model, feature_extractor


def _ensure_16k_mono_float(audio, sr):
    """
    Ensure 16kHz, mono, float32 numpy array.
    """
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    # Stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio, sr


def detect_deepfake(audio_data, sample_rate, feature_extractor, model):
    """
    Analyze one audio clip with the NEW model.

    Returns:
        dict: {
            "is_fake": bool,
            "confidence": float,
            "fake_score": float,
            "real_score": float
        }
    """
    # 1) Standardize audio
    audio_data, sr = _ensure_16k_mono_float(audio_data, sample_rate)

    if len(audio_data) < 1600:  # < 0.1s
        raise ValueError("Audio file is too short (minimum 0.1 seconds required)")

    # 2) Extract features
    inputs = feature_extractor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
        return_attention_mask=True
    )
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None

    # 3) Forward pass
    with torch.inference_mode():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits  # [1, num_labels]

    # 4) Probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)

    # 5) Map labels using config (try to find real/spoof)
    id2label = getattr(model.config, "id2label", {0: "fake", 1: "real"})
    norm_map = {int(i): str(l).lower() for i, l in id2label.items()}

    real_idx = next((i for i, l in norm_map.items() if "real" in l or "bonafide" in l), 1)
    fake_idx = next((i for i, l in norm_map.items() if "fake" in l or "spoof" in l), 0)

    real_score = float(probs[real_idx].item())
    fake_score = float(probs[fake_idx].item())

    # 6) Decision with thresholds and uncertainty
    TH_STRONG = 0.8  # strong confidence threshold

    if fake_score >= TH_STRONG and fake_score > real_score:
        is_fake = True
        confidence = fake_score
        decision_type = "FAKE_STRONG"
    elif real_score >= TH_STRONG and real_score > fake_score:
        is_fake = False
        confidence = real_score
        decision_type = "REAL_STRONG"
    else:
        # Uncertain zone: scores are close or below threshold
        is_fake = fake_score > real_score
        confidence = max(fake_score, real_score) * 0.5  # lower confidence
        decision_type = "UNCERTAIN"

    print(
        f"[NEW model] REAL={real_score:.3f}, FAKE={fake_score:.3f}, "
        f"decision={'FAKE' if is_fake else 'REAL'} ({decision_type}), "
        f"confidence={confidence:.3f}"
    )

    return {
        "is_fake": is_fake,
        "confidence": confidence,
        "fake_score": fake_score,
        "real_score": real_score
    }
