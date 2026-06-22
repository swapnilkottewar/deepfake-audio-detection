import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Use a verified, public model id
MODEL_ID = "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"
DEVICE = torch.device("cpu")  # Force CPU to avoid CUDA errors

print(f"Using device: {DEVICE}")


def load_model():
    """
    Loads the model and feature extractor for deepfake detection.
    Uses safetensors format to avoid PyTorch security vulnerability.
    """
    print(f"Loading model: {MODEL_ID}")
    print(f"Using device: {DEVICE}")
    
    try:
        # Try loading with safetensors first (secure method)
        print("Attempting to load with safetensors format...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_ID,
            return_attention_mask=True,
            use_safetensors=True
        )
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_ID,
            use_safetensors=True
        ).to(DEVICE)
        print("✓ Model loaded successfully with safetensors!")
        
    except Exception as e:
        print(f"Safetensors loading failed: {e}")
        print("Attempting alternative loading method...")
        
        # Fallback: Try without specifying safetensors
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                MODEL_ID,
                return_attention_mask=True
            )
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                MODEL_ID,
                trust_remote_code=False
            ).to(DEVICE)
            print("✓ Model loaded successfully with alternative method!")
            
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            raise RuntimeError(
                f"Could not load model {MODEL_ID}. "
                f"Please ensure: 1) PyTorch and transformers are up to date, "
                f"2) Model exists on HuggingFace"
            )
    
    model.eval()
    print(f"Model configuration: {model.config.num_labels} labels")
    return model, feature_extractor


def _ensure_16k_mono_float(audio, sr):
    """
    Ensures 16kHz, mono, float32 numpy array.
    """
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    
    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure float32 dtype
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    return audio, sr


def detect_deepfake(audio_data, sample_rate, feature_extractor, model):
    """
    Analyzes a single audio clip and returns a prediction.
    
    Args:
        audio_data: 1D or 2D array-like audio samples
        sample_rate: int, actual sampling rate of audio_data
        feature_extractor: Wav2Vec2FeatureExtractor instance
        model: Wav2Vec2ForSequenceClassification instance
    
    Returns:
        dict: {
            "is_fake": bool,
            "confidence": float,
            "fake_score": float,
            "real_score": float
        }
    """
    try:
        # 1) Standardize audio
        audio_data, _ = _ensure_16k_mono_float(audio_data, sample_rate)
        
        # Check if audio is empty or too short
        if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
            raise ValueError("Audio file is too short (minimum 0.1 seconds required)")
        
        # 2) Extract inputs; no padding needed for single sample
        inputs = feature_extractor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True
        )
        input_values = inputs.input_values.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None
        
        # 3) Forward pass with inference mode
        with torch.inference_mode():
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits  # shape [1, num_labels]
        
        # 4) Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)  # [num_labels]
        
        # 5) Map indices using model config
        id2label = getattr(model.config, "id2label", {0: "LABEL_0", 1: "LABEL_1"})
        norm_map = {i: str(l).lower() for i, l in id2label.items()}
        
        # Find the correct indices for real and fake
        real_idx = next((i for i, l in norm_map.items() if l in ["bonafide", "real"]), 0)
        fake_idx = next((i for i, l in norm_map.items() if l in ["spoof", "fake"]), 1)
        
        real_score = float(probs[real_idx].item())
        fake_score = float(probs[fake_idx].item())
        
        # 6) Final result
        if fake_score > real_score:
            is_fake = True
            confidence = fake_score
        else:
            is_fake = False
            confidence = real_score
        
        print(
            f"Detection result: {'FAKE' if is_fake else 'REAL'} "
            f"(real={real_score:.3f}, fake={fake_score:.3f})"
        )
        
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "fake_score": fake_score,
            "real_score": real_score
        }
        
    except Exception as e:
        print(f"Error during deepfake detection: {e}")
        raise
