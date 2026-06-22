import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa

# --- Import NEW model utilities only ---
try:
    from detector_new import load_model, detect_deepfake
except ImportError:
    print("Error: detector_new.py not found or import failed. Make sure it's in the same folder.")
    sys.exit(1)

app = Flask(__name__)

# Optional: limit upload size to ~25 MB
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Load the AI model once at startup ---
print("Loading AI model... This may take a moment.")
try:
    model, processor = load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    sys.exit(1)


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    print("Received a new audio file for analysis...")

    # 1) Validate file presence
    if 'audio' not in request.files:
        print("Error: No 'audio' file part in the request.")
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']

    # 2) Validate filename
    if file.filename == '':
        print("Error: File has no name.")
        return jsonify({"error": "No selected file"}), 400

    try:
        # 3) Decode audio from in-memory stream; resample to 16 kHz mono
        print(f"Processing audio file: {file.filename}")
        audio_data, sample_rate = librosa.load(file.stream, sr=16000, mono=True)

        # 4) Run inference with NEW model
        print("Analyzing with NEW model...")
        prediction = detect_deepfake(audio_data, sample_rate, processor, model)

        print(f"Analysis complete. Result: {prediction}")

        # 5) Return JSON - same basic format your frontend expects
        return jsonify({
            "is_fake": prediction["is_fake"],
            "confidence": float(prediction["confidence"])
        }), 200

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True}), 200


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("AudioSleuth Backend Server Starting (NEW model only)...")
    print("Server running at: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
