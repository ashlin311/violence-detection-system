import os
import time
import tempfile
from pathlib import Path
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

from model import run_inference
from utils import encode_image_to_base64
from gradcam import generate_gradcam

app = Flask(__name__)
CORS(app)

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _get_temp_suffix(filename):
    suffix = Path(filename).suffix.lower() if filename else ""
    return suffix if suffix in ALLOWED_VIDEO_SUFFIXES else ".mp4"


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "violence-detection-backend",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return "", 204

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=_get_temp_suffix(video_file.filename))
    os.close(tmp_fd)

    try:
        video_file.save(tmp_path)

        start_time = time.time()
        result = run_inference(tmp_path)
        processing_time = time.time() - start_time

        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "most_active_frame": result["most_active_frame"],
            "person_count": result["person_count"],
            "processing_time": processing_time,
            "original_frame": encode_image_to_base64(result["original_frame"]),
            "masked_frame": encode_image_to_base64(result["masked_frame"]),
            "gradcam_frame": None,
        }
        return jsonify(response)

    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e), "error_type": type(e).__name__}), 400
    except Exception as e:
        message = str(e)
        lower_message = message.lower()
        if any(token in lower_message for token in ("video", "frame", "decode", "codec", "open")):
            return jsonify({"error": message, "error_type": type(e).__name__}), 400

        traceback.print_exc()
        return jsonify({"error": message, "error_type": type(e).__name__}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    # Use 5001 by default to avoid collisions with local tooling on 5000.
    port = int(os.getenv("PORT", "5001"))
    app.run(host='0.0.0.0', port=port, debug=True)
