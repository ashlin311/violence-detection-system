import os
import time
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

from model import run_inference
from utils import encode_image_to_base64

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)