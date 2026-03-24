# Violence Detection System

An AI-powered video analysis system that detects violent activities in real-time using deep learning. The system combines temporal segmentation networks (TSN) with human detection and explainability features (Grad-CAM) to provide accurate violence detection with visual analysis.

## Features

- **AI Violence Detection** – ResNet18-based TSN model for real-time violence classification
- **Video Analysis** – Process uploaded videos and live camera feeds
- **Live Camera Monitoring** – Record and analyze from webcam with 8-second capture
- **Human Detection** – YOLOv8n-based person detection and masking
- **Frame Analysis** – View the most violent frame with original and masked versions
- **Explainability** – Grad-CAM heatmap visualization showing model attention areas
- **Full-Screen Image Viewer** – Inspect frames at full resolution

## Installation

### Requirements
- Python 3.8+
- GPU (CUDA-compatible) optional; CPU supported

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashlin311/violence-detection-system.git
   cd violence-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model exists**
   - Download or place `model/tsn_masked_best.pth` in the model directory

## Usage

### Starting the Backend
```bash
python app.py
```
- Runs on `http://localhost:5001`
- Health check: `GET /localhost:5001/health`
- Prediction: `POST /localhost:5001/predict` (multipart form with video file)

### Starting the Frontend
```bash
cd client
python -m http.server 5500
```
- Opens at `http://localhost:5500`

### Using the Application

1. **Upload Video**
   - Click "Upload Video" button
   - Select a video file (.mp4, .avi, .mov, .mkv, .webm, .m4v)
   - Wait for analysis to complete
   - View results: prediction, confidence, person count, frame visualizations

2. **Live Camera**
   - Click "Start Camera Scan"
   - Allow camera permissions
   - System records 8 seconds automatically
   - Results displayed after processing

3. **Inspect Frames**
   - Click any analysis frame (original, masked, or Grad-CAM)
   - View full-resolution image with × button to close

## API Documentation

### POST /predict
Analyze a video file and return violence detection results.

**Request:**
```
Content-Type: multipart/form-data
Body: {video: <file>}
```

**Response:**
```json
{
  "prediction": "Violent|Non Violent",
  "confidence": 0.95,
  "most_active_frame": 3,
  "person_count": 2,
  "processing_time": 2.34,
  "original_frame": "<base64 JPEG>",
  "masked_frame": "<base64 JPEG>",
  "gradcam_frame": "<base64 JPEG or null>"
}
```

### GET /health
Check backend status.

**Response:**
```json
{"status": "ok"}
```

## File Structure

```
violence-detection-system/
├── app.py                 # Flask backend & API endpoints
├── model.py              # TSN model & inference pipeline
├── masking.py            # YOLOv8 human detection & masking
├── gradcam.py            # Grad-CAM heatmap generation
├── utils.py              # Helper functions (frame extraction, tensor conversion)
├── requirements.txt      # Python dependencies
├── model/
│   └── tsn_masked_best.pth  # Trained TSN model weights
├── client/
│   ├── index.html        # Frontend UI
│   ├── app.js            # JavaScript logic & API client
│   ├── style.css         # Styling
│   └── favicon.ico
├── notebooks/
│   └── *.ipynb           # Training & experiment notebooks
└── README.md
```

## Key Components

### Model Training (model.py)
- Loads pretrained ResNet18 weights (optional)
- Fine-tuned on masked video frames
- Outputs 2-class prediction (Non Violent / Violent)

### Frame Extraction (utils.py)
- TSN sampling: evenly divides video into 8 segments, picks middle frame per segment (validation mode)
- Resizes frames to 224×224
- Normalizes with ImageNet statistics

### Human Masking (masking.py)
- YOLOv8 nano model for real-time person detection
- Dims non-person regions (0.3 brightness factor)
- Highlights detected persons at full brightness
- Returns masked frame and person count

### Grad-CAM (gradcam.py)
- Backpropagates gradients through ResNet18 Layer4
- Computes attention weights across 512 feature channels
- Upsamples 7×7 activation map to frame size
- Overlays jet colormap for visualization

## Model Performance

- **Classes**: 2 (Non Violent, Violent)
- **Input Resolution**: 224×224
- **Processing Time**: ~2-3 seconds per video (depends on duration and hardware)
- **GPU Memory**: ~1.5 GB

## Requirements

```
flask==2.3.0
flask-cors==4.0.0
torch==2.0.0
torchvision==0.15.0
opencv-python==4.7.0
ultralytics==8.0.0
numpy==1.24.0
```

See `requirements.txt` for exact versions.