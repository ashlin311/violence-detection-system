const API_BASE_URL = window.location.hostname
  ? `${window.location.protocol}//${window.location.hostname}:5001`
  : "http://127.0.0.1:5001";

const CAMERA_COUNTDOWN_SECONDS = 3;
const CAMERA_RECORD_SECONDS = 8;

let activeCameraStream = null;
let isCameraBusy = false;

function openOverlay(id) {
  document.getElementById(id).style.display = "flex";
}

function closeOverlay() {
  document.querySelectorAll(".overlay").forEach((overlay) => {
    overlay.style.display = "none";
    overlay.classList.remove("analysis-overlay--loading");
  });
}

function openUpload() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "video/*";

  input.onchange = () => {
    const file = input.files && input.files[0];
    if (file) {
      analyzeVideo(file);
    }
  };

  input.click();
}

function getAnalysisElements() {
  return {
    overlay: document.getElementById("analysis"),
    loading: document.getElementById("loading"),
    result: document.getElementById("result"),
    prediction: document.getElementById("prediction"),
    statusPill: document.getElementById("analysisStatusPill"),
    confidence: document.getElementById("confidence"),
    people: document.getElementById("people"),
    mostActiveFrame: document.getElementById("mostActiveFrame"),
    frameIndexCard: document.getElementById("frameIndexCard"),
    mostActiveFrameImage: document.getElementById("mostActiveFrameImage"),
    maskedFrameImage: document.getElementById("maskedFrameImage"),
    gradcamFrameImage: document.getElementById("gradcamImage"),
    gradcamSlider: document.getElementById("gradcamSlider"),
  };
}

function showAnalysisLoading() {
  const elements = getAnalysisElements();
  openOverlay("analysis");
  elements.overlay.classList.add("analysis-overlay--loading");
  elements.loading.classList.remove("is-hidden");
  elements.result.classList.add("is-hidden");
}

function showResult(data) {
  const elements = getAnalysisElements();
  elements.overlay.classList.remove("analysis-overlay--loading");
  elements.loading.classList.add("is-hidden");
  elements.result.classList.remove("is-hidden");

  const confidenceRaw = Number(data.confidence);
  const confidencePercent = Number.isFinite(confidenceRaw) ? confidenceRaw * 100 : 0;
  const confidenceValue = confidencePercent.toFixed(2);
  const predictionText = data.prediction || "Unknown";
  const predictionLower = predictionText.toLowerCase();
  elements.prediction.innerText = predictionText;
  elements.confidence.innerText = `${confidenceValue}%`;
  elements.people.innerText = String(data.person_count ?? 0);
  elements.mostActiveFrame.innerText = String(data.most_active_frame ?? "-");
  elements.frameIndexCard.innerText = String(data.most_active_frame ?? "-");
  elements.gradcamSlider.value = Math.max(0, Math.min(100, Math.round(confidencePercent)));

  elements.statusPill.classList.remove("analysis-status-pill--safe", "analysis-status-pill--alert");
  if (predictionLower.includes("non")) {
    elements.statusPill.classList.add("analysis-status-pill--safe");
  } else if (predictionLower.includes("viol")) {
    elements.statusPill.classList.add("analysis-status-pill--alert");
  }
  elements.mostActiveFrameImage.src = `data:image/jpeg;base64,${data.original_frame}`;
  elements.maskedFrameImage.src = `data:image/jpeg;base64,${data.masked_frame}`;
  if (data.gradcam_frame) {
    elements.gradcamFrameImage.src = `data:image/jpeg;base64,${data.gradcam_frame}`;
  } else {
    elements.gradcamFrameImage.removeAttribute("src");
  }

  // Add click listeners to analysis frame images
  elements.mostActiveFrameImage.style.cursor = "pointer";
  elements.mostActiveFrameImage.onclick = () => openFullscreenImage(elements.mostActiveFrameImage.src);

  elements.maskedFrameImage.style.cursor = "pointer";
  elements.maskedFrameImage.onclick = () => openFullscreenImage(elements.maskedFrameImage.src);

  elements.gradcamFrameImage.style.cursor = "pointer";
  elements.gradcamFrameImage.onclick = () => openFullscreenImage(elements.gradcamFrameImage.src);
}

function showAnalysisError(message) {
  const elements = getAnalysisElements();
  elements.overlay.classList.remove("analysis-overlay--loading");
  elements.loading.classList.add("is-hidden");
  elements.result.classList.remove("is-hidden");

  elements.prediction.innerText = "Error";
  elements.statusPill.classList.remove("analysis-status-pill--safe");
  elements.statusPill.classList.add("analysis-status-pill--alert");
  elements.confidence.innerText = "--";
  elements.people.innerText = "--";
  elements.mostActiveFrame.innerText = "--";
  elements.frameIndexCard.innerText = "--";
  elements.gradcamSlider.value = 0;
  elements.mostActiveFrameImage.removeAttribute("src");
  elements.maskedFrameImage.removeAttribute("src");
  elements.gradcamFrameImage.removeAttribute("src");
  console.error("Analysis error:", message);
}

async function analyzeVideo(file) {
  showAnalysisLoading();

  const form = new FormData();
  form.append("video", file);

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: form,
    });

    let payload = {};
    try {
      payload = await res.json();
    } catch (parseErr) {
      payload = {};
    }

    if (!res.ok) {
      console.error("Predict API error", {
        status: res.status,
        payload,
      });
      throw new Error(payload.error || `Request failed with status ${res.status}`);
    }

    showResult(payload);
    return true;
  } catch (err) {
    showAnalysisError(err.message || "Backend connection error");
    return false;
  }
}

function openFullscreenImage(src) {
  const overlay = document.getElementById("fullscreenImageOverlay");
  const fullscreenImage = document.getElementById("fullscreenImage");
  if (src && src !== "") {
    fullscreenImage.src = src;
    overlay.style.display = "flex";
  }
}

function closeFullscreenImage() {
  const overlay = document.getElementById("fullscreenImageOverlay");
  overlay.style.display = "none";
}

function openCameraRecordOverlay() {
  const overlay = document.getElementById("cameraRecordOverlay");
  if (overlay) {
    overlay.style.display = "flex";
  }
}

function closeCameraRecordOverlay() {
  const overlay = document.getElementById("cameraRecordOverlay");
  if (overlay) {
    overlay.style.display = "none";
  }
}

function getCameraElements() {
  return {
    preview: document.getElementById("cameraPreview"),
    video: document.getElementById("cameraVideo"),
    overlay: document.getElementById("cameraOverlay"),
    status: document.getElementById("cameraStatus"),
    recordVideo: document.getElementById("cameraRecordVideo"),
    recordOverlay: document.getElementById("cameraRecordLayer"),
    recordStatus: document.getElementById("cameraRecordStatus"),
    button: document.getElementById("cameraStartButton"),
  };
}

function getCameraIdleText() {
  const { status } = getCameraElements();
  return status.dataset.idleText || "Press Start Camera Scan to record and analyze.";
}

function setCameraStatus(text) {
  const { status, recordStatus } = getCameraElements();
  if (status) {
    status.innerText = text;
  }
  if (recordStatus) {
    recordStatus.innerText = text;
  }
}

function setCameraOverlay(text, isCountdown = false, isRecording = false) {
  const { overlay, recordOverlay } = getCameraElements();
  [overlay, recordOverlay].forEach((target) => {
    if (!target) return;
    target.classList.remove("is-hidden");
    target.classList.toggle("camera-overlay--countdown", isCountdown);
    target.classList.toggle("camera-overlay--recording", isRecording);
    target.innerText = text;
  });
}

function setCameraBusy(isBusy) {
  const { button } = getCameraElements();
  button.disabled = isBusy;
  button.innerText = isBusy ? "Scanning..." : "Start Camera Scan";
  isCameraBusy = isBusy;
}

function stopCameraStream() {
  if (activeCameraStream) {
    activeCameraStream.getTracks().forEach((track) => track.stop());
    activeCameraStream = null;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runCameraCountdown(seconds) {
  for (let current = seconds; current >= 1; current -= 1) {
    setCameraOverlay(String(current), true);
    setCameraStatus(`Recording starts in ${current}...`);
    await sleep(1000);
  }
}

function pickRecordingMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];

  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }

  return "";
}

function extensionFromMimeType(mimeType) {
  if (mimeType.includes("mp4")) return "mp4";
  return "webm";
}

function recordForDuration(stream, durationMs, mimeType, onTick) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let recorder;

    try {
      recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
    } catch (err) {
      reject(err);
      return;
    }

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    recorder.onerror = (event) => {
      reject(event.error || new Error("Recording failed"));
    };

    recorder.onstop = () => {
      const finalType = recorder.mimeType || mimeType || "video/webm";
      resolve(new Blob(chunks, { type: finalType }));
    };

    const totalSeconds = Math.ceil(durationMs / 1000);
    let remainingSeconds = totalSeconds;
    onTick(remainingSeconds);

    const interval = setInterval(() => {
      remainingSeconds -= 1;
      if (remainingSeconds > 0) {
        onTick(remainingSeconds);
      }
    }, 1000);

    recorder.start();

    setTimeout(() => {
      clearInterval(interval);
      if (recorder.state !== "inactive") {
        recorder.stop();
      }
    }, durationMs);
  });
}

async function openCamera() {
  if (isCameraBusy) {
    return;
  }

  if (
    !navigator.mediaDevices ||
    !navigator.mediaDevices.getUserMedia ||
    typeof MediaRecorder === "undefined"
  ) {
    alert("Camera recording is not supported on this browser.");
    return;
  }

  const { video, recordVideo } = getCameraElements();
  setCameraBusy(true);
  openCameraRecordOverlay();

  try {
    setCameraStatus("Requesting camera permission...");
    setCameraOverlay("Allow camera access");

    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    activeCameraStream = stream;

    if (video) {
      video.classList.add("is-hidden");
    }

    recordVideo.srcObject = stream;
    recordVideo.classList.remove("is-hidden");
    await recordVideo.play();

    await runCameraCountdown(CAMERA_COUNTDOWN_SECONDS);

    setCameraOverlay("Recording", false, true);
    const mimeType = pickRecordingMimeType();
    const blob = await recordForDuration(
      stream,
      CAMERA_RECORD_SECONDS * 1000,
      mimeType,
      (remainingSeconds) => {
        setCameraStatus(`Recording... ${remainingSeconds}s left`);
      }
    );

    setCameraStatus("Uploading and analyzing...");
    setCameraOverlay("Analyzing...");

    const outputType = blob.type || mimeType || "video/webm";
    const extension = extensionFromMimeType(outputType);
    const file = new File([blob], `camera_capture.${extension}`, { type: outputType });

    closeCameraRecordOverlay();
    const wasSuccessful = await analyzeVideo(file);
    setCameraStatus(wasSuccessful ? "Scan complete." : "Analysis failed.");
    setCameraOverlay("Camera preview");
  } catch (err) {
    setCameraStatus("Camera scan failed. Please try again.");
    setCameraOverlay("Camera preview");
    alert(err.message || "Camera access denied or recording failed.");
  } finally {
    stopCameraStream();
    if (video) {
      video.srcObject = null;
      video.classList.add("is-hidden");
    }
    if (recordVideo) {
      recordVideo.srcObject = null;
      recordVideo.classList.add("is-hidden");
    }
    closeCameraRecordOverlay();
    setCameraBusy(false);
  }
}

setCameraStatus(getCameraIdleText());

window.addEventListener("beforeunload", stopCameraStream);
