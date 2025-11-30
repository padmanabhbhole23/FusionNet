# ============================================================
# ✅ FusionNet Deepfake Detector — Optimized Flask Backend
# ============================================================
import os, uuid, atexit, shutil, traceback, urllib.request, json, threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np, cv2, joblib, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Optional: suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# ✅ Configuration
# ============================================================
ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", 5000))

UPLOAD_DIR = ROOT / "uploads"
FRAMES_DIR = ROOT / "DeepFake_Frames"
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

MODEL_DIR = ROOT / "models"
CNN_LSTM_PATH = MODEL_DIR / "cnn_lstm_model.keras"
SVM_PATH = MODEL_DIR / "svm_branch.pkl"
XGB_PATH = MODEL_DIR / "xgb_branch.pkl"
FUSION_PATH = MODEL_DIR / "fusion_model.pkl"
PCA_PATH = MODEL_DIR / "pca_transform.pkl"

MODEL_CANDIDATES = [ROOT / "final_efficientnetv2_b0.keras", ROOT / "final.keras", ROOT / "model.keras"]

IMG_SIZE_BASIC, IMG_SIZE_PRO = (128, 128), 224
MAX_FRAMES, DEFAULT_N_FRAMES = 40, 12
CLASS_NAMES = ["fake", "real"]
FAKE_IDX, REAL_IDX = 0, 1

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# 🔹 Detect if we are running on Render (used to avoid loading all models)
ON_RENDER = bool(os.environ.get("RENDER_EXTERNAL_URL")) or bool(os.environ.get("RENDER"))

# ============================================================
# ✅ Face Detector Configuration
# ============================================================
FACE_DET_DIR = ROOT / "face_detector"
PROTOTXT = FACE_DET_DIR / "deploy.prototxt"
CAFFEMODEL = FACE_DET_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONF_THRESH = 0.6

HAAR_DIR = ROOT / "haar_cascades"
EYE_CASCADE_PATH = HAAR_DIR / "haarcascade_eye.xml"
SMILE_CASCADE_PATH = HAAR_DIR / "haarcascade_smile.xml"

# ============================================================
# ✅ Cleanup logic
# ============================================================
def cleanup():
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR, ignore_errors=True)
atexit.register(cleanup)

def is_video_file(path): return Path(path).suffix.lower() in VIDEO_EXTS
def is_image_file(path): return Path(path).suffix.lower() in IMAGE_EXTS

# ============================================================
# ✅ Dependency setup
# ============================================================
def ensure_dependencies():
    FACE_DET_DIR.mkdir(exist_ok=True)
    HAAR_DIR.mkdir(exist_ok=True)

    files = {
        PROTOTXT: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        CAFFEMODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        EYE_CASCADE_PATH: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
        SMILE_CASCADE_PATH: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
    }

    for path, url in files.items():
        if not path.exists():
            urllib.request.urlretrieve(url, str(path))

    face_net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
    eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
    smile_cascade = cv2.CascadeClassifier(str(SMILE_CASCADE_PATH))
    return face_net, eye_cascade, smile_cascade

face_net, eye_cascade, smile_cascade = ensure_dependencies()

# ============================================================
# ✅ Lazy Model Handles (no heavy loading at import)
# ============================================================
cnn_lstm_model = None
svm = xgb = fusion = pca = None
pro_model = None
lime_explainer = None

def load_basic_models():
    """Lazy-load FusionNet basic models (CNN-LSTM + fusion)"""
    global cnn_lstm_model, svm, xgb, fusion, pca
    if cnn_lstm_model is not None and fusion is not None:
        return

    # 💡 On Render, we try to avoid loading these heavy models
    if ON_RENDER:
        print("⚠️ Skipping basic FusionNet model loading on Render to save memory.")
        return

    print("🔹 Loading Basic Models (FusionNet)...")
    cnn_lstm_model = keras.models.load_model(str(CNN_LSTM_PATH))
    svm = joblib.load(str(SVM_PATH))
    xgb = joblib.load(str(XGB_PATH))
    fusion = joblib.load(str(FUSION_PATH))
    pca = joblib.load(str(PCA_PATH))
    print("✅ Basic models loaded")

def load_pro_model():
    """Lazy-load EfficientNet Pro model"""
    global pro_model
    if pro_model is not None:
        return
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            print(f"🔹 Loading Pro model: {candidate.name}")
            pro_model = keras.models.load_model(str(candidate))
            print("✅ Pro model loaded")
            return
    raise FileNotFoundError("No Pro model file found in root.")

def load_lime():
    """Lazy-load LIME explainer"""
    global lime_explainer
    if lime_explainer is not None:
        return
    from lime import lime_image
    lime_explainer = lime_image.LimeImageExplainer()

# ============================================================
# ✅ Helper functions
# ============================================================
def extract_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104,177,123))
    face_net.setInput(blob)
    dets = face_net.forward()
    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf > FACE_CONF_THRESH:
            box = dets[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]
            if face.size:
                return cv2.cvtColor(cv2.resize(face, (IMG_SIZE_PRO, IMG_SIZE_PRO)), cv2.COLOR_BGR2RGB)
    return None

def extract_frames_basic(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.cvtColor(cv2.resize(frame, IMG_SIZE_BASIC), cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames: return None
    frames = np.array(frames, dtype="float32") / 255.0
    return np.pad(frames, ((0, max_frames - len(frames)), (0,0),(0,0),(0,0))) if len(frames) < max_frames else frames

# ============================================================
# ✅ Fast Pro Model (EfficientNet + LIME) — Optimized
# ============================================================
def predict_pro_fast(video_path):
    load_pro_model()
    load_lime()

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        return {"error": "No frames found."}

    indices = np.linspace(0, frame_count - 1, 12, dtype=int)
    faces, fake_scores = [], []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        face = extract_face(frame)
        if face is None:
            continue
        prob = pro_model.predict(preprocess_input(np.expand_dims(face, 0)), verbose=0)[0]
        fake_scores.append(prob[FAKE_IDX])
        faces.append(face)

    cap.release()
    if not fake_scores:
        return {"mode": "Pro", "final_result": "REAL", "confidence": 0.0}

    mean_fake = np.mean(fake_scores)
    pred = "FAKE" if mean_fake > 0.5 else "REAL"
    conf = mean_fake if pred == "FAKE" else 1 - mean_fake
    result = {"mode": "Pro", "final_result": pred, "confidence": float(conf)}

    # Only generate one heatmap if fake confidence < 0.9
    if pred == "FAKE" and conf < 0.9 and faces:
        face_rgb = faces[np.argmax(fake_scores)]
        explanation = lime_explainer.explain_instance(
            face_rgb,
            lambda imgs: pro_model.predict(preprocess_input(imgs.astype(np.float32))),
            top_labels=1,
            num_samples=300
        )
        temp, mask = explanation.get_image_and_mask(
            FAKE_IDX,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        hm = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), 0.6, hm, 0.4, 0)
        fname = f"heatmap_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(FRAMES_DIR / fname), blended)
        result["analysis_files"] = [fname]
    return result

# ============================================================
# ✅ Basic Model Prediction (FusionNet)
# ============================================================
def predict_basic(path):
    # On Render we skip heavy basic models and just fallback to Pro under the hood
    if ON_RENDER:
        # Keep response structure similar but use Pro prediction
        pro_result = predict_pro_fast(path)
        pro_result["mode"] = "Basic (Pro backend on Render)"
        return pro_result

    load_basic_models()
    if cnn_lstm_model is None or fusion is None:
        return {"error": "Basic model not available on this deployment."}

    frames = extract_frames_basic(path)
    if frames is None:
        return {"error": "No frames extracted."}
    lstm_prob = cnn_lstm_model.predict(np.expand_dims(frames, 0), verbose=0)[0][0]
    lstm_label = int(lstm_prob > 0.5)
    mean_frame = np.mean(frames, axis=0).flatten().reshape(1, -1)
    fusion_prob = fusion.predict_proba(pca.transform(mean_frame))[0][1]
    fusion_label = int(fusion_prob > 0.5)
    return {
        "mode": "Basic",
        "final_result": "FAKE" if fusion_label else "REAL",
        "lstm_result": "FAKE" if lstm_label else "REAL",
        "lstm_confidence": float(lstm_prob),
        "fusion_confidence": float(fusion_prob)
    }

# ============================================================
# ✅ Flask App Initialization
# ============================================================
app = Flask(__name__)

@app.route("/frames/<path:fname>")
def frames(fname): return send_from_directory(FRAMES_DIR, fname)

# ============================================================
# ✅ Main Prediction Route
# ============================================================
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        file = request.files.get("file")
        if not file: return jsonify({"error": "No file"}), 400

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        requested_version = request.form.get("version", "basic")

        # 🔹 On Render, always use Pro under the hood to avoid loading all models
        if ON_RENDER:
            effective_version = "pro"
        else:
            effective_version = requested_version

        # BASIC MODE
        if effective_version == "basic":
            result = predict_basic(save_path)
            result["status"] = "done"
            result["requested_version"] = requested_version
            return jsonify(result)

        # PRO MODE (async)
        def run_analysis():
            try:
                result = predict_pro_fast(save_path)
                result["status"] = "done"
                result["requested_version"] = requested_version
                with open(FRAMES_DIR / f"{filename}.json", "w") as f:
                    f.write(json.dumps(result))
            except Exception as e:
                with open(FRAMES_DIR / f"{filename}.json", "w") as f:
                    f.write(json.dumps({"error": str(e)}))

        threading.Thread(target=run_analysis).start()
        return jsonify({"status": "processing", "task_id": filename})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

# ============================================================
# ✅ Polling Route for Android
# ============================================================
@app.route("/result/<task_id>")
def get_result(task_id):
    json_path = FRAMES_DIR / f"{task_id}.json"
    if not json_path.exists():
        return jsonify({"status": "processing"})
    with open(json_path) as f:
        return jsonify(json.load(f))

# ============================================================
# ✅ Futuristic AI Web Dashboard (unchanged UI)
# ============================================================
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FusionNet — Deepfake Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --accent:#8b5cf6;
  --accent-soft:#a855f7;
  --bg:#020617;
  --card-bg:rgba(15,23,42,.95);
  --text:#e5e7eb;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
  min-height:100vh;
  margin:0;
  font-family:'Poppins',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  display:flex;
  align-items:center;
  justify-content:center;
  background:radial-gradient(circle at 0 0,#1d263b 0,#020617 50%,#000 100%);
  color:var(--text);
  overflow-x:hidden;
}
.shell{
  width:100%;
  max-width:980px;
  padding:24px;
}
@media(max-width:900px){
  .shell{padding-inline:16px;}
}
.brand-title{
  font-weight:700;
  font-size:2.4rem;
  background:linear-gradient(120deg,#e5e7eb,#a855f7,#38bdf8);
  -webkit-background-clip:text;
  color:transparent;
  text-align:center;
  margin-bottom:10px;
}
.subtitle{
  text-align:center;
  opacity:.75;
  font-size:.9rem;
  margin-bottom:26px;
}
.grid{
  display:grid;
  grid-template-columns:minmax(0,2fr) minmax(0,1.5fr);
  gap:22px;
}
@media(max-width:900px){
  .grid{grid-template-columns:1fr;}
}
.card{
  position:relative;
  border-radius:24px;
  padding:22px 24px 18px;
  background:linear-gradient(135deg,rgba(15,23,42,.98),rgba(15,23,42,.9));
  box-shadow:0 24px 60px rgba(15,23,42,.95);
  border:1px solid rgba(148,163,184,.35);
  backdrop-filter:blur(18px);
  overflow:hidden;
}
.card::before{
  content:"";
  position:absolute;
  inset:-1px;
  border-radius:inherit;
  padding:1px;
  background:linear-gradient(135deg,rgba(139,92,246,.9),rgba(6,182,212,.4));
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  opacity:.16;
  pointer-events:none;
  z-index:-1;
}
.card-header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:12px;
}
.card-title{
  font-size:1.05rem;
  font-weight:600;
}
.chip{
  font-size:.75rem;
  padding:4px 10px;
  border-radius:999px;
  border:1px solid rgba(148,163,184,.7);
  text-transform:uppercase;
  letter-spacing:.06em;
  opacity:.9;
}
.field{
  margin-bottom:14px;
}
.label{
  font-size:.8rem;
  text-transform:uppercase;
  letter-spacing:.08em;
  opacity:.7;
  margin-bottom:6px;
}
select,
input[type="file"]{
  width:100%;
  border-radius:999px;
  border:1px solid rgba(148,163,184,.6);
  background:rgba(15,23,42,.92);
  color:var(--text);
  padding:10px 14px;
  font-size:.9rem;
  outline:none;
  transition:border .18s,box-shadow .18s,background .18s,transform .08s;
}
select:focus,
input[type="file"]:focus{
  border-color:var(--accent-soft);
  box-shadow:0 0 0 1px rgba(139,92,246,.4);
}
.helper{
  font-size:.78rem;
  opacity:.6;
  margin-top:4px;
}
.btn-primary{
  margin-top:10px;
  width:100%;
  border:none;
  border-radius:999px;
  padding:11px 18px;
  font-weight:600;
  font-size:.9rem;
  letter-spacing:.08em;
  text-transform:uppercase;
  background:linear-gradient(120deg,#8b5cf6,#ec4899,#22d3ee);
  background-size:230% 230%;
  color:white;
  box-shadow:0 16px 35px rgba(129,140,248,.55);
  cursor:pointer;
  transition:transform .14s ease,box-shadow .14s ease,background-position .6s ease;
}
.btn-primary:hover{
  transform:translateY(-1px);
  background-position:100% 0;
  box-shadow:0 22px 45px rgba(129,140,248,.8);
}
.btn-primary:active{
  transform:translateY(1px);
  box-shadow:0 10px 24px rgba(15,23,42,.9);
}
.loader-wrap{
  margin-top:14px;
  display:none;
  align-items:center;
  gap:10px;
  font-size:.9rem;
  opacity:.9;
}
.loader-dot{
  width:8px;
  height:8px;
  border-radius:999px;
  background:var(--accent);
  box-shadow:0 0 0 0 rgba(139,92,246,.6);
  animation:pulse 1.3s ease-out infinite;
}
@keyframes pulse{
  0%{transform:scale(1);box-shadow:0 0 0 0 rgba(139,92,246,.6);}
  80%{transform:scale(1.8);box-shadow:0 0 0 12px transparent;}
  100%{transform:scale(1);box-shadow:0 0 0 0 transparent;}
}
.side-card{
  border-radius:24px;
  padding:20px 22px 16px;
  background:radial-gradient(circle at top left,rgba(59,130,246,.22),rgba(15,23,42,.96));
  border:1px solid rgba(30,64,175,.9);
  box-shadow:0 20px 50px rgba(15,23,42,.95);
}
.side-title{
  font-size:1.02rem;
  font-weight:600;
  margin-bottom:4px;
}
.side-pill{
  font-size:.78rem;
  opacity:.75;
  margin-bottom:10px;
}
.side-metric{
  display:flex;
  justify-content:space-between;
  align-items:center;
  font-size:.86rem;
  margin-top:8px;
}
.side-badge{
  padding:3px 9px;
  border-radius:999px;
  background:rgba(34,197,94,.16);
  color:#bbf7d0;
  font-size:.72rem;
}
#result{
  margin-top:12px;
}
.result-inner h5{
  font-size:1.1rem;
  margin-bottom:4px;
}
.confidence{
  font-feature-settings:"tnum" 1,"lnum" 1;
  font-variant-numeric:tabular-nums;
}
.tag{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:3px 10px;
  border-radius:999px;
  background:rgba(15,23,42,.95);
  border:1px solid rgba(148,163,184,.6);
  font-size:.78rem;
  margin-top:6px;
}
.tag-dot{
  width:6px;
  height:6px;
  border-radius:999px;
  background:#22c55e;
}
.result-img{
  margin-top:10px;
  border-radius:18px;
  max-width:240px;
  width:100%;
  border:1px solid rgba(30,64,175,.9);
  box-shadow:0 18px 40px rgba(15,23,42,.95);
}
.alert-error{
  margin-top:14px;
  padding:10px 13px;
  border-radius:14px;
  background:rgba(248,113,113,.12);
  border:1px solid rgba(248,113,113,.9);
  color:#fecaca;
  font-size:.86rem;
}
.footer-note{
  margin-top:16px;
  font-size:.78rem;
  opacity:.55;
  text-align:center;
}
</style>
</head>
<body>
<div class="shell">
  <h1 class="brand-title">FusionNet Deepfake Detector</h1>
  <p class="subtitle">AI-powered analysis for detecting manipulated videos and synthetic faces in real time.</p>

  <div class="grid">
    <div class="card">
      <div class="card-header">
        <div class="card-title">Scan Media</div>
        <div class="chip">Real-time AI</div>
      </div>

      <form id="form" enctype="multipart/form-data" method="post">
        <div class="field">
          <div class="label">Model Version</div>
          <select name="version">
            <option value="basic">Basic · Fast</option>
            <option value="pro">Pro · Detailed + Heatmap</option>
          </select>
          <div class="helper">Basic is ideal for quick checks. Pro adds facial focus and explainability.</div>
        </div>

        <div class="field">
          <div class="label">Upload Video / Image</div>
          <input type="file" name="file" required>
          <div class="helper">Supported: MP4, AVI, MOV, MKV, WEBM, JPG, PNG.</div>
        </div>

        <button type="submit" class="btn-primary">Run Deepfake Scan</button>

        <div id="loader" class="loader-wrap">
          <div class="loader-dot"></div>
          <span>Analyzing frames and facial regions with FusionNet…</span>
        </div>
      </form>

      <div class="footer-note">Your media is processed locally on this server and not sent to any third-party service.</div>
    </div>

    <div class="side-card">
      <div class="side-title">Detection Output</div>
      <div class="side-pill">Results update automatically once the model finishes inference.</div>
      <div class="side-metric">
        <span>Status</span>
        <span class="side-badge" id="status-badge">Idle</span>
      </div>
      <div id="result"></div>
    </div>
  </div>
</div>

<script>
const form = document.getElementById('form');
const loader = document.getElementById('loader');
const resultEl = document.getElementById('result');
const statusBadge = document.getElementById('status-badge');

function setStatus(text, color){
  statusBadge.textContent = text;
  if(color === 'processing'){
    statusBadge.style.background = 'rgba(234,179,8,.16)';
    statusBadge.style.color = '#facc15';
  } else if(color === 'error'){
    statusBadge.style.background = 'rgba(248,113,113,.16)';
    statusBadge.style.color = '#fecaca';
  } else if(color === 'fake'){
    statusBadge.style.background = 'rgba(239,68,68,.18)';
    statusBadge.style.color = '#fecaca';
  } else if(color === 'real'){
    statusBadge.style.background = 'rgba(34,197,94,.18)';
    statusBadge.style.color = '#bbf7d0';
  } else {
    statusBadge.style.background = 'rgba(148,163,184,.25)';
    statusBadge.style.color = '#e5e7eb';
  }
}

form.onsubmit = async (e) => {
  e.preventDefault();
  loader.style.display = 'flex';
  setStatus('Processing', 'processing');
  resultEl.innerHTML = '';
  const data = new FormData(form);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: data
    });

    const json = await res.json();

    if (json.status === 'processing') {
      checkStatus(json.task_id);
      return;
    }
    showResult(json);
  } catch (err) {
    console.error(err);
    loader.style.display = 'none';
    setStatus('Error', 'error');
    resultEl.innerHTML = '<div class="alert-error">Something went wrong while sending the request.</div>';
  }
};

async function checkStatus(id){
  for(let i = 0; i < 40; i++){
    await new Promise(r => setTimeout(r, 2000));
    try{
      const r = await fetch('/result/' + id);
      const j = await r.json();
      if(j.status === 'done'){
        showResult(j);
        return;
      }
    }catch(e){
      console.error(e);
    }
  }
  loader.style.display = 'none';
  setStatus('Timeout', 'error');
  resultEl.innerHTML = '<div class="alert-error">The model took too long to respond. Please try again.</div>';
}

function showResult(j){
  loader.style.display = 'none';

  if(j.error){
    setStatus('Error', 'error');
    resultEl.innerHTML = '<div class="alert-error">' + j.error + '</div>';
    return;
  }

  const mode = j.mode || (j.confidence !== undefined ? 'Pro' : 'Basic');
  const confidence = (j.confidence ?? j.fusion_confidence ?? j.lstm_confidence ?? 0);
  const percent = (confidence * 100).toFixed(2);
  const label = (j.final_result || '').toUpperCase();

  if(label === 'FAKE'){
    setStatus('Fake detected', 'fake');
  } else if(label === 'REAL'){
    setStatus('Likely real', 'real');
  } else {
    setStatus('Done', '');
  }

  let html = '<div class="result-inner">';
  html += '<h5>Result: <strong>' + label + '</strong></h5>';
  html += '<p class="confidence">Confidence: <strong>' + percent + '%</strong></p>';
  html += '<div class="tag"><span class="tag-dot"></span><span>Mode: ' + mode + '</span></div>';

  if(j.lstm_result){
    html += '<p style="margin-top:8px;font-size:.82rem;opacity:.8;">LSTM branch: <strong>' + j.lstm_result +
            '</strong> · Score: ' + (j.lstm_confidence*100).toFixed(1) + '%</p>';
  }

  if(j.analysis_files){
    j.analysis_files.forEach(f => {
      html += '<img src="/frames/' + f + '" class="result-img" alt="Explanation heatmap">';
    });
  }

  html += '</div>';
  resultEl.innerHTML = html;
}
</script>
</body>
</html>
"""

@app.route("/")
def index(): return render_template_string(INDEX_HTML)

# ============================================================
# ✅ Run Server
# ============================================================
if __name__ == "__main__":
    print(f"🚀 Running FusionNet on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
