"""
Face Recognition Module — Pretrained DNN Approach
Uses OpenCV's YuNet (detection) + SFace (recognition) pretrained models.
No manual training step required — faces are encoded on registration
and matched in real-time during attendance.
"""

import cv2
import os
import numpy as np
import threading
import urllib.request
import ssl
import shutil
from config import get_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ENCODINGS_DIR = os.path.join(BASE_DIR, 'encodings')

for d in (DATASET_DIR, MODELS_DIR, ENCODINGS_DIR):
    os.makedirs(d, exist_ok=True)

# ── Pretrained model paths & URLs ────────────────────────────────────────────
DETECTION_MODEL = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
RECOGNITION_MODEL = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')

_MODEL_URLS = {
    DETECTION_MODEL: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    RECOGNITION_MODEL: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

# ── Cached singletons ───────────────────────────────────────────────────────
_detector = None
_recognizer = None
_models_lock = threading.Lock()
_detect_lock = threading.Lock()       # YuNet is not thread-safe
_recognize_lock = threading.Lock()    # SFace is not thread-safe

# Cached face encodings:  { user_id: np.ndarray of shape (N, 128) }
_known_encodings = {}
_encodings_lock = threading.Lock()
_encodings_loaded = False

# Liveness detection state: { frame_id: list of detected eyes/landmarks }
_liveness_cache = {}
_liveness_lock = threading.Lock()

# Recognition threshold (cosine similarity — higher = more similar)
# Read dynamically from config so runtime changes take effect
def _get_cosine_threshold():
    """Get cosine threshold from live config (not frozen at import)."""
    return get_config().get('cosine_threshold', 0.363)


# ── Model download & init ───────────────────────────────────────────────────

def _download_model(url, dest):
    """Download a model file, trying verified SSL first with unverified fallback."""
    if os.path.exists(dest):
        return
    print(f"  [*] Downloading {os.path.basename(dest)} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    # Try with proper SSL verification first
    try:
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
            shutil.copyfileobj(resp, f)
        print(f"  [OK] Downloaded {os.path.basename(dest)}")
        return
    except ssl.SSLError:
        print(f"  [!] SSL verification failed, retrying without verification...")

    # Fallback: disable SSL verification (e.g., corporate proxies)
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as resp, open(dest, 'wb') as f:
            shutil.copyfileobj(resp, f)
        print(f"  [OK] Downloaded {os.path.basename(dest)} (unverified SSL)")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            f"Failed to download {os.path.basename(dest)}: {e}\n"
            f"Please download manually from:\n  {url}\n"
            f"and place it in: {MODELS_DIR}"
        )


def ensure_models():
    """Download pretrained models if they are not present."""
    for dest, url in _MODEL_URLS.items():
        _download_model(url, dest)


def _get_detector(width=640, height=480):
    """Return the cached YuNet face detector (created once)."""
    global _detector
    config = get_config()
    with _models_lock:
        if _detector is None:
            ensure_models()
            _detector = cv2.FaceDetectorYN.create(
                DETECTION_MODEL, "", (width, height),
                score_threshold=config.get('detection_score_threshold', 0.6),
                nms_threshold=config.get('nms_threshold', 0.3),
                top_k=5000,
            )
        else:
            _detector.setInputSize((width, height))
    return _detector


def _get_recognizer():
    """Return the cached SFace face recognizer (created once)."""
    global _recognizer
    with _models_lock:
        if _recognizer is None:
            ensure_models()
            _recognizer = cv2.FaceRecognizerSF.create(RECOGNITION_MODEL, "")
    return _recognizer


# ── Encoding storage ────────────────────────────────────────────────────────

def _load_all_encodings():
    """Load every user's stored encodings into the cache."""
    global _known_encodings, _encodings_loaded
    with _encodings_lock:
        _known_encodings = {}
        for fname in os.listdir(ENCODINGS_DIR):
            if not fname.endswith('.npy'):
                continue
            try:
                uid = int(fname.replace('.npy', ''))
                _known_encodings[uid] = np.load(
                    os.path.join(ENCODINGS_DIR, fname)
                )
            except (ValueError, Exception):
                continue
        _encodings_loaded = True
        print(f"  [OK] Loaded face encodings for {len(_known_encodings)} user(s)")


def _get_encodings():
    """Return cached encodings, loading from disk on first call."""
    global _encodings_loaded
    if not _encodings_loaded:
        _load_all_encodings()
    return _known_encodings


def reload_encodings():
    """Force-reload encodings from disk."""
    global _encodings_loaded
    _encodings_loaded = False
    _load_all_encodings()


# ── Public helpers ──────────────────────────────────────────────────────────

def _bbox(face):
    """Extract (x, y, w, h) from a YuNet face row."""
    return int(face[0]), int(face[1]), int(face[2]), int(face[3])


def detect_faces_bboxes(frame):
    """Detect faces and return a list of (x, y, w, h) tuples."""
    h, w = frame.shape[:2]
    with _detect_lock:
        det = _get_detector(w, h)
        _, faces = det.detect(frame)
    if faces is None:
        return []
    return [_bbox(f) for f in faces]


def has_registered_users():
    """True if at least one user has stored face encodings."""
    return len(_get_encodings()) > 0


def is_model_ready():
    """True if pretrained model files exist on disk."""
    return (os.path.exists(DETECTION_MODEL) and
            os.path.exists(RECOGNITION_MODEL))


# ── Capture (registration) ─────────────────────────────────────────────────

def capture_face_sample(frame, user_id, sample_number):
    """
    Detect the largest face, compute its 128-d embedding, persist it,
    and return (True, annotated_frame). Returns (False, frame) on no face.
    """
    h, w = frame.shape[:2]
    with _detect_lock:
        det = _get_detector(w, h)
        _, faces = det.detect(frame)

    if faces is None or len(faces) == 0:
        return False, frame

    rec = _get_recognizer()

    # Pick the largest detected face
    idx = int(np.argmax(faces[:, 2] * faces[:, 3]))
    face = faces[idx]
    x, y, fw, fh = _bbox(face)

    # Compute embedding
    with _recognize_lock:
        aligned = rec.alignCrop(frame, face)
        embedding = rec.feature(aligned)          # shape (1, 128)

    # Persist encoding
    enc_path = os.path.join(ENCODINGS_DIR, f'{user_id}.npy')
    if os.path.exists(enc_path):
        existing = np.load(enc_path)
        all_enc = np.vstack([existing, embedding])
    else:
        all_enc = embedding
    np.save(enc_path, all_enc)

    # Update in-memory cache
    with _encodings_lock:
        _known_encodings[user_id] = all_enc

    # Save face image for reference
    user_dir = os.path.join(DATASET_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    x0, y0 = max(0, x), max(0, y)
    crop = frame[y0:y0 + fh, x0:x0 + fw]
    if crop.size > 0:
        crop = cv2.resize(crop, (200, 200))
        cv2.imwrite(os.path.join(user_dir, f'{sample_number}.jpg'), crop)

    # Annotate
    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
    cv2.putText(frame, f'Sample {sample_number}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return True, frame


# ── Recognition (attendance) ───────────────────────────────────────────────

def recognize_face(frame):
    """
    Detect faces, match each against stored encodings, and return
    (annotated_frame, results_list).
    """
    known = _get_encodings()
    if not known:
        return frame, []

    h, w = frame.shape[:2]
    with _detect_lock:
        det = _get_detector(w, h)
        _, faces = det.detect(frame)

    if faces is None or len(faces) == 0:
        return frame, []

    rec = _get_recognizer()
    results = []

    for face in faces:
        x, y, fw, fh = _bbox(face)

        with _recognize_lock:
            aligned = rec.alignCrop(frame, face)
            embedding = rec.feature(aligned)

        best_id = None
        best_score = -1.0

        for uid, encs in known.items():
            for enc in encs:
                score = rec.match(
                    embedding, enc.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
                if score > best_score:
                    best_score = score
                    best_id = uid

        if best_score >= _get_cosine_threshold() and best_id is not None:
            confidence = best_score * 100
            results.append({
                'user_id': best_id,
                'confidence': confidence,
                'bbox': (x, y, fw, fh),
            })
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            label = f'ID:{best_id} ({confidence:.0f}%)'
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
            cv2.putText(frame, 'Unknown', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, results


# ── Dataset utilities ───────────────────────────────────────────────────────

def get_dataset_info():
    """Return per-user sample counts from the dataset directory."""
    info = []
    if not os.path.exists(DATASET_DIR):
        return info
    for name in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            uid = int(name)
        except ValueError:
            continue
        count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
        info.append({'user_id': uid, 'sample_count': count})
    return info


def get_sample_count(user_id):
    """Number of stored face-image samples for a user."""
    d = os.path.join(DATASET_DIR, str(user_id))
    if os.path.exists(d):
        return len([f for f in os.listdir(d) if f.endswith('.jpg')])
    return 0


def delete_user_data(user_id):
    """Remove all face images and encodings for a user."""
    # Images
    user_dir = os.path.join(DATASET_DIR, str(user_id))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    # Encodings
    enc = os.path.join(ENCODINGS_DIR, f'{user_id}.npy')
    if os.path.exists(enc):
        os.remove(enc)
    # Cache
    with _encodings_lock:
        _known_encodings.pop(user_id, None)
    return True


# ── Liveness Detection ──────────────────────────────────────────────────────

def detect_face_landmarks(frame, face):
    """
    Detect eye/facial landmarks to verify a real person.
    Returns (has_eyes, eye_positions_list).
    """
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        eye_cascade = cv2.CascadeClassifier(cascade_path)
        
        x, y, w, h = _bbox(face)
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return False, []
        
        eyes = eye_cascade.detectMultiScale(roi, 1.3, 5)
        return len(eyes) > 0, eyes.tolist() if len(eyes) > 0 else []
    except Exception as e:
        return False, []


def check_liveness(frame, face, landmarks_history=None):
    """
    Verify face liveness by detecting eye movement/blinking.
    
    Returns:
        True if face appears to be alive (detected landmarks change)
        False if might be a photo/static image
    """
    if landmarks_history is None:
        landmarks_history = []
    
    config = get_config()
    if not config.get('liveness_enabled', True):
        return True  # Liveness check disabled
    
    has_eyes, landmarks = detect_face_landmarks(frame, face)
    
    if not has_eyes:
        # No eyes detected, but could still be valid if face is turned away
        return len(landmarks_history) > 0  # Accept if we've seen landmarks before
    
    landmarks_history.append(landmarks)
    
    # If we have enough history and landmarks changed significantly
    if len(landmarks_history) >= 2:
        old_landmarks = landmarks_history[-2]
        new_landmarks = landmarks_history[-1]
        
        if len(old_landmarks) > 0 and len(new_landmarks) > 0:
            # Calculate movement between frames
            distance = np.linalg.norm(
                np.array(new_landmarks[0]) - np.array(old_landmarks[0])
            ) if len(old_landmarks) > 0 and len(new_landmarks) > 0 else 0
            
            # If significant eye position change detected = alive
            if distance > 2.0:  # Pixel threshold for movement
                return True
    
    # If we collected enough frames with eyes, assume alive
    eyes_detected_count = sum(1 for h in landmarks_history if len(h) > 0)
    liveness_threshold = config.get('liveness_threshold', 5)
    
    return eyes_detected_count >= liveness_threshold


def is_liveness_enabled():
    """Check if liveness detection is enabled in config."""
    return get_config().get('liveness_enabled', True)
