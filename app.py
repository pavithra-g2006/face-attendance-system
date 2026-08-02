"""
FaceTrack — Flask Application
Routes for registration, live attendance, and records.
Uses pretrained DNN models (no manual training step).
"""

import cv2
import time
import os
import threading
import functools
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from datetime import datetime
from config import init_config, get_config
from database import (
    init_db, add_user, get_user, get_all_users, get_user_count,
    mark_attendance, get_attendance_records, get_today_attendance_count,
    get_available_dates, user_name_exists, delete_user, get_attendance_stats,
    get_user_statistics, export_attendance_report,
)
from face_recognition_module import (
    detect_faces_bboxes, capture_face_sample, recognize_face,
    has_registered_users, ensure_models, get_dataset_info,
    get_sample_count, delete_user_data, check_liveness, is_liveness_enabled,
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'facetrack-secret-key-change-in-production')

# ── Admin Credentials ────────────────────────────────────────────────────────
# Set via environment variables in production:
#   ADMIN_USERNAME=your_username  ADMIN_PASSWORD=your_secure_password
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'ADMIN')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'ADMIN123')

# Load configuration
config = init_config()
TOTAL_SAMPLES = config.get('total_samples', 30)


# ── Authentication ──────────────────────────────────────────────────────────
def login_required(f):
    """Decorator to protect admin-only routes."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': 'Authentication required. Please login.'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ── Camera management ───────────────────────────────────────────────────────
camera = None
camera_lock = threading.Lock()

# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_TIME = 2  # seconds


def get_camera():
    """Open the camera with optimized settings for laptop webcams."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = _initialize_camera()
    return camera


def _initialize_camera():
    """Initialize and configure camera with fallback options."""
    # Try different backends for Windows
    backends = [
        (0, cv2.CAP_DSHOW, "DirectShow"),     # Best for Windows
        (0, cv2.CAP_MSMF, "MSMF"),            # Alternative Windows backend
        (0, cv2.CAP_V4L2, "V4L2"),            # Linux fallback
        (1, cv2.CAP_DSHOW, "DirectShow (USB)"),
        (1, cv2.CAP_MSMF, "MSMF (USB)"),
    ]
    
    for idx, backend, backend_name in backends:
        try:
            cam = cv2.VideoCapture(idx, backend)
            
            if not cam.isOpened():
                cam.release()
                continue
            
            # Configure camera settings
            if _configure_camera(cam):
                print(f"  [OK] Camera opened on index {idx} using {backend_name}")
                print(f"      Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
                
                # Warm up camera (discard first few frames)
                print(f"  [*] Warming up camera ({CAMERA_WARMUP_TIME}s)...")
                start_time = time.time()
                while time.time() - start_time < CAMERA_WARMUP_TIME:
                    ret, _ = cam.read()
                    if not ret:
                        raise RuntimeError("Camera read failed during warmup")
                
                print(f"  [OK] Camera ready")
                return cam
        
        except Exception as e:
            if cam is not None:
                try:
                    cam.release()
                except:
                    pass
            continue
    
    # No camera found
    print("  [!] No camera detected. Tried:")
    for idx, backend, name in backends:
        print(f"      - Index {idx} ({name})")
    return None


def _configure_camera(cam):
    """Configure camera for optimal face recognition performance."""
    try:
        # Set resolution with fallback
        resolutions = [
            (640, 480),
            (800, 600),
            (960, 720),
            (1280, 720),
            (320, 240),  # Fallback for slower systems
        ]
        
        resolution_set = False
        for width, height in resolutions:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                resolution_set = True
                break
        
        if not resolution_set:
            print(f"  [!] Could not set resolution to {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"      Using available resolution: {actual_width}x{actual_height}")
        
        # Set frame rate
        cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        actual_fps = cam.get(cv2.CAP_PROP_FPS)
        
        # Exposure settings for better face detection
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Small buffer to reduce latency
        
        # White balance
        try:
            cam.set(cv2.CAP_PROP_AUTO_WB, 1)
        except:
            pass  # Not all cameras support this
        
        return True
    
    except Exception as e:
        print(f"  [!] Error configuring camera: {e}")
        return False


def release_camera():
    """Release the camera resource."""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


# ── Shared state (protected by lock) ────────────────────────────────────────
_state_lock = threading.Lock()

capture_state = {
    'active': False,
    'user_id': None,
    'samples_taken': 0,
    'total_samples': TOTAL_SAMPLES,
    'status': 'idle',
}
recognition_state = {
    'active': False,
    'results': [],
    'status': 'idle',
}


# ── MJPEG streaming ────────────────────────────────────────────────────────

def _black_frame(text="No camera detected"):
    """Return a JPEG-encoded black frame with an error message."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def _process_frame(frame, mode):
    """Process a single frame based on the mode (shared logic, no loop).
    
    Returns the processed frame (with annotations/overlays applied).
    Also handles side-effects like updating capture/recognition state.
    """
    frame = cv2.flip(frame, 1)

    # ── Register mode ────────────────────────────────────────────
    if mode == 'register':
        with _state_lock:
            active = capture_state['active']
            uid = capture_state['user_id']
            next_sample = capture_state['samples_taken'] + 1

        if active:
            captured, frame = capture_face_sample(frame, uid, next_sample)
            if captured:
                with _state_lock:
                    capture_state['samples_taken'] = next_sample
                    if next_sample >= capture_state['total_samples']:
                        capture_state['active'] = False
                        capture_state['status'] = 'complete'
        else:
            # Show detection boxes while idle on register page
            for (x, y, w, h) in detect_faces_bboxes(frame):
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 255), 2)

    # ── Attendance mode ──────────────────────────────────────────
    elif mode == 'attendance':
        with _state_lock:
            active = recognition_state['active']

        if active:
            frame, results = recognize_face(frame)
            if results:
                with _state_lock:
                    recognition_state['results'] = results
                for r in results:
                    user = get_user(r['user_id'])
                    if user:
                        marked, message = mark_attendance(
                            r['user_id'], user['name'], r['confidence'])
                        r['name'] = user['name']
                        r['newly_marked'] = marked
                        r['message'] = message
        else:
            for (x, y, w, h) in detect_faces_bboxes(frame):
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 255), 2)

    # ── Preview mode (dashboard) ────────────────────────────────
    else:
        for (x, y, w, h) in detect_faces_bboxes(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 255), 2)

    return frame


def capture_single_frame(mode='preview'):
    """Capture a SINGLE frame from the laptop camera — NO LOOP.
    
    Opens camera → reads one frame → processes it → returns JPEG bytes.
    This is the loop-free alternative to generate_frames().
    
    Args:
        mode: 'preview', 'register', or 'attendance'
        
    Returns:
        JPEG bytes of the processed frame.
    """
    cam = get_camera()

    if cam is None:
        return _black_frame()

    success, frame = cam.read()
    if not success:
        return _black_frame("Failed to read frame")

    # Process the single frame (detection, recognition, etc.)
    frame = _process_frame(frame, mode)

    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return _black_frame("Failed to encode frame")

    return buf.tobytes()


def generate_frames(mode='preview'):
    """Yield MJPEG frames for the given mode (streaming with loop)."""
    cam = get_camera()

    if cam is None:
        frame_bytes = _black_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    while True:
        success, frame = cam.read()
        if not success:
            time.sleep(0.1)
            cam = get_camera()
            if cam is None:
                break
            continue

        frame = _process_frame(frame, mode)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        time.sleep(0.03)          # ≈ 30 fps cap


# ── Auth Routes ─────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Admin login page and handler."""
    if session.get('admin_logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session.permanent = True
            if request.is_json:
                return jsonify({'success': True, 'redirect': url_for('index')})
            return redirect(url_for('index'))
        else:
            if request.is_json:
                return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Admin logout."""
    session.clear()
    return redirect(url_for('login_page'))


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    stats = {
        'total_users': get_user_count(),
        'today_attendance': get_today_attendance_count(),
        'has_users': has_registered_users(),
        'users': get_all_users(),
    }
    return render_template('index.html', stats=stats, is_admin=session.get('admin_logged_in', False))


@app.route('/register')
@login_required
def register_page():
    return render_template('register.html', total_samples=TOTAL_SAMPLES)


@app.route('/api/register', methods=['POST'])
@login_required
def register_user():
    """Register a new user for face recognition.
    
    Returns:
        success: bool
        user_id: int (on success)
        message: str (error or success message)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'Invalid request format'
            }), 400
            
        name = data.get('name', '').strip()

        # Validation
        if not name:
            return jsonify({'success': False, 'message': 'Name cannot be empty'}), 400
        
        if len(name) < 2:
            return jsonify({
                'success': False,
                'message': 'Name must be at least 2 characters long'
            }), 400
        
        if len(name) > 100:
            return jsonify({
                'success': False,
                'message': 'Name must not exceed 100 characters'
            }), 400
        
        # Check for invalid characters
        if not all(c.isalnum() or c.isspace() or c in "'-" for c in name):
            return jsonify({
                'success': False,
                'message': 'Name contains invalid characters. Use letters, numbers, spaces, hyphens, and apostrophes only.'
            }), 400
        
        if user_name_exists(name):
            return jsonify({
                'success': False,
                'message': f'"{name}" is already registered. Please use a different name.'
            }), 409  # Conflict status code

        # Prevent starting a new registration while another is in progress
        with _state_lock:
            if capture_state['active']:
                return jsonify({
                    'success': False,
                    'message': 'Another registration is already in progress. Please wait for it to complete or stop it first.'
                }), 409

        # Create user
        try:
            user_id = add_user(name)
        except Exception as e:
            print(f"  [!] Database error: {e}")
            return jsonify({
                'success': False,
                'message': 'Database error. Please try again.'
            }), 500

        # Start capture session
        with _state_lock:
            capture_state.update({
                'active': True,
                'user_id': user_id,
                'samples_taken': 0,
                'total_samples': TOTAL_SAMPLES,
                'status': 'capturing',
            })

        if config.get('enable_logs', True):
            print(f"  [*] Registration started: {name} (ID: {user_id})")

        return jsonify({
            'success': True,
            'user_id': user_id,
            'name': name,
            'message': (f'Registration started for {name}. '
                       f'Position your face in the webcam. '
                       f'We will capture {TOTAL_SAMPLES} samples automatically.')
        }), 201
        
    except Exception as e:
        print(f"  [!] Registration error: {e}")
        return jsonify({
            'success': False,
            'message': 'An unexpected error occurred. Please try again.'
        }), 500


@app.route('/api/capture_status')
def capture_status():
    with _state_lock:
        return jsonify({
            'active': capture_state['active'],
            'samples_taken': capture_state['samples_taken'],
            'total_samples': capture_state['total_samples'],
            'status': capture_state['status'],
        })


@app.route('/api/stop_capture', methods=['POST'])
@login_required
def stop_capture():
    with _state_lock:
        capture_state['active'] = False
        capture_state['status'] = 'idle'
    return jsonify({'success': True})


@app.route('/video_feed/<mode>')
def video_feed(mode):
    if mode not in ('preview', 'register', 'attendance'):
        return jsonify({'error': 'Invalid mode'}), 400
    return Response(generate_frames(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/capture_frame/<mode>')
def capture_frame(mode):
    """Capture a SINGLE frame from the laptop camera — NO LOOP.
    
    Returns a JPEG image. Call this endpoint once for a single snapshot.
    The frontend can poll this at intervals if continuous display is needed,
    or call it once for a one-shot capture.
    """
    if mode not in ('preview', 'register', 'attendance'):
        return jsonify({'error': 'Invalid mode'}), 400
    
    frame_bytes = capture_single_frame(mode)
    return Response(frame_bytes, mimetype='image/jpeg')


# ── Attendance ──────────────────────────────────────────────────────────────

@app.route('/attendance')
def attendance_page():
    return render_template(
        'attendance.html',
        records=get_attendance_records(),
        has_users=has_registered_users(),
    )


@app.route('/api/start_attendance', methods=['POST'])
def start_attendance():
    """Start face recognition for attendance marking."""
    try:
        if not has_registered_users():
            return jsonify({
                'success': False,
                'message': 'No registered faces found. Please register at least one person first.'
            }), 409
        
        with _state_lock:
            recognition_state.update({
                'active': True,
                'results': [],
                'status': 'recognizing'
            })
        
        if config.get('enable_logs', True):
            print(f"  [*] Attendance recognition started")
        
        return jsonify({
            'success': True,
            'message': 'Face recognition started. Position yourself in front of the camera.'
        }), 200
    
    except Exception as e:
        print(f"  [!] Attendance start error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to start attendance. Please try again.'
        }), 500


@app.route('/api/stop_attendance', methods=['POST'])
def stop_attendance():
    """Stop face recognition for attendance."""
    try:
        with _state_lock:
            recognition_state.update({
                'active': False,
                'status': 'idle'
            })
        return jsonify({'success': True, 'message': 'Recognition stopped.'})
    except Exception as e:
        print(f"  [!] Stop attendance error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to stop recognition.'
        }), 500


@app.route('/api/attendance_status')
def attendance_status():
    with _state_lock:
        results_copy = list(recognition_state['results'])
        status = recognition_state['status']
        active = recognition_state['active']

    named = []
    for r in results_copy:
        user = get_user(r['user_id'])
        named.append({
            'user_id': r['user_id'],
            'name': user['name'] if user else 'Unknown',
            'confidence': r['confidence'],
        })
    return jsonify({'active': active, 'results': named, 'status': status})


@app.route('/api/today_records')
def today_records():
    return jsonify({'records': get_attendance_records()})


# ── Records ─────────────────────────────────────────────────────────────────

@app.route('/records')
def records_page():
    date = request.args.get('date')
    records = get_attendance_records(date) if date else get_attendance_records()
    return render_template('records.html', records=records,
                           dates=get_available_dates(), selected_date=date)


@app.route('/api/records/<date>')
def get_records(date):
    return jsonify({'records': get_attendance_records(date), 'date': date})


# ── User management ────────────────────────────────────────────────────────

@app.route('/api/delete_user/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user_route(user_id):
    user = get_user(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    delete_user(user_id)
    delete_user_data(user_id)
    return jsonify({'success': True,
                    'message': f'Deleted "{user["name"]}" (ID: {user_id}).'})


# ── Admin Dashboard ─────────────────────────────────────────────────────────

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard with system statistics and controls."""
    stats = get_attendance_stats()
    users = get_all_users()
    user_stats = [get_user_statistics(u['id']) for u in users]
    
    return render_template('admin.html',
                         stats=stats,
                         users=user_stats,
                         config=get_config())


@app.route('/api/admin/stats')
@login_required
def api_admin_stats():
    """Get attendance statistics for admin dashboard."""
    return jsonify(get_attendance_stats())


@app.route('/api/admin/user_stats/<int:user_id>')
@login_required
def api_user_stats(user_id):
    """Get specific user statistics."""
    stats = get_user_statistics(user_id)
    if not stats:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(stats)


@app.route('/api/admin/export', methods=['GET'])
@login_required
def api_export_report():
    """Export attendance report for date range."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    records = export_attendance_report(start_date, end_date)
    return jsonify({'records': records, 'count': len(records)})


@app.route('/api/admin/config', methods=['GET', 'POST'])
@login_required
def api_admin_config():
    """Get or update configuration."""
    if request.method == 'GET':
        return jsonify(get_config())
    
    try:
        data = request.get_json()
        from config import set_config_value
        
        # Validate and update each config value
        for key, value in data.items():
            if key in ['cosine_threshold', 'detection_score_threshold', 'liveness_threshold']:
                # Validate numeric ranges
                if not (0 <= float(value) <= 1 if key != 'liveness_threshold' else 0 <= int(value) <= 100):
                    return jsonify({'success': False, 'message': f'Invalid value for {key}'}), 400
            
            set_config_value(key, value)
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


# ── Global Error Handlers ───────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    return jsonify({
        'success': False,
        'message': 'Bad request. Please check your input and try again.'
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    return jsonify({
        'success': False,
        'message': 'Resource not found.'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 Method Not Allowed errors."""
    return jsonify({
        'success': False,
        'message': 'Method not allowed for this endpoint.'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    print(f"  [!] Internal error: {error}")
    return jsonify({
        'success': False,
        'message': 'An internal server error occurred. Please try again later.'
    }), 500


# ── Lifecycle ───────────────────────────────────────────────────────────────
# NOTE: Do NOT use teardown_appcontext to release the camera!
# teardown_appcontext fires after EVERY request, which kills the camera
# while the MJPEG video_feed stream is still active.
# Instead, release camera only on process exit.
import atexit
atexit.register(release_camera)


if __name__ == '__main__':
    print("==============================================")
    print("  FaceTrack - Attendance System")
    print("==============================================")

    print("\n  Checking pretrained models ...")
    ensure_models()

    init_db()

    print(f"\n  -> http://localhost:5000")
    print("==============================================\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
