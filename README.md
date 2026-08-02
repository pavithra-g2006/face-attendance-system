# 🎭 FaceTrack - Face Recognition Attendance System

**A production-ready face recognition attendance system using pretrained DNN models (YuNet + SFace) with no manual training required.**

> ✅ Zero training needed • 🎥 Real-time detection • � Admin login system • 📱 Mobile responsive

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Login System](#login-system)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Endpoints](#api-endpoints)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

---

## 🎯 How It Works

### For Users (No Login Required)
Users don't need any account or password. They simply:

1. **Face the Camera** — Stand in front of the webcam
2. **Auto Recognition** — The system detects and recognizes the face
3. **Attendance Marked** — Done! Attendance is recorded instantly

### For Admin (Login Required)
Only the admin can register new users and manage the system:

1. **Login** at `/login` with admin credentials
2. **Register users** by capturing 30 face samples via webcam
3. **Manage system** — view stats, edit config, delete users, export reports

---

## 🔒 Login System

### Admin Credentials

| Field    | Value       |
|----------|-------------|
| Username | `ADMIN`     |
| Password | `ADMIN123`  |

### Access Control

| Page / Feature | Public | Admin Only |
|----------------|--------|------------|
| Welcome Page (`/`) | ✅ | — |
| Mark Attendance (`/attendance`) | ✅ | — |
| View Records (`/records`) | ✅ | — |
| Register New Users (`/register`) | ❌ | ✅ |
| Admin Dashboard (`/admin`) | ❌ | ✅ |
| Delete Users | ❌ | ✅ |
| System Configuration | ❌ | ✅ |
| Export Reports | ❌ | ✅ |

### How to Login
1. Click **"Admin Login"** on the welcome page or navigation bar
2. Enter username: `ADMIN` and password: `ADMIN123`
3. After login, you'll see **Register**, **Admin**, and **Logout** in the navigation
4. Click **Logout** when done

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FaceTrack Attendance System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Flask Web Application (app.py)                   │   │
│  │  Auth: Session-based admin login (ADMIN/ADMIN123)        │   │
│  │  Routes: /login, /attendance, /register, /admin          │   │
│  │  Real-time MJPEG video streaming & face detection        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                    ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         PUBLIC                 │    ADMIN ONLY           │   │
│  ├────────────────────────────────┼─────────────────────────┤   │
│  │  Welcome Page    (/)           │  Register   (/register) │   │
│  │  Attendance  (/attendance)     │  Admin Panel  (/admin)  │   │
│  │  Records     (/records)        │  Delete Users           │   │
│  │  Login       (/login)          │  Export Reports         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                    ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Face Recognition Module (face_recognition_module.py)   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Detection:  YuNet (2023) - Face bounding boxes          │   │
│  │  Recognition: SFace (2021) - 128-D face embeddings       │   │
│  │  Matching:   Cosine similarity (threshold: 0.363)        │   │
│  │  Liveness:   Eye detection & movement tracking           │   │
│  └──────────────────────────────────────────────────────────┘   │
│       ↓                              ↓                           │
│  ┌─────────────────┐        ┌────────────────┐                  │
│  │ /encodings/     │        │   /dataset/    │                  │
│  │ Face vectors    │        │ Sample images  │                  │
│  │ (128-D numpy)   │        │ (JPEG)         │                  │
│  └─────────────────┘        └────────────────┘                  │
│       ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Database (SQLite + CSV)                          │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  SQLite:  Users, Attendance Log, Timestamps              │   │
│  │  CSV:     Daily attendance records                       │   │
│  │  Config:  System settings (JSON)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Authentication System
- Session-based Flask login
- Hardcoded admin credentials (ADMIN / ADMIN123)
- `@login_required` decorator protects admin routes
- API routes return 401 JSON for unauthenticated requests
- Page routes redirect to `/login` for unauthenticated users

#### 2. Camera System
- Supports integrated laptop cameras and USB cameras
- Multiple backend support: DirectShow, MSMF, V4L2
- Automatic resolution fallback (640×480 → 800×600 → etc.)
- Auto-focus, auto-exposure, auto white-balance
- 30 FPS optimized capture

#### 3. Face Detection (YuNet 2023)
- Pretrained model: `face_detection_yunet_2023mar.onnx` (6.3 MB)
- Detection confidence: 0.6
- Outputs: Bounding boxes (x, y, width, height)
- Real-time performance: ~20-30 FPS

#### 4. Face Recognition (SFace 2021)
- Pretrained model: `face_recognition_sface_2021dec.onnx` (35 MB)
- Feature extraction: 128-dimensional vectors
- Matching: Cosine similarity
- Threshold: 0.363 (configurable)

#### 5. Liveness Detection
- Eye detection using Haar cascades
- Tracks eye movement between frames
- Prevents spoofing with photos/videos
- Configurable sensitivity

#### 6. Storage System
- **SQLite Database:** User management, audit logs
- **CSV Files:** Daily attendance records
- **NumPy Arrays:** Face encodings (efficient storage)
- **JPEG Images:** Sample faces for reference

---

## ✨ Key Features

### Registration (Admin Only)
- ✅ Capture 30 face samples automatically
- ✅ No manual training required
- ✅ Instant face encoding (128-D vector)
- ✅ Multiple samples for accuracy improvement
- ✅ Protected behind admin login

### Attendance (No Login Required)
- ✅ Real-time face recognition
- ✅ Users just show their face — no login needed
- ✅ Liveness detection to prevent spoofing
- ✅ Automatic CSV logging
- ✅ Duplicate marking prevention (configurable timeout)
- ✅ Confidence scores on matches

### Security
- ✅ Admin-only login system (ADMIN / ADMIN123)
- ✅ Session-based authentication
- ✅ Protected registration & admin routes
- ✅ Tunable confidence threshold
- ✅ Liveness detection (eye movement)
- ✅ Thread-safe operations
- ✅ Error handling & validation

### Monitoring (Admin Only)
- ✅ Admin dashboard with statistics
- ✅ User attendance analytics
- ✅ Real-time metrics
- ✅ Export reports (CSV)
- ✅ Configuration management

### User Experience
- ✅ Clean welcome page for users
- ✅ Mobile responsive design
- ✅ Dark mode premium UI
- ✅ Toast notifications
- ✅ Real-time video streaming
- ✅ Progress tracking

---

## 📦 Prerequisites

### System Requirements
- Windows 10/11 or Linux
- Python 3.8+
- Webcam (integrated or USB)
- 2GB RAM minimum
- 500 MB disk space

### Software Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Webcam drivers installed

### Verify Python Installation
```powershell
python --version
pip --version
```

---

## 🚀 Installation

### Step 1: Clone/Download Project
```powershell
cd d:\Face
```

### Step 2: Check Requirements
```powershell
cat requirements.txt
```

**Expected output:**
```
flask==3.0.0
opencv-contrib-python==4.9.0.80
numpy==1.26.3
Pillow==10.2.0
```

### Step 3: Create Virtual Environment (Recommended)
```powershell
# On Windows
python -m venv venv
venv\Scripts\Activate.ps1

# Or on Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Test Camera
```powershell
python test_camera.py
```

This will verify:
- ✓ Camera availability
- ✓ Resolution support
- ✓ Frame rate performance
- ✓ Lighting conditions
- ✓ Face detection capability

### Step 6: Run Application
```powershell
python app.py
```

**Expected startup output:**
```
==============================================
  FaceTrack - Attendance System
==============================================

  Checking pretrained models ...
  [OK] Models found

  === Configuration Loaded ===
  Cosine Threshold:         0.363
  Liveness Detection:       True
  Auto-save Attendance:     True
  ===========================

  -> http://localhost:5000
==============================================
```

### Step 7: Open in Browser
Navigate to: **http://localhost:5000**

---

## ⚡ Quick Start

### 1. Admin Login
```
1. Open http://localhost:5000
2. Click "Admin Login" on the welcome page
3. Username: ADMIN
4. Password: ADMIN123
5. You now have access to Register & Admin panel
```

### 2. Register a User (Admin Only)
```
1. Login as admin first
2. Click "Register" in navigation
3. Enter full name (e.g., "John Doe")
4. Position face in webcam
5. Click "Start Capture"
6. System captures 30 samples automatically
7. Status updates: 1/30, 2/30, ..., 30/30
8. Registration complete!
```

### 3. Mark Attendance (No Login Required)
```
1. Open http://localhost:5000
2. Click "Mark Attendance"
3. Click "Start Recognition"
4. Show your face to the webcam
5. Face is detected and recognized
6. Attendance marked automatically!
```

### 4. View Records
```
1. Click "Records" in navigation
2. Select date or view today
3. See all attendance entries with times
4. Check confidence scores
```

### 5. Admin Control (Admin Only)
```
1. Login as admin first
2. Click "Admin" in navigation
3. View system statistics
4. Edit configuration (threshold, liveness)
5. Manage users (view stats, delete)
6. Export attendance reports
7. Click "Logout" when done
```

---

## ⚙️ Configuration

### Default Configuration (auto-created)
File: `config.json`

```json
{
  "cosine_threshold": 0.363,
  "detection_score_threshold": 0.6,
  "nms_threshold": 0.3,
  "liveness_enabled": true,
  "liveness_threshold": 5,
  "liveness_frames": 30,
  "total_samples": 30,
  "auto_save_attendance": true,
  "min_confidence_for_attendance": 0.363,
  "duplicate_check_timeout": 5,
  "enable_logs": true,
  "log_attendance": true
}
```

### Configuration Options

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `cosine_threshold` | 0.363 | 0.0-1.0 | Face match confidence |
| `detection_score_threshold` | 0.6 | 0.0-1.0 | Face detection confidence |
| `liveness_enabled` | true | bool | Enable anti-spoofing |
| `liveness_threshold` | 5 | 1-30 | Required eye movements |
| `total_samples` | 30 | 10-100 | Samples per registration |
| `duplicate_check_timeout` | 5 | 1-60 | Seconds before re-marking |
| `auto_save_attendance` | true | bool | CSV logging on match |

### Change Configuration
**Method 1: Edit JSON**
```json
{
  "cosine_threshold": 0.40,
  "liveness_enabled": false
}
```

**Method 2: Admin Dashboard (requires login)**
- Login as admin
- Navigate to `/admin`
- Click "Edit Config"
- Modify settings
- Click "Save Changes"

---

## 📖 Usage Guide

### Laptop Camera Troubleshooting

**Problem: "No camera detected"**
- Run: `python test_camera.py`
- Check Device Manager for camera
- Disable other apps using camera
- Reinstall camera drivers

**Problem: Blurry faces**
- Increase lighting (50-150 brightness)
- Position face 30-50 cm away
- System has autofocus (auto-enabled)

**Problem: Face not detected**
- Check lighting conditions
- Face should be frontal
- Run diagnostics: `python test_camera.py`

**Problem: Slow FPS (<15)**
- Close background apps
- Lower resolution
- Update GPU drivers
- System auto-fallbacks to lower resolution

### Best Practices

1. **Registration (Admin)**
   - Good lighting (natural or LED)
   - Neutral background
   - Face centered in frame
   - Look directly at camera
   - System captures automatically (don't move away)

2. **Attendance (Users)**
   - Similar lighting as registration
   - Similar distance from camera
   - No large changes in appearance (hat, glasses)
   - Still or slow movements
   - No login required — just face the camera

3. **System Maintenance (Admin)**
   - Review records regularly
   - Delete old attendance records if needed
   - Clean camera lens
   - Update browser cache

---

## 🔌 API Endpoints

### Web Pages
```
GET  /                 Welcome page (public)
GET  /login            Admin login page (public)
GET  /logout           Admin logout (clears session)
GET  /attendance       Attendance page (public)
GET  /records          Records page (public)
GET  /register         Registration page (admin only — redirects to /login)
GET  /admin            Admin dashboard (admin only — redirects to /login)
```

### Authentication API
```
POST /login
  Body: {"username": "ADMIN", "password": "ADMIN123"}
  Success: {"success": true, "redirect": "/"}
  Failure: {"success": false, "message": "Invalid username or password"} (401)
```

### Registration API (Admin Only)
```
POST /api/register
  Body: {"name": "John Doe"}
  Response: {"success": true, "user_id": 1, "message": "..."}
  Unauthorized: {"success": false, "message": "Authentication required..."} (401)

GET  /api/capture_status
  Response: {"active": true, "samples_taken": 15, "total_samples": 30}

POST /api/stop_capture (admin only)
  Response: {"success": true}
```

### Attendance API (Public)
```
POST /api/start_attendance
  Response: {"success": true, "message": "Recognition started"}

POST /api/stop_attendance
  Response: {"success": true}

GET  /api/attendance_status
  Response: {"active": true, "results": [...], "status": "recognizing"}

GET  /api/today_records
  Response: {"records": [...]}
```

### Records API (Public)
```
GET  /records?date=2026-02-26
  Returns: Attendance records for date

GET  /api/records/<date>
  Response: {"records": [...], "date": "2026-02-26"}
```

### Admin API (Admin Only — returns 401 if not logged in)
```
GET  /api/admin/stats
  Response: {"total_users": 10, "today_attendance": 8, ...}

GET  /api/admin/user_stats/<user_id>
  Response: {"user_id": 1, "name": "John", "total_attendance": 20, ...}

GET  /api/admin/export?start_date=2026-01-01&end_date=2026-02-26
  Response: CSV export of attendance records

GET  /api/admin/config
  Response: {"cosine_threshold": 0.363, ...}

POST /api/admin/config
  Body: {"cosine_threshold": 0.40, ...}
  Response: {"success": true}

DELETE /api/delete_user/<user_id>
  Response: {"success": true, "message": "User deleted"}
```

### Video Streams
```
GET /video_feed/preview    Dashboard preview
GET /video_feed/register   Registration feed
GET /video_feed/attendance Attendance feed
```

---

## 📁 File Structure

```
d:\Face\
├── app.py                          # Main Flask application + auth system
├── config.py                       # Configuration management
├── database.py                     # SQLite & CSV management
├── face_recognition_module.py      # YuNet + SFace models
├── test_camera.py                  # Camera diagnostics tool
├── optimize_config.py              # Configuration optimizer
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── config.json                     # Configuration (auto-created)
├── face_attendance.db              # SQLite database (auto-created)
│
├── models/                         # Pretrained ONNX models
│   ├── face_detection_yunet_2023mar.onnx         (auto-downloaded)
│   └── face_recognition_sface_2021dec.onnx       (auto-downloaded)
│
├── encodings/                      # Face embeddings (128-D vectors)
│   ├── 1.npy                       # User 1 face vectors
│   ├── 2.npy                       # User 2 face vectors
│   └── ...
│
├── dataset/                        # Sample face images
│   ├── 1/
│   │   ├── 1.jpg, 2.jpg, ..., 30.jpg
│   ├── 2/
│   │   └── ...
│
├── attendance/                     # Daily attendance records
│   ├── 2026-02-26.csv
│   ├── 2026-02-25.csv
│   └── ...
│
├── static/
│   ├── css/
│   │   └── style.css               # Premium dark mode UI
│   └── js/
│       ├── app.js                  # Frontend logic
│       └── utils.js                # Error handling & validation
│
├── templates/
│   ├── base.html                   # Base template (conditional nav)
│   ├── index.html                  # Welcome page (public)
│   ├── login.html                  # Admin login page
│   ├── register.html               # Registration page (admin only)
│   ├── attendance.html             # Attendance page (public)
│   ├── records.html                # Records page (public)
│   └── admin.html                  # Admin dashboard (admin only)
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "No Module Named 'cv2'"
```powershell
pip install opencv-contrib-python==4.9.0.80
```

#### 2. "No Module Named 'flask'"
```powershell
pip install flask==3.0.0
```

#### 3. Models not downloading
- Check internet connection
- Manual download:
  - [YuNet](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
  - [SFace](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)
- Place in `models/` folder

#### 4. Camera Permission Denied
- Windows: Allow in Firewall
- Settings → Privacy & Security → Camera
- Enable app access to camera

#### 5. Face Not Detected
```powershell
python test_camera.py
```
- Check lighting (run test)
- Position face clearly
- Check resolution support

#### 6. "Address already in use"
```powershell
# Kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Or change port in app.py
app.run(host='0.0.0.0', port=5001)
```

#### 7. Slow Performance
- Close background apps
- Check CPU/RAM usage
- Update GPU drivers
- Run in 640×480 resolution

#### 8. Session Expired / Can't Login
- Clear browser cookies
- Restart the Flask server
- Ensure credentials: `ADMIN` / `ADMIN123` (case-sensitive)

---

## 📊 Performance Metrics

| Metric | Expected | Notes |
|--------|----------|-------|
| Face Detection | 20-30 FPS | Real-time |
| Face Recognition | <100ms | Per face |
| Registration | ~5 seconds | 30 samples |
| Database Queries | <50ms | SQLite |
| CSV Logging | <10ms | Auto-saved |
| Login Auth | <10ms | Session-based |

---

## 🛡️ Security Features

- ✅ Admin-only login system (session-based)
- ✅ Protected registration & admin routes
- ✅ Unauthenticated API requests return 401
- ✅ Liveness detection (eye movement)
- ✅ Duplicate marking prevention
- ✅ Configurable confidence thresholds
- ✅ Thread-safe operations
- ✅ Input validation & error handling
- ✅ No face images in logs (only vectors & metadata)

---

## 📝 License

This project uses:
- **Flask**: BSD License
- **OpenCV**: Apache 2 License
- **YuNet & SFace**: Apache 2 License (OpenCV Zoo)
- **NumPy**: BSD License
- **Pillow**: HPND License

---

## 🤝 Support

**Issues?**
1. Run: `python test_camera.py`
2. Check browser console (F12)
3. Check app.py console output
4. Review logs in `attendance/` folder

**Feature Requests?**
- Modify `config.json`
- Update thresholds via Admin panel (login required)
- Customize UI in `static/` folder

---

## 🎯 Quick Reference

```
┌─────────────────────────────────────────────┐
│         ADMIN CREDENTIALS                    │
│         Username:  ADMIN                     │
│         Password:  ADMIN123                  │
├─────────────────────────────────────────────┤
│  FOR USERS:                                  │
│    → Go to /attendance                       │
│    → Show face to camera                     │
│    → Attendance marked automatically!        │
├─────────────────────────────────────────────┤
│  FOR ADMIN:                                  │
│    → Login at /login                         │
│    → Register users at /register             │
│    → Manage system at /admin                 │
│    → Logout when done                        │
└─────────────────────────────────────────────┘
```

---

**Last Updated:** February 26, 2026  
**Version:** 2.0.0  
**Status:** Production Ready ✅
