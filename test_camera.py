"""
Camera Diagnostics Tool - FaceTrack
Tests camera compatibility and performance with face recognition system
"""

import cv2
import time
import numpy as np


def test_camera_available():
    """Test if camera is detected by the system."""
    print("\n" + "="*60)
    print("  CAMERA AVAILABILITY TEST")
    print("="*60)
    
    backends = [
        (0, cv2.CAP_DSHOW, "DirectShow"),
        (0, cv2.CAP_MSMF, "MSMF"),
        (1, cv2.CAP_DSHOW, "DirectShow (USB)"),
        (1, cv2.CAP_MSMF, "MSMF (USB)"),
    ]
    
    found = False
    for idx, backend, name in backends:
        try:
            cam = cv2.VideoCapture(idx, backend)
            if cam.isOpened():
                print(f"  ✓ Camera found on index {idx} ({name})")
                cam.release()
                found = True
            else:
                cam.release()
        except Exception as e:
            print(f"  ✗ Error testing index {idx} ({name}): {e}")
    
    if not found:
        print("  ✗ NO CAMERA DETECTED")
        return False
    
    return True


def test_camera_resolution():
    """Test if camera supports required resolutions."""
    print("\n" + "="*60)
    print("  CAMERA RESOLUTION TEST")
    print("="*60)
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("  ✗ Could not open camera")
        return False
    
    resolutions = [
        (640, 480),
        (800, 600),
        (960, 720),
        (1280, 720),
        (1920, 1080),
    ]
    
    print("  Testing resolutions:")
    supported = []
    
    for width, height in resolutions:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w == width and actual_h == height:
            print(f"    ✓ {width}x{height}")
            supported.append((width, height))
        else:
            print(f"    ✗ {width}x{height} (camera set to {actual_w}x{actual_h})")
    
    cam.release()
    return len(supported) > 0


def test_camera_fps():
    """Test camera frame rate performance."""
    print("\n" + "="*60)
    print("  CAMERA FPS TEST")
    print("="*60)
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("  ✗ Could not open camera")
        return False
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    # Warm up
    for _ in range(10):
        cam.read()
    
    print("  Measuring actual FPS (10 frames)...")
    start = time.time()
    frame_count = 0
    
    while frame_count < 10:
        ret, frame = cam.read()
        if ret:
            frame_count += 1
        else:
            print("  ✗ Failed to read frames")
            cam.release()
            return False
    
    elapsed = time.time() - start
    actual_fps = frame_count / elapsed
    
    print(f"  Actual FPS: {actual_fps:.2f} FPS")
    
    if actual_fps >= 20:
        print("  ✓ FPS is acceptable (≥20 FPS)")
        result = True
    elif actual_fps >= 10:
        print("  ⚠ FPS is slow (10-20 FPS) - may affect real-time performance")
        result = True
    else:
        print("  ✗ FPS is too slow (<10 FPS)")
        result = False
    
    cam.release()
    return result


def test_face_detection():
    """Test if YuNet face detector works with camera."""
    print("\n" + "="*60)
    print("  FACE DETECTION TEST")
    print("="*60)
    
    try:
        # Check if model files exist
        import os
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        detection_model = os.path.join(models_dir, 'face_detection_yunet_2023mar.onnx')
        
        if not os.path.exists(detection_model):
            print(f"  ✗ Face detection model not found at {detection_model}")
            print("    Run the application once to auto-download models")
            return False
        
        # Try to create detector
        detector = cv2.FaceDetectorYN.create(
            detection_model, "", (640, 480),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        
        if detector is None:
            print("  ✗ Failed to create face detector")
            return False
        
        print("  ✓ Face detector initialized")
        
        # Try detection on camera feed
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            print("  ✗ Could not open camera")
            return False
        
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("  Attempting face detection on camera feed...")
        print("  (Position your face in front of the camera)")
        
        faces_detected = 0
        frames_processed = 0
        start_time = time.time()
        
        while time.time() - start_time < 10:  # 10 second timeout
            ret, frame = cam.read()
            if not ret:
                continue
            
            frames_processed += 1
            _, faces = detector.detect(frame)
            
            if faces is not None and len(faces) > 0:
                faces_detected += 1
                print(f"  ✓ Face detected! ({len(faces)} face(s))")
                break
            
            if frames_processed % 10 == 0:
                print(f"    ... checking ({frames_processed} frames processed)")
        
        cam.release()
        
        if faces_detected > 0:
            print("  ✓ Face detection working correctly")
            return True
        else:
            print(f"  ⚠ No faces detected in {frames_processed} frames")
            print("    Make sure your face is clearly visible and well-lit")
            return False
    
    except Exception as e:
        print(f"  ✗ Error during face detection test: {e}")
        return False


def test_lighting_conditions():
    """Test if current lighting is suitable for face recognition."""
    print("\n" + "="*60)
    print("  LIGHTING CONDITIONS TEST")
    print("="*60)
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("  ✗ Could not open camera")
        return False
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warm up
    for _ in range(5):
        cam.read()
    
    print("  Analyzing brightness levels...")
    brightness_values = []
    
    for _ in range(30):
        ret, frame = cam.read()
        if ret:
            # Convert to grayscale and measure brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
    
    cam.release()
    
    if not brightness_values:
        print("  ✗ Could not read frames")
        return False
    
    avg_brightness = np.mean(brightness_values)
    
    print(f"  Average brightness: {avg_brightness:.1f}/255")
    
    if avg_brightness < 30:
        print("  ✗ Too dark - increase lighting")
        print("    Recommended: Place face near light source")
        return False
    elif avg_brightness < 60:
        print("  ⚠ Dim lighting - may affect face detection")
        print("    Recommended: Increase ambient light")
        return True
    elif avg_brightness > 200:
        print("  ⚠ Very bright - may cause glare")
        print("    Recommended: Reduce direct light on face")
        return True
    else:
        print("  ✓ Lighting is good")
        return True


def generate_report():
    """Generate comprehensive camera compatibility report."""
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "    FACETRACK - CAMERA COMPATIBILITY REPORT".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {}
    
    # Run all tests
    results['availability'] = test_camera_available()
    
    if results['availability']:
        results['resolution'] = test_camera_resolution()
        results['fps'] = test_camera_fps()
        results['lighting'] = test_lighting_conditions()
        results['detection'] = test_face_detection()
    else:
        print("\n  Cannot proceed with other tests - camera not available")
        results['resolution'] = False
        results['fps'] = False
        results['lighting'] = False
        results['detection'] = False
    
    # Summary
    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    tests = [
        ('Camera Availability', results['availability']),
        ('Resolution Support', results['resolution']),
        ('Frame Rate Performance', results['fps']),
        ('Lighting Conditions', results['lighting']),
        ('Face Detection', results['detection']),
    ]
    
    all_pass = True
    for test_name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:.<40} {status}")
        if not result:
            all_pass = False
    
    print("="*60)
    
    if all_pass:
        print("\n  ✓ YOUR SYSTEM IS READY FOR FACE RECOGNITION!")
        print("    You can now run the FaceTrack application.")
    else:
        print("\n  ⚠ SOME TESTS FAILED")
        print("    See above for specific issues and recommendations.")
        print("    Common fixes:")
        print("    - Ensure camera is connected and not in use by other apps")
        print("    - Increase ambient lighting")
        print("    - Install latest camera drivers")
        print("    - Check if Windows firewall is blocking camera access")
    
    print("\n")


if __name__ == '__main__':
    generate_report()
    input("Press Enter to exit...")
