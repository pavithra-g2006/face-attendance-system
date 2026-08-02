"""
FaceTrack - Configuration Optimizer
Automatically suggests optimal settings based on your scenario
"""

import json
import os

class ConfigOptimizer:
    """Suggests optimal configurations for different scenarios."""
    
    PRESETS = {
        "home": {
            "name": "Home / Personal Use",
            "description": "Single or few trusted users, controlled lighting",
            "config": {
                "cosine_threshold": 0.35,
                "liveness_enabled": False,
                "detection_score_threshold": 0.60,
                "duplicate_check_timeout": 5,
                "total_samples": 30,
                "nms_threshold": 0.3,
                "enable_logs": True,
            },
            "notes": [
                "✓ Relaxed thresholds for convenience",
                "✓ Liveness can be disabled for speed",
                "✓ Works with basic lighting",
            ]
        },
        
        "office": {
            "name": "Corporate Office",
            "description": "Professional environment, controlled conditions, trained staff",
            "config": {
                "cosine_threshold": 0.38,
                "liveness_enabled": True,
                "detection_score_threshold": 0.60,
                "duplicate_check_timeout": 5,
                "total_samples": 30,
                "nms_threshold": 0.3,
                "enable_logs": True,
            },
            "notes": [
                "✓ Balanced accuracy and speed",
                "✓ Liveness enabled for security",
                "✓ Standard professional settings",
            ]
        },
        
        "school": {
            "name": "School / College",
            "description": "Multiple students, variable lighting, seasonal changes",
            "config": {
                "cosine_threshold": 0.35,
                "liveness_enabled": True,
                "detection_score_threshold": 0.55,
                "duplicate_check_timeout": 10,
                "total_samples": 40,
                "nms_threshold": 0.3,
                "enable_logs": True,
            },
            "notes": [
                "✓ Lower cosine_threshold for variety",
                "✓ More samples for robustness",
                "✓ Longer duplicate timeout",
                "✓ Slightly lower detection threshold",
            ]
        },
        
        "outdoor": {
            "name": "Outdoor / Transportation",
            "description": "Uncontrolled lighting, high-throughput, security-sensitive",
            "config": {
                "cosine_threshold": 0.40,
                "liveness_enabled": True,
                "detection_score_threshold": 0.65,
                "duplicate_check_timeout": 15,
                "total_samples": 50,
                "nms_threshold": 0.25,
                "enable_logs": True,
            },
            "notes": [
                "✓ Higher cosine_threshold for security",
                "✓ More samples for outdoor variability",
                "✓ Higher detection threshold",
                "⚠ Requires proper lighting setup",
            ]
        },
        
        "security": {
            "name": "High-Security / Access Control",
            "description": "Maximum security, comprehensive logging, audit trail",
            "config": {
                "cosine_threshold": 0.45,
                "liveness_enabled": True,
                "detection_score_threshold": 0.70,
                "duplicate_check_timeout": 30,
                "total_samples": 60,
                "nms_threshold": 0.2,
                "enable_logs": True,
                "log_attendance": True,
            },
            "notes": [
                "✓ Strictest thresholds available",
                "✓ Most samples required",
                "✓ Maximum security margin",
                "✓ Full comprehensive logging",
            ]
        },
        
        "low_light": {
            "name": "Low Light / Night Operations",
            "description": "Dim lighting, needs to be more permissive",
            "config": {
                "cosine_threshold": 0.32,
                "liveness_enabled": False,
                "detection_score_threshold": 0.50,
                "duplicate_check_timeout": 5,
                "total_samples": 50,
                "nms_threshold": 0.35,
                "enable_logs": True,
            },
            "notes": [
                "⚠ REQUIRES good infrared or additional lighting",
                "✓ Lowered thresholds for dim conditions",
                "✓ Liveness disabled due to low brightness",
                "✓ More samples to compensate",
                "✗ Currently not well-supported",
            ]
        },
        
        "strict": {
            "name": "Strict / No False Positives",
            "description": "Critical application, prevent wrong person marking",
            "config": {
                "cosine_threshold": 0.50,
                "liveness_enabled": True,
                "detection_score_threshold": 0.75,
                "duplicate_check_timeout": 30,
                "total_samples": 60,
                "nms_threshold": 0.2,
                "enable_logs": True,
            },
            "notes": [
                "✓ Extremely strict thresholds",
                "✓ Some legitimate users may be rejected",
                "✓ Priority: no false positives",
                "✗ Expect 5-10% false negative rate",
            ]
        },
        
        "fast": {
            "name": "Speed-Optimized / High Throughput",
            "description": "Maximize speed, acceptable accuracy reduction",
            "config": {
                "cosine_threshold": 0.30,
                "liveness_enabled": False,
                "detection_score_threshold": 0.45,
                "duplicate_check_timeout": 3,
                "total_samples": 20,
                "nms_threshold": 0.4,
                "enable_logs": False,
            },
            "notes": [
                "✓ Fastest processing possible",
                "✓ Minimal samples required",
                "✗ Higher false positive rate expected",
                "✗ Not recommended for security",
            ]
        },
    }
    
    THRESHOLD_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════╗
║              COSINE THRESHOLD TUNING GUIDE                            ║
║═══════════════════════════════════════════════════════════════════════║

Cosine Threshold: Similarity score required to match (0.0 - 1.0)
Higher = More strict (fewer false positives, more false negatives)
Lower = More permissive (more false positives, fewer false negatives)

THRESHOLD VALUES:
═════════════════════════════════════════════════════════════════════════
  0.30 - 0.35 │ Very Permissive
              │ • Pros: Catches most users even with changes
              │ • Cons: Higher false positive rate (5-10%)
              │ • Use: Home, speed-critical, training phase
              │
  0.35 - 0.40 │ Balanced (RECOMMENDED)
              │ • Pros: Good balance of accuracy and coverage
              │ • Cons: Minimal false positives (1-2%)
              │ • Use: Office, school, general purpose
              │
  0.40 - 0.50 │ Strict
              │ • Pros: Very few false positives (<1%)
              │ • Cons: Some legitimate users rejected (2-5%)
              │ • Use: Security access, high-stakes
              │
  0.50+ │ Very Strict
            │ • Pros: Almost no false positives
            │ • Cons: Some users may not work (5-15%)
            │ • Use: Maximum security only
            │
═════════════════════════════════════════════════════════════════════════

WHAT TO ADJUST IF:
  Wrong person marked (false positive)
    → INCREASE cosine_threshold by 0.02
    → Example: 0.35 → 0.37
    → Effect: Stricter matching

  Registered person not recognized (false negative)
    → DECREASE cosine_threshold by 0.02
    → Example: 0.38 → 0.36
    → Effect: More permissive matching

  Appearance changes not recognized (new glasses, haircut)
    → DECREASE cosine_threshold by 0.03-0.05
    → Reason: Encodings shift with appearance change
    → OR: Re-register user with new appearance

╚═══════════════════════════════════════════════════════════════════════╝
"""

    DETECTION_THRESHOLD_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════╗
║           DETECTION SCORE THRESHOLD TUNING GUIDE                      ║
║═══════════════════════════════════════════════════════════════════════║

Detection Threshold: Confidence needed to detect a face (0.0 - 1.0)
Higher = More selective (fewer detections, fewer false detections)
Lower = More lenient (more detections, more false detections)

THRESHOLD VALUES:
═════════════════════════════════════════════════════════════════════════
  0.45 - 0.55 │ Permissive
              │ • Detects faces in poor conditions
              │ • May have false detections
              │ • Use: Dark environments, far distances
              │
  0.55 - 0.65 │ Standard (RECOMMENDED)
              │ • Good balance for most scenarios
              │ • ~99% detection of valid faces
              │ • Use: Office, school, standard setups
              │
  0.65 - 0.75 │ Strict
              │ • Only detects clear, frontal faces
              │ • <1% false detections
              │ • Use: Security, clarity required
              │
  0.75+ │ Very Strict
           │ • Only perfect faces detected
           │ • May miss angled or partial faces
           │ • Use: High precision only
           │
═════════════════════════════════════════════════════════════════════════

WHAT TO ADJUST IF:
  "Face detected: 0" messages (face not detected)
    → DECREASE detection_score_threshold by 0.05
    → Example: 0.60 → 0.55
    → Effect: Easier detection

  Too many false detections (non-faces detected)
    → INCREASE detection_score_threshold by 0.05
    → Example: 0.60 → 0.65
    → Effect: Stricter detection

  Performance too slow
    → INCREASE detection_score_threshold slightly
    → Fewer faces to process = faster
    → But may miss some valid faces

╚═══════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def get_preset(scenario):
        """Get preset configuration for a scenario."""
        if scenario not in ConfigOptimizer.PRESETS:
            return None
        return ConfigOptimizer.PRESETS[scenario]
    
    @staticmethod
    def list_presets():
        """List all available presets."""
        print("\nAvailable Configuration Presets:\n")
        for key, preset in ConfigOptimizer.PRESETS.items():
            print(f"  {key:12} - {preset['name']}")
            print(f"               {preset['description']}")
            print()
    
    @staticmethod
    def apply_preset(preset_name, config_path="config.json"):
        """Apply a preset configuration."""
        preset = ConfigOptimizer.get_preset(preset_name)
        if not preset:
            print(f"✗ Preset '{preset_name}' not found")
            return False
        
        try:
            with open(config_path, 'w') as f:
                json.dump(preset['config'], f, indent=2)
            
            print(f"✓ Applied preset: {preset['name']}")
            print(f"  Configuration saved to {config_path}")
            print(f"\n{preset['description']}")
            print(f"\nNotes:")
            for note in preset['notes']:
                print(f"  {note}")
            return True
        
        except Exception as e:
            print(f"✗ Error applying preset: {e}")
            return False
    
    @staticmethod
    def show_guides():
        """Display all tuning guides."""
        print(ConfigOptimizer.THRESHOLD_GUIDE)
        print("\n")
        print(ConfigOptimizer.DETECTION_THRESHOLD_GUIDE)


TROUBLESHOOTING_MATRIX = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    TROUBLESHOOTING DECISION TREE                       ║
╚═══════════════════════════════════════════════════════════════════════╝

PROBLEM: System won't start
├─ Error: "No module named opencv"
│  └─ SOLUTION: pip install -r requirements.txt
├─ Error: "No camera detected"
│  └─ SOLUTION: Run test_camera.py for diagnostics
└─ Error: "Database locked"
   └─ SOLUTION: Close other instances, restart app

PROBLEM: Face not detected during registration
├─ Brightness too low (< 30)
│  └─ SOLUTION: Increase lighting, move lamp closer
├─ Face too far (> 100cm)
│  └─ SOLUTION: Move closer, 40-60cm optimal
├─ Face angled > 30°
│  └─ SOLUTION: Look directly at camera
└─ Detection threshold too high
   └─ SOLUTION: Lower detection_score_threshold to 0.55

PROBLEM: Face detected but not recognized (Unknown)
├─ User not registered
│  └─ SOLUTION: Register on /register page
├─ Cosine threshold too high
│  └─ SOLUTION: Lower from 0.40 → 0.35
├─ Appearance changed significantly
│  └─ SOLUTION: Re-register user
└─ Lighting different from registration
   └─ SOLUTION: Re-register in similar lighting

PROBLEM: Wrong person marked (false positive)
├─ Cosine threshold too low
│  └─ SOLUTION: Increase from 0.35 → 0.38
├─ Liveness disabled
│  └─ SOLUTION: Enable liveness_enabled: true
└─ Too many registrations with similar faces
   └─ SOLUTION: Increase cosine_threshold gradually

PROBLEM: System very slow (lag 3-5 seconds)
├─ Old computer
│  └─ SOLUTION: Close background apps, lower resolution
├─ High resolution capture
│  └─ SOLUTION: Set resolution to 640×480 in config
├─ HDD usage
│  └─ SOLUTION: Run on SSD if possible
└─ Multiple processes running
   └─ SOLUTION: Disable Chrome, Zoom, etc.

PROBLEM: Duplicate attendance marks
├─ Same person recognized twice in < 5 seconds
│  └─ SOLUTION: Increase duplicate_check_timeout to 10
└─ Data entry error
   └─ SOLUTION: Check CSV file in /attendance/

PROBLEM: Liveness detection failing
├─ Eyes not visible (sunglasses)
│  └─ SOLUTION: Remove glasses for registration
├─ Lighting too dark
│  └─ SOLUTION: Increase brightness to see eyes
└─ Liveness threshold too high
   └─ SOLUTION: Lower liveness_threshold to 3

PROBLEM: Camera shows "No camera detected" on page
├─ Camera disconnected
│  └─ SOLUTION: Plug in camera, refresh page
├─ Another app using camera
│  └─ SOLUTION: Close Zoom, Teams, Discord etc.
└─ Driver issue
   └─ SOLUTION: Update camera drivers, restart

═══════════════════════════════════════════════════════════════════════════
For more help:
  1. Check console output for error messages
  2. Run: python test_camera.py (comprehensive diagnostics)
  3. Review logs in database
  4. Check brightness: python test_camera.py shows brightness level
═══════════════════════════════════════════════════════════════════════════
"""


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("  FaceTrack - Configuration Optimizer")
    print("="*70)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            ConfigOptimizer.list_presets()
        elif command == "apply" and len(sys.argv) > 2:
            ConfigOptimizer.apply_preset(sys.argv[2])
        elif command == "guide":
            ConfigOptimizer.show_guides()
        elif command == "troubleshoot":
            print(TROUBLESHOOTING_MATRIX)
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python optimize_config.py list           - List presets")
            print("  python optimize_config.py apply <preset> - Apply preset")
            print("  python optimize_config.py guide          - Show tuning guides")
            print("  python optimize_config.py troubleshoot   - Troubleshooting guide")
    else:
        print("\nUsage:")
        print("  python optimize_config.py list           - List presets")
        print("  python optimize_config.py apply <preset> - Apply preset")
        print("  python optimize_config.py guide          - Show tuning guides")
        print("  python optimize_config.py troubleshoot   - Troubleshooting guide")
        print("\nExample:")
        print("  python optimize_config.py apply office")
        print("\nAvailable presets:")
        ConfigOptimizer.list_presets()
