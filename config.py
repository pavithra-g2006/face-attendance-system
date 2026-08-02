"""
Configuration Management for FaceTrack
Centralized settings for face recognition and attendance system
"""

import os
import json
from datetime import datetime

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Default configuration
DEFAULT_CONFIG = {
    # Face recognition thresholds
    'cosine_threshold': 0.363,              # Cosine similarity threshold (0.0-1.0)
    'detection_score_threshold': 0.6,       # YuNet detection confidence (0.0-1.0)
    'nms_threshold': 0.3,                   # Non-maximum suppression threshold
    
    # Liveness detection
    'liveness_enabled': True,               # Enable liveness detection
    'liveness_threshold': 5,                # Minimum eye movements required
    'liveness_frames': 30,                  # Number of frames to analyze
    
    # Registration settings
    'total_samples': 30,                    # Face samples per registration
    'auto_save_attendance': True,           # Auto-save to CSV on match
    
    # Security
    'min_confidence_for_attendance': 0.363, # Minimum confidence to mark attendance
    'duplicate_check_timeout': 5,           # Seconds before same person can be marked again
    
    # Logging
    'enable_logs': True,
    'log_attendance': True,
}


def load_config():
    """Load configuration from file, or create default if not exists."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults (defaults take precedence for missing keys)
            config = {**DEFAULT_CONFIG, **user_config}
            return config
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [!] Error loading config: {e}. Using defaults.")
            return DEFAULT_CONFIG.copy()
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  [OK] Configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"  [!] Error saving config: {e}")
        return False


def get_config():
    """Get current configuration (cached)."""
    if not hasattr(get_config, '_cache'):
        get_config._cache = load_config()
    return get_config._cache


def set_config_value(key, value):
    """Update a single configuration value and refresh cache."""
    config = get_config()
    if key in config:
        config[key] = value
        save_config(config)
        # Ensure the in-memory cache is updated
        get_config._cache = config
        return True
    return False


def reset_config():
    """Reset to default configuration."""
    config = DEFAULT_CONFIG.copy()
    save_config(config)
    get_config._cache = config
    return True


# Initialize config on import
def init_config():
    """Initialize configuration on application startup."""
    config = load_config()
    print("=== Configuration Loaded ===")
    print(f"  Cosine Threshold:         {config['cosine_threshold']}")
    print(f"  Liveness Detection:       {config['liveness_enabled']}")
    print(f"  Auto-save Attendance:     {config['auto_save_attendance']}")
    print("=" * 27)
    return config
