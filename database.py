import sqlite3
import os
import csv
import threading
from datetime import datetime, timedelta

try:
    from config import get_config
except ImportError:
    def get_config():
        return {'auto_save_attendance': True, 'duplicate_check_timeout': 5}

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_attendance.db')
ATTENDANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attendance')

# File-level lock for CSV write safety
_csv_lock = threading.Lock()


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance log table for audit trail
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            status TEXT DEFAULT 'marked',
            date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    os.makedirs(os.path.join(os.path.dirname(__file__), 'attendance'), exist_ok=True)


def add_user(name):
    """Add a new user and return their ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_user(user_id):
    """Get a user by their ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def get_all_users():
    """Get all registered users."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users ORDER BY id')
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def get_user_count():
    """Get total number of registered users."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM users')
    count = cursor.fetchone()['count']
    conn.close()
    return count


def user_name_exists(name):
    """Check if a user with the given name already exists (case-insensitive)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM users WHERE LOWER(name) = LOWER(?)', (name,))
    count = cursor.fetchone()['count']
    conn.close()
    return count > 0


def delete_user(user_id):
    """Delete a user from the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def mark_attendance(user_id, name, confidence):
    """Mark attendance for a user in today's CSV file and database (thread-safe)."""
    config = get_config()
    today = datetime.now().strftime('%Y-%m-%d')
    csv_path = os.path.join(ATTENDANCE_DIR, f'{today}.csv')
    now = datetime.now().strftime('%H:%M:%S')
    duplicate_timeout = config.get('duplicate_check_timeout', 5)

    with _csv_lock:
        # Check if already marked TODAY
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['user_id'] == str(user_id):
                        return False, "Already marked today"  # Already marked

        # Check duplicate within timeout window (last N seconds)
        conn = get_db()
        cursor = conn.cursor()
        cutoff_time = (datetime.now() - timedelta(seconds=duplicate_timeout)).isoformat()
        cursor.execute(
            'SELECT timestamp FROM attendance_log WHERE user_id = ? AND timestamp > ? ORDER BY timestamp DESC LIMIT 1',
            (user_id, cutoff_time)
        )
        recent = cursor.fetchone()
        conn.close()
        
        if recent:
            return False, f"Already marked {duplicate_timeout}s ago"

        # Write to CSV
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['user_id', 'name', 'time', 'confidence', 'date'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'user_id': user_id,
                'name': name,
                'time': now,
                'confidence': f'{confidence:.1f}',
                'date': today
            })

        # Log to database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO attendance_log (user_id, name, confidence, status, date) VALUES (?, ?, ?, ?, ?)',
            (user_id, name, confidence, 'marked', today)
        )
        conn.commit()
        conn.close()

        if config.get('enable_logs', True):
            print(f"  [✓] Attendance marked: {name} ({confidence:.1f}%)")

    return True, "Attendance marked successfully"


def get_attendance_records(date=None):
    """Get attendance records for a specific date (default: today)."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    csv_path = os.path.join(ATTENDANCE_DIR, f'{date}.csv')
    records = []

    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            records = list(reader)

    return records


def get_today_attendance_count():
    """Get count of attendance records for today."""
    records = get_attendance_records()
    return len(records)


def get_available_dates():
    """Get list of dates that have attendance records."""
    dates = []
    if os.path.exists(ATTENDANCE_DIR):
        for filename in sorted(os.listdir(ATTENDANCE_DIR), reverse=True):
            if filename.endswith('.csv'):
                dates.append(filename.replace('.csv', ''))
    return dates


# ── Admin Statistics ────────────────────────────────────────────────────────

def get_attendance_stats():
    """Get comprehensive attendance statistics."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Total attendance count
    cursor.execute('SELECT COUNT(*) as count FROM attendance_log')
    total_count_row = cursor.fetchone()
    total_count = total_count_row['count'] if total_count_row else 0
    
    # Users with most attendance
    cursor.execute('''
        SELECT user_id, name, COUNT(*) as count 
        FROM attendance_log 
        GROUP BY user_id 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    top_users = [dict(row) for row in cursor.fetchall()]
    
    # Today's attendance
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('SELECT COUNT(*) as count FROM attendance_log WHERE date = ?', (today,))
    today_count_row = cursor.fetchone()
    today_count = today_count_row['count'] if today_count_row else 0
    
    # Average confidence
    cursor.execute('SELECT AVG(confidence) as avg_conf FROM attendance_log WHERE confidence IS NOT NULL')
    avg_conf_row = cursor.fetchone()
    avg_confidence = avg_conf_row['avg_conf'] if avg_conf_row and avg_conf_row['avg_conf'] else 0
    
    # Total users
    cursor.execute('SELECT COUNT(*) as count FROM users')
    total_users_row = cursor.fetchone()
    total_users = total_users_row['count'] if total_users_row else 0
    
    conn.close()
    
    return {
        'total_attendance_records': total_count,
        'today_attendance': today_count,
        'total_users': total_users,
        'avg_confidence': round(avg_confidence, 2),
        'top_users': top_users,
    }


def get_user_statistics(user_id):
    """Get attendance statistics for a specific user."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return None
    
    cursor.execute('SELECT COUNT(*) as count FROM attendance_log WHERE user_id = ?', (user_id,))
    total_count_row = cursor.fetchone()
    total_count = total_count_row['count'] if total_count_row else 0
    
    cursor.execute('SELECT AVG(confidence) as avg_conf FROM attendance_log WHERE user_id = ? AND confidence IS NOT NULL', (user_id,))
    avg_conf_row = cursor.fetchone()
    avg_confidence = avg_conf_row['avg_conf'] if avg_conf_row and avg_conf_row['avg_conf'] else 0
    
    cursor.execute('''
        SELECT timestamp FROM attendance_log 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''', (user_id,))
    last_attendance_row = cursor.fetchone()
    last_attendance = last_attendance_row['timestamp'] if last_attendance_row else None
    
    conn.close()
    
    return {
        'user_id': user_id,
        'name': user['name'],
        'total_attendance': total_count,
        'avg_confidence': round(avg_confidence, 2),
        'last_attendance': last_attendance,
    }


def export_attendance_report(start_date=None, end_date=None):
    """Export attendance records for date range to CSV."""
    if start_date is None:
        # Default: last 30 days
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, name, date, GROUP_CONCAT(timestamp, '; ') as times, 
               COUNT(*) as attendance_count, AVG(confidence) as avg_confidence
        FROM attendance_log 
        WHERE date BETWEEN ? AND ?
        GROUP BY user_id, name, date
        ORDER BY date DESC, user_id
    ''', (start_date, end_date))
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return records
