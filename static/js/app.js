/**
 * FaceTrack - Frontend JavaScript
 * Handles toast notifications and shared UI interactions
 */

// ── Toast Notifications ──────────────────────────

function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        // Auto-create toast container if not in the DOM
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = '';
    switch (type) {
        case 'success':
            icon = '<svg viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" width="20" height="20"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
            break;
        case 'error':
            icon = '<svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" width="20" height="20"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
            break;
        case 'warning':
            icon = '<svg viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" width="20" height="20"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
            break;
        default:
            icon = '<svg viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2" width="20" height="20"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }

    // Use escapeHtml if available (from utils.js) for XSS safety
    const safeMessage = typeof escapeHtml === 'function' ? escapeHtml(message) : message;
    toast.innerHTML = `${icon}<span>${safeMessage}</span>`;
    container.appendChild(toast);

    // Remove after animation
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3000);
}

// ── Auto-refresh webcam feed on error ────────────

document.addEventListener('DOMContentLoaded', () => {
    const webcamFeed = document.getElementById('webcam-feed');
    if (webcamFeed) {
        webcamFeed.onerror = () => {
            setTimeout(() => {
                const src = webcamFeed.src;
                webcamFeed.src = '';
                webcamFeed.src = src;
            }, 2000);
        };
    }
});
