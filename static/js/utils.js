/**
 * FaceTrack - Error Handling & Utility Functions
 * Centralized error handling and helper functions.
 * NOTE: showToast() is defined in app.js — do NOT duplicate here.
 */

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// API Error Handler
async function handleApiError(response, defaultMessage = 'Operation failed') {
    if (!response.ok) {
        try {
            const data = await response.json();
            return data.message || data.error || defaultMessage;
        } catch {
            return `Error ${response.status}: ${response.statusText}`;
        }
    }
    return null;
}

// Validation helpers
function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function validateName(name) {
    return name.length >= 2 && name.length <= 100 && /^[a-zA-Z\s'-]+$/.test(name);
}

function validateDateRange(startDate, endDate) {
    if (!startDate || !endDate) return false;
    return new Date(startDate) <= new Date(endDate);
}

// Loading state helpers
function setLoading(element, isLoading = true) {
    if (isLoading) {
        element.disabled = true;
        element.style.opacity = '0.6';
        element.innerHTML = '<span class="spinner-inline"></span> Loading...';
    } else {
        element.disabled = false;
        element.style.opacity = '1';
        // Note: Caller should restore original HTML
    }
}

// Retry logic for failed requests
async function retryFetch(url, options = {}, maxRetries = 3) {
    let lastError;
    
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fetch(url, options);
        } catch (error) {
            lastError = error;
            if (i < maxRetries - 1) {
                // Wait before retrying (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
            }
        }
    }
    
    throw lastError;
}

// Form validation
function validateFormGroup(formElement) {
    const inputs = formElement.querySelectorAll('input[required], textarea[required]');
    const errors = [];
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            errors.push(`${input.previousElementSibling?.textContent || 'Field'} is required`);
        }
    });
    
    return {
        isValid: errors.length === 0,
        errors
    };
}

// Camera permission checker
async function requestCameraPermission() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach(track => track.stop());
        return true;
    } catch (error) {
        if (error.name === 'NotAllowedError') {
            showToast('Camera permission denied. Please enable camera access in settings.', 'error');
        } else if (error.name === 'NotFoundError') {
            showToast('No camera found. Please connect a camera device.', 'error');
        } else {
            showToast(`Camera error: ${error.message}`, 'error');
        }
        return false;
    }
}

// Session management
function getSessionData(key) {
    try {
        return JSON.parse(sessionStorage.getItem(key));
    } catch {
        return null;
    }
}

function setSessionData(key, data) {
    try {
        sessionStorage.setItem(key, JSON.stringify(data));
        return true;
    } catch {
        showToast('Failed to save session data', 'warning');
        return false;
    }
}

// Debounce function for reducing API calls
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
    };
}

// Network detection
function isOnline() {
    return navigator.onLine;
}

window.addEventListener('online', () => {
    showToast('Connection restored', 'success');
});

window.addEventListener('offline', () => {
    showToast('Connection lost. Some features may not work.', 'warning');
});
