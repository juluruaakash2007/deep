/**
 * DeepShield — Shared API client + utilities
 * All pages import this file first.
 */

const API_BASE = '';

// ── Token management ─────────────────────────────────────────
const Auth = {
  getToken: () => localStorage.getItem('ds_token'),
  getUser:  () => JSON.parse(localStorage.getItem('ds_user') || 'null'),
  setSession(token, user) {
    localStorage.setItem('ds_token', token);
    localStorage.setItem('ds_user', JSON.stringify(user));
  },
  clear() {
    localStorage.removeItem('ds_token');
    localStorage.removeItem('ds_user');
  },
  isLoggedIn: () => !!localStorage.getItem('ds_token'),
  requireAuth() {
    if (!this.isLoggedIn()) {
      window.location.href = '/auth.html';
      return false;
    }
    return true;
  }
};

// ── Fetch wrapper ────────────────────────────────────────────
async function api(path, options = {}) {
  const token = Auth.getToken();
  const headers = { ...options.headers };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(API_BASE + path, { ...options, headers });

  if (res.status === 401) {
    Auth.clear();
    window.location.href = '/auth.html';
    return;
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

async function apiForm(path, formData) {
  const token = Auth.getToken();
  const headers = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(API_BASE + path, { method: 'POST', headers, body: formData });
  if (res.status === 401) { Auth.clear(); window.location.href = '/auth.html'; return; }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ── Toast notifications ──────────────────────────────────────
function toast(message, type = 'info', duration = 4000) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateX(110%)';
    setTimeout(() => el.remove(), 300); }, duration);
}

// ── Format helpers ───────────────────────────────────────────
function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
}
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/1024/1024).toFixed(1) + ' MB';
}
function predictionColor(p) {
  return { FAKE: '#ef4444', REAL: '#10b981', SUSPICIOUS: '#f59e0b' }[p] || '#888';
}
function predictionIcon(p) {
  return { FAKE: '🔴', REAL: '🟢', SUSPICIOUS: '🟡' }[p] || '⚪';
}
function riskBadge(risk) {
  const map = { CRITICAL:'badge--fake', HIGH:'badge--fake', MEDIUM:'badge--suspicious', LOW:'badge--real' };
  return `<span class="badge ${map[risk] || ''}">${risk}</span>`;
}
function mediaBadge(type) {
  return `<span class="badge badge--${type.toLowerCase()}">${type === 'image' ? '🖼️' : type === 'video' ? '🎬' : '🎵'} ${type}</span>`;
}

// ── Populate navbar user info ─────────────────────────────────
function initNavbar(activePage) {
  const user = Auth.getUser();
  if (!user) return;

  const avatarEl = document.getElementById('nav-avatar');
  const nameEl   = document.getElementById('nav-username');
  if (avatarEl) avatarEl.textContent = user.name ? user.name[0].toUpperCase() : 'U';
  if (nameEl)   nameEl.textContent   = user.name || 'User';

  document.querySelectorAll('.navbar__link').forEach(l => {
    if (l.dataset.page === activePage) l.classList.add('active');
  });

  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) logoutBtn.addEventListener('click', () => {
    Auth.clear();
    window.location.href = '/auth.html';
  });
}
