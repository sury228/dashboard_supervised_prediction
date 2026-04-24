/* ═══════════════════════════════════════════════════
   ML Dashboard – Main JavaScript
   ═══════════════════════════════════════════════════ */

// ── Sidebar toggle ────────────────────────────────
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebar = document.getElementById("sidebar");
if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  document.addEventListener("click", (e) => {
    if (window.innerWidth <= 768 && !sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
      sidebar.classList.remove("open");
    }
  });
}

// ── Stat counter animation ────────────────────────
function animateCounters() {
  document.querySelectorAll("[data-count]").forEach(el => {
    const target = +el.dataset.count;
    const duration = 1200;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(target * eased);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

// ── Intersection Observer for fade-in ─────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = "running";
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".feature-card, .stat-card, .step").forEach(el => {
  el.style.animationPlayState = "paused";
  observer.observe(el);
});

// ── Toast notifications ───────────────────────────
function showToast(message, type = "info") {
  const container = document.querySelector(".flash-container") || document.body;
  const icons = {
    danger: "bi-exclamation-triangle-fill",
    warning: "bi-exclamation-circle-fill",
    success: "bi-check-circle-fill",
    info: "bi-info-circle-fill"
  };
  const alert = document.createElement("div");
  alert.className = `alert alert-${type} alert-dismissible fade show`;
  alert.setAttribute("role", "alert");
  alert.innerHTML = `<i class="bi ${icons[type] || icons.info} me-2"></i>${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
  container.prepend(alert);
  setTimeout(() => { if (alert.parentNode) alert.remove(); }, 5000);
}

// ── Loading overlay helpers ───────────────────────
function showLoading(msg) {
  const overlay = document.getElementById("loading-overlay");
  const msgEl = document.getElementById("loading-msg");
  if (overlay) { overlay.classList.remove("d-none"); }
  if (msgEl && msg) { msgEl.textContent = msg; }
}
function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.add("d-none");
}

// ── Init on DOM ready ─────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  animateCounters();
});