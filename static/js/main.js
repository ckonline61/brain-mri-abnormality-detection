// BrainScan AI – Main JS
document.addEventListener('DOMContentLoaded', function () {
  // Animate KPI numbers on dashboard
  document.querySelectorAll('.kpi-num').forEach(el => {
    const target = parseInt(el.textContent);
    if (!isNaN(target) && target > 0) {
      let start = 0;
      const step = Math.ceil(target / 40);
      const timer = setInterval(() => {
        start = Math.min(start + step, target);
        el.textContent = start;
        if (start >= target) clearInterval(timer);
      }, 30);
    }
  });
});
