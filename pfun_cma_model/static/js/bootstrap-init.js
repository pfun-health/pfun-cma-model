// pfun_cma_model/static/js/bootstrap-init.js
window.addEventListener("DOMContentLoaded", () => {
  // Initialize all tooltips
  const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

  // Initialize all offcanvas
  const offcanvasElementList = document.querySelectorAll('.offcanvas');
  const offcanvasList = [...offcanvasElementList].map((offcanvasEl) => {
    // initialize offcanvas element
    let offcanvas = new bootstrap.Offcanvas(offcanvasEl, { backdrop: true, scroll: true }); // allow body scrolling when offcanvas is open

    // add custom overflow-y style to offcanvas element
    // Use jQuery if available for consistency with original script, or vanilla JS
    if (typeof $ !== 'undefined') {
        $(offcanvasEl).css({ 'overflow-y': 'scroll' });
        $(offcanvasEl).on('shown.bs.offcanvas', function () {
            // on offcanvas shown, adjust backdrop opacity
            $(".offcanvas-backdrop").css({ 'opacity': '0.001' }); // adjust backdrop opacity
        });
    } else {
        // Vanilla JS fallback
        offcanvasEl.style.overflowY = 'scroll';
        offcanvasEl.addEventListener('shown.bs.offcanvas', () => {
             const backdrop = document.querySelector('.offcanvas-backdrop');
             if (backdrop) backdrop.style.opacity = '0.001';
        });
    }
    return offcanvas;
  });
});
