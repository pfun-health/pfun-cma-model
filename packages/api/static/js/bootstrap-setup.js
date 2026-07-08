document.addEventListener("DOMContentLoaded", () => {
    // Initialize all tooltips
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((tooltipTriggerEl) => {
        new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // Initialize all offcanvas
    document.querySelectorAll('.offcanvas').forEach((offcanvasEl) => {
	// initialize offcanvas element
	new bootstrap.Offcanvas(offcanvasEl, { backdrop: true, scroll: true }); // allow body scrolling when offcanvas is open
	// add custom overflow-y style to offcanvas element
	$(offcanvasEl).css({ 'overflow-y': 'scroll' });
	$(offcanvasEl).on('shown.bs.offcanvas', function () {
	    // on offcanvas shown, adjust backdrop opacity
	    $(".offcanvas-backdrop").css({ 'opacity': '0.001' }); // adjust backdrop opacity
	});
    });

});
