document.addEventListener("DOMContentLoaded", () => {
    // Initialize all tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))

    // Initialize all offcanvas
    const offcanvasElementList = document.querySelectorAll('.offcanvas')
    const offcanvasList = [...offcanvasElementList].map((offcanvasEl) => {
	// initialize offcanvas element
	let offcanvas = new bootstrap.Offcanvas(offcanvasEl, { backdrop: true, scroll: true }); // allow body scrolling when offcanvas is open
	// add custom overflow-y style to offcanvas element
	$(offcanvasEl).css({ 'overflow-y': 'scroll' });
	$(offcanvasEl).on('shown.bs.offcanvas', function () {
	    // on offcanvas shown, adjust backdrop opacity
	    $(".offcanvas-backdrop").css({ 'opacity': '0.001' }); // adjust backdrop opacity
	});
	return offcanvas;
    });

});
