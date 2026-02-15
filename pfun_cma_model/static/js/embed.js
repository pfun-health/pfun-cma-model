// pfun_cma_model/static/js/embed.js

function showFrame(id, element) {
    // Hide all frames
    document.querySelectorAll('iframe').forEach(f => f.classList.remove('active'));
    // Remove active class from all links
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

    // Show selected
    const frame = document.getElementById(id);
    if (frame) frame.classList.add('active');
    if (element) element.classList.add('active');
}

document.addEventListener("DOMContentLoaded", () => {
    const links = document.querySelectorAll('.nav-link[data-target-frame]');
    links.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const targetId = link.getAttribute('data-target-frame');
            showFrame(targetId, link);
        });
    });
});
