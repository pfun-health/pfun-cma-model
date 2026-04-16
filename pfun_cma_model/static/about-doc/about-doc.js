document.addEventListener("DOMContentLoaded", function () {
    // Fetch the README.html content from the API endpoint and inject it into the container
    fetch('/static/about-doc/README.html')
        .then(response => response.text())
        .then(html => {
            document.querySelector("#readme-html-content").innerHTML = html;
        })
        .catch(error => {
            console.error('Error fetching about document:', error);
            document.querySelector("#readme-html-content").innerHTML = '<p>Error loading about document.</p>';
        });
});