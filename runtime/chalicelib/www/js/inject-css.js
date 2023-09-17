async function injectCSSFile(url) {
    await new Promise((resolve, reject) => {
        fetch(url)
            .then(response => response.ok ? response.blob() : reject())
            .then(blob => {
                const objectURL = URL.createObjectURL(blob);
                const linkElement = document.createElement("link");

                linkElement.setAttribute("rel", "stylesheet");
                linkElement.setAttribute("type", "text/css");
                linkElement.setAttribute("href", objectURL);

                linkElement.onload = () => resolve();
                linkElement.onerror = () => reject();

                document.head.appendChild(linkElement);
            })
            .catch(error => reject(error));
    });
}

// Usage example:
// injectCSSFile("https://example.com/my-styles.css")
//     .then(() => {
//         console.log("CSS file successfully injected!");
//     })
//     .catch(error => {
//         console.log("Failed to inject the CSS file.", error);
//     });
