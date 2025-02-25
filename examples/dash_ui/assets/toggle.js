// JavaScript to toggle the slider menu's visibility
async function set_toggle_slider_visibility() {
    (async () => {
        var toggle_btn = document.getElementById('toggle-button');
        toggle_btn.addEventListener("click", () => {
            var menu = document.getElementById('sliders-menu');
            if (menu.style.display === 'none' || menu.style.display === '') {
                console.log("show menu!");
                menu.style.display = 'block';
            } else {
                console.log("hide menu!");
                menu.style.display = 'none';
            }
        });
    })();
    console.log("set toggler!");
}

document.addEventListener("readystatechange", async () => {
    console.log('readyState:', document.readyState);
    if(document.readyState !== 'complete') {
        console.log("not yet complete");
    } else {
        setTimeout(() => {
            set_toggle_slider_visibility(); console.log("...loaded!");
        }, 1000);
    }
});