from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8001/demo/fluid-demo")

    # Get the canvas element
    canvas = page.query_selector("#fluid-canvas")
    if canvas:
        # Get the bounding box of the canvas
        box = canvas.bounding_box()
        if box:
            # Simulate a drag across the canvas
            page.mouse.move(box['x'] + box['width'] / 4, box['y'] + box['height'] / 4)
            page.mouse.down()
            page.mouse.move(box['x'] + box['width'] * 3 / 4, box['y'] + box['height'] * 3 / 4, steps=20)
            page.mouse.up()

    # Wait for the simulation to render
    page.wait_for_timeout(1000)

    page.screenshot(path="jules-scratch/verification/verification.png")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
