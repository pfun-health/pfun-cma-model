// pfun_cma_model/static/js/demos-common.js
window.addEventListener("DOMContentLoaded", () => {
    const runForm = document.getElementById("runForm");
    if (!runForm) return;

    const resetBtn = document.getElementById("resetSimulationBtn");
    const ranges = runForm.querySelectorAll('input[type=range]');

    // Function to update output element for a range input
    const updateRangeOutput = (range) => {
        const outputElement = document.getElementById(`rangeValue-${range.id}`);
        if (outputElement) {
            outputElement.textContent = range.value;
        }
    };

    // Add input listeners to all range inputs
    ranges.forEach(range => {
        // Initial update
        updateRangeOutput(range);

        range.addEventListener('input', () => {
            updateRangeOutput(range);
        });
    });

    if (resetBtn) {
        resetBtn.addEventListener("click", (e) => {
            // runForm.reset(); // Don't call explicitly if button is type="reset", native handler does it.

            // Just update outputs after delay
            setTimeout(() => {
                ranges.forEach(range => {
                    updateRangeOutput(range);
                });
            }, 0);
        });

        // Trigger reset on load to ensure consistent state
        resetBtn.click();
    }
});
