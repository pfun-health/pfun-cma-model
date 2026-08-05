
class LlmDemo {
    constructor() {
        this.dom = {
            form: document.getElementById('query-form'),
            queryInput: document.getElementById('query-input'),
            submitBtn: $('#submit-btn'),
            loadingContainer: $('#loading-container'),
            responseOutput: $('#response-output'),
            formattedOutput: $('#formatted-response-output'),
            outputTitle: $('#output-title'),
        };

        this.retryStorageKey = 'ntry_count';
        this.maxRetries = 4;

        this.initialize();
    }

    initialize() {
        localStorage.setItem(this.retryStorageKey, '0');
        this.setupEventListeners();
        this.setupJqToast();
    }

    setupEventListeners() {
        if (!this.dom.form) {
            console.error('LLM demo: query form not found');
            return;
        }

        this.dom.form.addEventListener('submit', (event) => this.onFormSubmit(event));
    }

    async setupJqToast() {
        try {
            const module = await import('/static/js/jquery.toast.min.js');
            const jqToast = module.default ?? module;
            jqToast(jQuery, window, document);
            console.debug('...finished setup for jquery toast, try: $.toast(...)');
        } catch (error) {
            console.warn('Failed to load jquery toast:', error);
        }
    }

    showLoadingContainer() {
        this.dom.loadingContainer.removeClass('d-none');
        this.dom.loadingContainer.show();
    }

    hideLoadingContainer() {
        this.dom.loadingContainer.hide();
    }

    async showAlerts() {
        const errorMsg = 'Whoops! The server is busy right now.\nRetrying your request... Please wait.';
        const timeTillHide = 4500; // time in ms until the toast should hide

        const errToast = $.toast({
            heading: 'Error',
            text: errorMsg,
            showHideTransition: 'fade',
            icon: 'error',
            hideAfter: timeTillHide,
            stack: true,
        });

        setTimeout(() => {
            errToast.update({
                heading: 'Trying again...',
                text: 'Attempting your request again... Please wait.',
                hideAfter: timeTillHide,
            });

            setTimeout(async () => {
                await this.onFormSubmit();
            }, timeTillHide + 500);
        }, timeTillHide + 1230);
    }

    getRetryCount() {
        return parseInt(localStorage.getItem(this.retryStorageKey) ?? '0', 10);
    }

    incrementRetryCount() {
        const nextCount = this.getRetryCount() + 1;
        localStorage.setItem(this.retryStorageKey, `${nextCount}`);
        return nextCount;
    }

    disableSubmit() {
        this.dom.submitBtn.addClass('disabled');
    }

    enableSubmit() {
        this.dom.submitBtn.removeClass('disabled');
    }

    clearOutput() {
        this.dom.formattedOutput.html('');
        this.dom.responseOutput.html('');
    }

    renderResponse(data) {
        /**
         * Render the successfully generated recommendations.
         */
        const strContent = JSON.stringify(data, null, 2);
        const scenarioDesc = data?.qualitative_description ?? '';
        const recsData = data?.recommendations ?? {};

        this.dom.formattedOutput.append(`<p class="lead">${scenarioDesc}</p>`);

        Object.entries(recsData).forEach(([key, value]) => {
            this.dom.formattedOutput.append(`<dt>${key}</dt><dd>${value}</dd>`);
        });

        this.dom.responseOutput.text(strContent);
        this.dom.outputTitle.get(0)?.scrollIntoView({ behavior: 'smooth' });
    }

    async onFormSubmit(event) {
        if (event?.preventDefault) {
            event.preventDefault();
        }

        // loading
        this.showLoadingContainer();

        // definitions for structured query
        const query = this.dom.queryInput?.value ?? '';
        const queryUrl = new URL(window.location.origin + '/llm/generate-scenario');
        queryUrl.searchParams.set('prompt', query);

        // Disable submit button, clear old output
        this.disableSubmit();
        this.clearOutput();

        try {
            if (query === '<<TEST_ERROR>>') {
                throw new Error('<<TEST_ERROR>>');
            }

            // Attempt the request
            const response = await fetch(queryUrl.toString(), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: null,
            });

            // done loading
            this.hideLoadingContainer();

            if (!response.ok) {
                // Handle "not ok" response
                const errorBody = await response.json().catch(() => null);
                throw new Error(errorBody?.detail ?? response.statusText);
            }

            // (successful response) Get the response data as JSON
            const data = await response.json();
            const strContent = JSON.stringify(data, null, 2);

            // Handle error (not the response we expected)
            if (strContent.startsWith('Err')) {
                throw new Error(strContent);
            }
            
            // Render, upon verifying response is as expected
            this.renderResponse(data);
            this.enableSubmit(); // re-enable submit button
            
        } catch (error) {
            const retryCount = this.incrementRetryCount();

            if (retryCount >= this.maxRetries) {
                alert(
                    'Your request has been tried too many times.\n\nAfter the page reloads, please try again.\nThe page should automatically refresh after you close this dialog.'
                );
                window.location.reload();
                return;
            }

            console.warn(`Error: ${error.message}`);
            await this.showAlerts();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new LlmDemo();
});
