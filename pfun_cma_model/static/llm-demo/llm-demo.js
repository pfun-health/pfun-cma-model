
class LlmDemo {
    constructor() {
        this.dom = {
            form: document.getElementById('query-form'),
            queryInput: document.getElementById('query-input'),
            submitBtn: $('#submit-btn'),
            loadingContainer: $('#loading-container'),
            responseOutput: $('#response-output'),
            formattedOutput: $('#formatted-response-output'),
            healthSummaryContent: $('#health-summary-content'),
            forecastedEventsContent: $('#forecasted-events-content'),
        };

        this.retryStorageKey = 'ntry_count';
        this.maxRetries = 4;

        this.initialize();
    }

    initialize() {
        localStorage.setItem(this.retryStorageKey, '0');
        this.setupEventListeners();
        this.setupJqToast();
        this.initializeScrollSpy();
    }

    initializeScrollSpy() {
        // Initialize Bootstrap ScrollSpy
        const contentElement = document.querySelector('[data-bs-spy="scroll"]');
        if (contentElement) {
            const scrollSpy = new bootstrap.ScrollSpy(contentElement, {
                target: '#sidebar-nav',
                offset: 80,
            });
            console.debug('ScrollSpy initialized for sidebar navigation');
        }
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
        this.dom.healthSummaryContent.html('');
        this.dom.forecastedEventsContent.html('<p class="text-muted">Forecasted health events and recommendations will appear here after you submit a query.</p>');

        const rawOutputElement = document.getElementById('raw-output-section');
        if (rawOutputElement) {
            const collapseInstance = bootstrap.Collapse.getInstance(rawOutputElement) || new bootstrap.Collapse(rawOutputElement, { toggle: false });
            collapseInstance.hide();
        }
    }

    renderResponse(data) {
        /**
         * Render the successfully generated recommendations.
         */
        const strContent = JSON.stringify(data, null, 2);
        const scenarioDesc = data?.qualitative_description ?? '';
        const recsData = data?.recommendations ?? {};

        // Render health summary
        if (scenarioDesc) {
            this.dom.healthSummaryContent.html(`<p class="fs-5">${scenarioDesc}</p>`);
        }

        // Render health tips
        this.dom.formattedOutput.html('');
        this.dom.formattedOutput.append('<dl class="row">');
        Object.entries(recsData).forEach(([key, value]) => {
            let title = key.replace(/_/g, ' ');
            title = title.charAt(0).toUpperCase() + title.slice(1);
            this.dom.formattedOutput.append(`
                <dt class="col-sm-3"><strong>${title}</strong></dt>
                <dd class="col-sm-9">${value}</dd>
            `);
        });
        this.dom.formattedOutput.append('</dl>');

        // Render raw output
        this.dom.responseOutput.text(strContent);

        // Update forecasted events section if available
        if (data?.forecasted_events) {
            this.dom.forecastedEventsContent.html(this.formatForecastedEvents(data.forecasted_events));
        }

        // Scroll to output section
        const outputSection = document.getElementById('output-section');
        if (outputSection) {
            outputSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    formatForecastedEvents(events) {
        /**
         * Format forecasted events for display
         */
        if (typeof events === 'string') {
            return `<p>${events}</p>`;
        }
        if (Array.isArray(events)) {
            return `<ul class="list-group">${events.map(e => `<li class="list-group-item">${e}</li>`).join('')}</ul>`;
        }
        if (typeof events === 'object') {
            return `<dl class="row">${Object.entries(events).map(([k, v]) => `
                <dt class="col-sm-4"><strong>${k}</strong></dt>
                <dd class="col-sm-8">${v}</dd>
            `).join('')}</dl>`;
        }
        return '<p class="text-muted">No events available</p>';
    }

    async onFormSubmit(event) {
        if (event?.preventDefault) {
            event.preventDefault();
        }

        this.showLoadingContainer();

        const query = this.dom.queryInput?.value ?? '';
        const queryUrl = new URL(window.location.origin + '/llm/generate-scenario');
        queryUrl.searchParams.set('prompt', query);
        queryUrl.searchParams.set('stream', 'true');

        this.disableSubmit();
        this.clearOutput();

        try {
            if (query === '<<TEST_ERROR>>') {
                throw new Error('<<TEST_ERROR>>');
            }

            const response = await fetch(queryUrl.toString(), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: null,
            });

            this.hideLoadingContainer();

            if (!response.ok) {
                const errorBody = await response.json().catch(() => null);
                throw new Error(errorBody?.detail ?? response.statusText);
            }

            const data = await response.json();
            const strContent = JSON.stringify(data, null, 2);

            if (strContent.startsWith('Err')) {
                throw new Error(strContent);
            }

            this.renderResponse(data);
            this.enableSubmit();
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
