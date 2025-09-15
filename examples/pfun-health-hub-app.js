/**
 * Health API Integration Hub - Main Application
 * Designed for easy Chrome Extension adaptation
 */

class HealthAPIHub {
    constructor() {
        this.currentPage = 'home';
        this.apiClients = new Map();
        this.charts = new Map();
        this.mockData = this.initializeMockData();
        this.apiStatus = this.initializeAPIStatus();
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeAPIClients();
        this.renderCurrentPage();
        this.updateAPIStatus();
        this.initializeCharts();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.navigateTo(e.target.dataset.page);
            });
        });

        // Connect buttons
        document.querySelectorAll('.connect-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.initiateOAuth(e.target.dataset.api);
            });
        });

        // OAuth demo buttons
        document.querySelectorAll('.oauth-demo-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.demonstrateOAuthFlow(e.target.dataset.api);
            });
        });

        // Modal controls
        document.querySelector('.modal-close')?.addEventListener('click', () => {
            this.hideModal();
        });

        document.querySelector('.modal-backdrop')?.addEventListener('click', () => {
            this.hideModal();
        });

        // Settings actions
        this.setupSettingsEventListeners();
    }

    setupSettingsEventListeners() {
        // Refresh buttons
        document.querySelectorAll('.connection-actions .btn--outline').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const connectionItem = e.target.closest('.connection-item');
                const apiName = connectionItem.querySelector('h4').textContent.toLowerCase().replace(' ', '');
                this.refreshAPIConnection(apiName);
            });
        });

        // Disconnect buttons  
        document.querySelectorAll('.connection-actions .btn--secondary').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const connectionItem = e.target.closest('.connection-item');
                const apiName = connectionItem.querySelector('h4').textContent.toLowerCase().replace(' ', '');
                this.disconnectAPI(apiName);
            });
        });

        // Connect buttons in settings
        document.querySelectorAll('.connection-actions .btn--primary').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const connectionItem = e.target.closest('.connection-item');
                const apiName = connectionItem.querySelector('h4').textContent.toLowerCase().replace(' ', '');
                this.initiateOAuth(apiName);
            });
        });
    }

    initializeMockData() {
        return {
            fitbit: {
                steps: [
                    { date: '2025-09-12', value: 8420 },
                    { date: '2025-09-11', value: 10250 },
                    { date: '2025-09-10', value: 7890 },
                    { date: '2025-09-09', value: 12340 },
                    { date: '2025-09-08', value: 9870 }
                ],
                heartRate: [
                    { time: '2025-09-12T08:00:00Z', value: 68 },
                    { time: '2025-09-12T09:00:00Z', value: 72 },
                    { time: '2025-09-12T10:00:00Z', value: 88 },
                    { time: '2025-09-12T11:00:00Z', value: 76 },
                    { time: '2025-09-12T12:00:00Z', value: 74 }
                ],
                sleep: {
                    date: '2025-09-11',
                    duration: 28800000,
                    efficiency: 89,
                    stages: { deep: 120, light: 280, rem: 95, wake: 25 }
                }
            },
            dexcom: {
                glucose: [
                    { time: '2025-09-12T11:00:00Z', value: 110, trend: 'flat' },
                    { time: '2025-09-12T10:45:00Z', value: 108, trend: 'flat' },
                    { time: '2025-09-12T10:30:00Z', value: 105, trend: 'rising' },
                    { time: '2025-09-12T10:15:00Z', value: 102, trend: 'rising' },
                    { time: '2025-09-12T10:00:00Z', value: 98, trend: 'rising' }
                ]
            }
        };
    }

    initializeAPIStatus() {
        return new Map([
            ['fitbit', { connected: true, lastSync: new Date('2025-09-12T10:30:00Z') }],
            ['googlefit', { connected: false, lastSync: null }],
            ['dexcom', { connected: true, lastSync: new Date('2025-09-12T11:15:00Z') }],
            ['applehealth', { connected: false, lastSync: null }]
        ]);
    }

    initializeAPIClients() {
        // Base API Client Pattern
        class BaseAPIClient {
            constructor(config) {
                this.config = config;
                this.baseURL = config.baseURL;
                this.clientId = config.clientId;
                this.accessToken = null;
                this.refreshToken = null;
            }

            async makeRequest(endpoint, options = {}) {
                const url = `${this.baseURL}${endpoint}`;
                const headers = {
                    'Content-Type': 'application/json',
                    ...options.headers
                };

                if (this.accessToken) {
                    headers.Authorization = `Bearer ${this.accessToken}`;
                }

                try {
                    const response = await fetch(url, {
                        ...options,
                        headers
                    });

                    if (response.status === 401) {
                        await this.refreshAccessToken();
                        return this.makeRequest(endpoint, options);
                    }

                    return response.json();
                } catch (error) {
                    console.error(`API request failed: ${error.message}`);
                    throw error;
                }
            }

            async refreshAccessToken() {
                // Implement token refresh logic
                console.log('Refreshing access token...');
            }

            generatePKCE() {
                const codeVerifier = this.generateRandomString(128);
                const codeChallenge = btoa(codeVerifier).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
                return { codeVerifier, codeChallenge };
            }

            generateRandomString(length) {
                const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
                let result = '';
                for (let i = 0; i < length; i++) {
                    result += chars.charAt(Math.floor(Math.random() * chars.length));
                }
                return result;
            }
        }

        // Fitbit API Client
        class FitbitAPIClient extends BaseAPIClient {
            constructor() {
                super({
                    baseURL: 'https://api.fitbit.com',
                    clientId: 'fitbit_client_id',
                    scopes: ['activity', 'heartrate', 'location', 'nutrition', 'profile', 'settings', 'sleep', 'social', 'weight']
                });
            }

            async getActivities(date) {
                return this.makeRequest(`/1/user/-/activities/date/${date}.json`);
            }

            async getHeartRate(date) {
                return this.makeRequest(`/1/user/-/activities/heart/date/${date}/1d.json`);
            }

            async getSleep(date) {
                return this.makeRequest(`/1.2/user/-/sleep/date/${date}.json`);
            }
        }

        // Google Fit API Client
        class GoogleFitAPIClient extends BaseAPIClient {
            constructor() {
                super({
                    baseURL: 'https://www.googleapis.com/fitness/v1',
                    clientId: 'googlefit_client_id',
                    scopes: ['https://www.googleapis.com/auth/fitness.activity.read', 'https://www.googleapis.com/auth/fitness.body.read']
                });
            }

            async getDataSources() {
                return this.makeRequest('/users/me/dataSources');
            }

            async getDatasets(dataSourceId, datasetId) {
                return this.makeRequest(`/users/me/dataSources/${dataSourceId}/datasets/${datasetId}`);
            }
        }

        // Dexcom API Client
        class DexcomAPIClient extends BaseAPIClient {
            constructor() {
                super({
                    baseURL: 'https://sandbox-api.dexcom.com',
                    clientId: 'dexcom_client_id',
                    scopes: ['offline_access']
                });
            }

            async getDevices() {
                return this.makeRequest('/v3/users/self/devices');
            }

            async getEGVs(startDate, endDate) {
                return this.makeRequest(`/v3/users/self/egvs?startDate=${startDate}&endDate=${endDate}`);
            }

            async getEvents(startDate, endDate) {
                return this.makeRequest(`/v3/users/self/events?startDate=${startDate}&endDate=${endDate}`);
            }
        }

        // Initialize clients
        this.apiClients.set('fitbit', new FitbitAPIClient());
        this.apiClients.set('googlefit', new GoogleFitAPIClient());
        this.apiClients.set('dexcom', new DexcomAPIClient());
    }

    navigateTo(page) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.page === page);
        });

        // Update pages
        document.querySelectorAll('.page').forEach(pageEl => {
            pageEl.classList.toggle('active', pageEl.dataset.page === page);
        });

        this.currentPage = page;

        // Initialize page-specific functionality
        if (page === 'dashboard') {
            setTimeout(() => this.initializeCharts(), 100);
        }
    }

    updateAPIStatus() {
        this.apiStatus.forEach((status, apiId) => {
            // Update status panel icons
            const statusItem = document.querySelector(`.status-item[data-api="${apiId}"]`);
            const statusIcon = statusItem?.querySelector('.status-icon');
            
            if (statusIcon) {
                statusIcon.classList.toggle('connected', status.connected);
                statusIcon.classList.toggle('disconnected', !status.connected);
            }

            // Update API cards on home page
            const apiCard = document.querySelector(`.api-card[data-api="${apiId}"]`);
            if (apiCard) {
                apiCard.classList.toggle('connected', status.connected);
                apiCard.classList.toggle('disconnected', !status.connected);

                const statusElement = apiCard.querySelector('.status');
                if (statusElement) {
                    statusElement.textContent = status.connected ? 'Connected' : 'Not Connected';
                    statusElement.classList.toggle('connected', status.connected);
                    statusElement.classList.toggle('disconnected', !status.connected);
                }

                const lastSyncElement = apiCard.querySelector('.last-sync');
                if (lastSyncElement) {
                    if (status.lastSync) {
                        const timeDiff = this.getTimeDifference(status.lastSync);
                        lastSyncElement.textContent = `Last sync: ${timeDiff}`;
                        lastSyncElement.style.display = 'block';
                    } else {
                        lastSyncElement.style.display = 'none';
                    }
                }

                // Update connect/disconnect buttons
                const connectBtn = apiCard.querySelector('.connect-btn');
                const otherActions = apiCard.querySelector('.api-actions');
                if (connectBtn && otherActions) {
                    if (status.connected) {
                        connectBtn.style.display = 'none';
                        // Show other action buttons
                        const viewBtn = otherActions.querySelector('.btn--outline');
                        const disconnectBtn = otherActions.querySelector('.btn--secondary');
                        if (viewBtn) viewBtn.style.display = 'inline-flex';
                        if (disconnectBtn) disconnectBtn.style.display = 'inline-flex';
                    } else {
                        connectBtn.style.display = 'inline-flex';
                        // Hide other action buttons
                        const viewBtn = otherActions.querySelector('.btn--outline');
                        const disconnectBtn = otherActions.querySelector('.btn--secondary');
                        if (viewBtn) viewBtn.style.display = 'none';
                        if (disconnectBtn) disconnectBtn.style.display = 'none';
                    }
                }
            }

            // Update settings page connection items
            this.updateSettingsConnectionStatus(apiId, status);
        });
    }

    updateSettingsConnectionStatus(apiId, status) {
        const connectionItems = document.querySelectorAll('.connection-item');
        connectionItems.forEach(item => {
            const h4 = item.querySelector('h4');
            const itemApiId = h4.textContent.toLowerCase().replace(' ', '');
            
            if (itemApiId === apiId) {
                const statusText = item.querySelector('.connection-info p');
                const actions = item.querySelector('.connection-actions');
                
                if (statusText) {
                    if (status.connected) {
                        const timeDiff = status.lastSync ? this.getTimeDifference(status.lastSync) : 'Never';
                        statusText.textContent = `Connected • Last sync: ${timeDiff}`;
                    } else {
                        statusText.textContent = 'Not connected';
                    }
                }
                
                if (actions) {
                    // Clear existing buttons
                    actions.innerHTML = '';
                    
                    if (status.connected) {
                        // Show refresh and disconnect buttons
                        const refreshBtn = document.createElement('button');
                        refreshBtn.className = 'btn btn--outline btn--sm';
                        refreshBtn.textContent = 'Refresh';
                        refreshBtn.addEventListener('click', () => this.refreshAPIConnection(apiId));
                        
                        const disconnectBtn = document.createElement('button');
                        disconnectBtn.className = 'btn btn--secondary btn--sm';
                        disconnectBtn.textContent = 'Disconnect';
                        disconnectBtn.addEventListener('click', () => this.disconnectAPI(apiId));
                        
                        actions.appendChild(refreshBtn);
                        actions.appendChild(disconnectBtn);
                    } else {
                        // Show connect button
                        const connectBtn = document.createElement('button');
                        connectBtn.className = 'btn btn--primary btn--sm';
                        connectBtn.textContent = 'Connect';
                        connectBtn.addEventListener('click', () => this.initiateOAuth(apiId));
                        
                        actions.appendChild(connectBtn);
                    }
                }
            }
        });
    }

    getTimeDifference(date) {
        const now = new Date();
        const diff = now - date;
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

        if (hours > 0) {
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        } else {
            return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        }
    }

    initiateOAuth(apiId) {
        this.demonstrateOAuthFlow(apiId);
    }

    demonstrateOAuthFlow(apiId) {
        const modal = document.getElementById('oauthModal');
        const message = document.getElementById('oauthMessage');
        const progressSteps = document.querySelectorAll('.progress-step');

        // Show modal
        modal.classList.remove('hidden');
        message.textContent = `Initiating ${apiId.charAt(0).toUpperCase() + apiId.slice(1)} OAuth flow...`;

        // Reset progress
        progressSteps.forEach((step, index) => {
            step.classList.toggle('active', index === 0);
        });

        // Simulate OAuth steps
        const steps = [
            { delay: 1000, message: 'Redirecting to authorization server...', stepIndex: 0 },
            { delay: 2500, message: 'Waiting for user consent...', stepIndex: 1 },
            { delay: 4000, message: 'Receiving authorization code...', stepIndex: 2 },
            { delay: 5500, message: 'Exchanging code for access token...', stepIndex: 3 },
            { delay: 6500, message: 'Connection successful!', stepIndex: 3, complete: true }
        ];

        steps.forEach(step => {
            setTimeout(() => {
                message.textContent = step.message;
                progressSteps.forEach((el, index) => {
                    el.classList.toggle('active', index <= step.stepIndex);
                });

                if (step.complete) {
                    // Update API status
                    this.apiStatus.set(apiId, {
                        connected: true,
                        lastSync: new Date()
                    });

                    // Update UI across all pages
                    this.updateAPIStatus();

                    // Close modal after delay
                    setTimeout(() => {
                        this.hideModal();
                    }, 1500);
                }
            }, step.delay);
        });
    }

    hideModal() {
        document.getElementById('oauthModal').classList.add('hidden');
    }

    refreshAPIConnection(apiId) {
        console.log(`Refreshing ${apiId} connection...`);
        
        // Simulate refresh
        setTimeout(() => {
            const status = this.apiStatus.get(apiId);
            if (status && status.connected) {
                status.lastSync = new Date();
                this.updateAPIStatus();
            }
        }, 1000);
    }

    disconnectAPI(apiId) {
        console.log(`Disconnecting ${apiId}...`);
        
        this.apiStatus.set(apiId, {
            connected: false,
            lastSync: null
        });

        this.updateAPIStatus();
    }

    initializeCharts() {
        if (this.currentPage !== 'dashboard') return;

        // Steps Chart
        this.createStepsChart();
        
        // Heart Rate Chart
        this.createHeartRateChart();
        
        // Glucose Chart
        this.createGlucoseChart();
    }

    createStepsChart() {
        const canvas = document.getElementById('stepsChart');
        if (!canvas) return;

        // Destroy existing chart
        if (this.charts.has('steps')) {
            this.charts.get('steps').destroy();
        }

        const ctx = canvas.getContext('2d');
        const data = this.mockData.fitbit.steps;

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })),
                datasets: [{
                    label: 'Steps',
                    data: data.map(d => d.value),
                    backgroundColor: '#1FB8CD',
                    borderColor: '#1FB8CD',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('steps', chart);
    }

    createHeartRateChart() {
        const canvas = document.getElementById('heartRateChart');
        if (!canvas) return;

        // Destroy existing chart
        if (this.charts.has('heartRate')) {
            this.charts.get('heartRate').destroy();
        }

        const ctx = canvas.getContext('2d');
        const data = this.mockData.fitbit.heartRate;

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.time).toLocaleTimeString('en-US', { 
                    hour: 'numeric', 
                    minute: '2-digit',
                    hour12: true 
                })),
                datasets: [{
                    label: 'Heart Rate (BPM)',
                    data: data.map(d => d.value),
                    borderColor: '#FFC185',
                    backgroundColor: 'rgba(255, 193, 133, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 60,
                        max: 100
                    }
                }
            }
        });

        this.charts.set('heartRate', chart);
    }

    createGlucoseChart() {
        const canvas = document.getElementById('glucoseChart');
        if (!canvas) return;

        // Destroy existing chart
        if (this.charts.has('glucose')) {
            this.charts.get('glucose').destroy();
        }

        const ctx = canvas.getContext('2d');
        const data = this.mockData.dexcom.glucose;

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.time).toLocaleTimeString('en-US', { 
                    hour: 'numeric', 
                    minute: '2-digit',
                    hour12: true 
                })),
                datasets: [{
                    label: 'Glucose (mg/dL)',
                    data: data.map(d => d.value),
                    borderColor: '#B4413C',
                    backgroundColor: 'rgba(180, 65, 60, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 80,
                        max: 140,
                        ticks: {
                            callback: function(value) {
                                return value + ' mg/dL';
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('glucose', chart);
    }

    renderCurrentPage() {
        // Initial page rendering logic if needed
        this.updateAPIStatus();
    }

    // Chrome Extension Compatibility Methods
    getChromeExtensionManifest() {
        return {
            "manifest_version": 3,
            "name": "Health API Integration Hub",
            "version": "1.0.0",
            "description": "Integrate and visualize data from multiple health APIs",
            "permissions": [
                "storage",
                "activeTab",
                "identity"
            ],
            "host_permissions": [
                "https://api.fitbit.com/*",
                "https://www.googleapis.com/*",
                "https://sandbox-api.dexcom.com/*"
            ],
            "background": {
                "service_worker": "background.js"
            },
            "action": {
                "default_popup": "popup.html",
                "default_title": "Health API Hub"
            },
            "content_security_policy": {
                "extension_pages": "script-src 'self'; object-src 'self'"
            },
            "oauth2": {
                "client_id": "your_client_id",
                "scopes": [
                    "https://www.googleapis.com/auth/fitness.activity.read",
                    "https://www.googleapis.com/auth/fitness.body.read"
                ]
            }
        };
    }

    // Simulated Chrome Extension Storage
    async chromeStorageSet(key, value) {
        // In actual extension: chrome.storage.local.set({[key]: value})
        localStorage.setItem(key, JSON.stringify(value));
    }

    async chromeStorageGet(key) {
        // In actual extension: chrome.storage.local.get([key])
        const value = localStorage.getItem(key);
        return value ? JSON.parse(value) : null;
    }

    // Simulated Chrome Extension Messaging
    sendMessageToBackground(message) {
        // In actual extension: chrome.runtime.sendMessage(message)
        console.log('Message to background script:', message);
        
        // Simulate response
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({ success: true, data: 'Mock response from background' });
            }, 100);
        });
    }

    // Extension-specific OAuth handler
    async handleExtensionOAuth(provider) {
        // In actual extension, this would use chrome.identity.launchWebAuthFlow
        console.log(`Extension OAuth flow for ${provider}`);
        
        const authUrl = this.buildOAuthURL(provider);
        console.log('Auth URL:', authUrl);
        
        // Simulate successful authentication
        return {
            access_token: 'mock_access_token',
            refresh_token: 'mock_refresh_token',
            expires_in: 3600
        };
    }

    buildOAuthURL(provider) {
        const configs = {
            fitbit: {
                authUrl: 'https://www.fitbit.com/oauth2/authorize',
                clientId: 'fitbit_client_id',
                redirectUri: 'chrome-extension://extension-id/oauth-callback.html',
                scope: 'activity heartrate location nutrition profile settings sleep social weight'
            },
            googlefit: {
                authUrl: 'https://accounts.google.com/o/oauth2/v2/auth',
                clientId: 'googlefit_client_id',
                redirectUri: 'chrome-extension://extension-id/oauth-callback.html',
                scope: 'https://www.googleapis.com/auth/fitness.activity.read https://www.googleapis.com/auth/fitness.body.read'
            },
            dexcom: {
                authUrl: 'https://sandbox-api.dexcom.com/v2/oauth2/login',
                clientId: 'dexcom_client_id',
                redirectUri: 'chrome-extension://extension-id/oauth-callback.html',
                scope: 'offline_access'
            }
        };

        const config = configs[provider];
        if (!config) return null;

        const params = new URLSearchParams({
            response_type: 'code',
            client_id: config.clientId,
            redirect_uri: config.redirectUri,
            scope: config.scope,
            code_challenge_method: 'S256',
            code_challenge: 'mock_code_challenge'
        });

        return `${config.authUrl}?${params.toString()}`;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new HealthAPIHub();
    
    // Make app globally available for debugging
    window.healthAPIHub = app;
    
    console.log('Health API Integration Hub initialized');
    console.log('Chrome Extension Manifest:', app.getChromeExtensionManifest());
});