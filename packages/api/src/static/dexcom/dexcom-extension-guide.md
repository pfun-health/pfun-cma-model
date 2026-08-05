# Dexcom CGM Dashboard - Chrome Extension Adaptation Guide

## Overview
This guide explains how to adapt the current React-based Dexcom CGM dashboard into a Chrome extension using Manifest V3. The existing codebase has been structured with extension compatibility in mind.

## Chrome Extension Structure

### Manifest V3 Configuration
```json
{
  "manifest_version": 3,
  "name": "Dexcom CGM Dashboard",
  "version": "1.0.0",
  "description": "Continuous glucose monitoring dashboard with Dexcom API integration",
  
  "permissions": [
    "identity",
    "storage",
    "activeTab",
    "https://api.dexcom.com/*",
    "https://sandbox-api.dexcom.com/*"
  ],
  
  "host_permissions": [
    "https://api.dexcom.com/*",
    "https://sandbox-api.dexcom.com/*"
  ],
  
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  
  "action": {
    "default_popup": "popup.html",
    "default_title": "Dexcom CGM Dashboard",
    "default_icon": {
      "16": "icons/icon16.png",
      "32": "icons/icon32.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  },
  
  "oauth2": {
    "client_id": "YOUR_DEXCOM_CLIENT_ID",
    "scopes": ["offline_access"]
  },
  
  "web_accessible_resources": [
    {
      "resources": ["popup.html", "style.css", "app.js"],
      "matches": ["<all_urls>"]
    }
  ]
}
```

## OAuth Implementation for Chrome Extensions

### Background Service Worker (background.js)
```javascript
// Chrome Extension Background Service Worker
// Handles OAuth flow and API token management

class DexcomExtensionAuth {
  constructor() {
    this.clientId = 'YOUR_DEXCOM_CLIENT_ID';
    this.redirectUri = chrome.identity.getRedirectURL();
    this.baseUrl = 'https://sandbox-api.dexcom.com'; // or production
  }

  // Generate PKCE code verifier and challenge
  generateCodeVerifier() {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return Array.from(array, byte => 
      String.fromCharCode(byte).replace(/[^A-Za-z0-9\-._~]/g, '')
    ).join('').substring(0, 128);
  }

  async generateCodeChallenge(codeVerifier) {
    const encoder = new TextEncoder();
    const data = encoder.encode(codeVerifier);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return btoa(String.fromCharCode(...new Uint8Array(digest)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');
  }

  // Start OAuth flow using chrome.identity.launchWebAuthFlow
  async startAuthFlow() {
    try {
      const codeVerifier = this.generateCodeVerifier();
      const codeChallenge = await this.generateCodeChallenge(codeVerifier);
      const state = Math.random().toString(36).substring(7);

      // Store PKCE data for later use
      await chrome.storage.session.set({
        codeVerifier,
        state,
        authInProgress: true
      });

      const authUrl = new URL(`${this.baseUrl}/v2/oauth2/login`);
      authUrl.searchParams.set('client_id', this.clientId);
      authUrl.searchParams.set('redirect_uri', this.redirectUri);
      authUrl.searchParams.set('response_type', 'code');
      authUrl.searchParams.set('scope', 'offline_access');
      authUrl.searchParams.set('code_challenge', codeChallenge);
      authUrl.searchParams.set('code_challenge_method', 'S256');
      authUrl.searchParams.set('state', state);

      // Launch auth flow
      const redirectUrl = await chrome.identity.launchWebAuthFlow({
        url: authUrl.href,
        interactive: true
      });

      return this.handleAuthCallback(redirectUrl);
    } catch (error) {
      console.error('Auth flow failed:', error);
      throw error;
    }
  }

  // Handle OAuth callback
  async handleAuthCallback(redirectUrl) {
    const url = new URL(redirectUrl);
    const code = url.searchParams.get('code');
    const state = url.searchParams.get('state');
    const error = url.searchParams.get('error');

    if (error) {
      throw new Error(`OAuth error: ${error}`);
    }

    // Verify state parameter
    const { state: storedState, codeVerifier } = await chrome.storage.session.get(['state', 'codeVerifier']);
    if (state !== storedState) {
      throw new Error('Invalid state parameter');
    }

    // Exchange authorization code for access token
    return this.exchangeCodeForToken(code, codeVerifier);
  }

  // Exchange authorization code for access token
  async exchangeCodeForToken(code, codeVerifier) {
    const tokenUrl = `${this.baseUrl}/v2/oauth2/token`;
    const body = new URLSearchParams({
      client_id: this.clientId,
      grant_type: 'authorization_code',
      code: code,
      redirect_uri: this.redirectUri,
      code_verifier: codeVerifier
    });

    const response = await fetch(tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: body
    });

    if (!response.ok) {
      throw new Error(`Token exchange failed: ${response.status}`);
    }

    const tokens = await response.json();
    
    // Store tokens securely
    await chrome.storage.local.set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      expiresAt: Date.now() + (tokens.expires_in * 1000),
      tokenType: tokens.token_type
    });

    // Clean up session data
    await chrome.storage.session.clear();

    return tokens;
  }

  // Get valid access token (handles refresh if needed)
  async getAccessToken() {
    const { accessToken, refreshToken, expiresAt } = await chrome.storage.local.get(['accessToken', 'refreshToken', 'expiresAt']);
    
    if (!accessToken) {
      throw new Error('No access token available');
    }

    // Check if token needs refresh
    if (Date.now() >= expiresAt - 60000) { // Refresh 1 minute before expiry
      return this.refreshAccessToken(refreshToken);
    }

    return accessToken;
  }

  // Refresh access token
  async refreshAccessToken(refreshToken) {
    const tokenUrl = `${this.baseUrl}/v2/oauth2/token`;
    const body = new URLSearchParams({
      client_id: this.clientId,
      grant_type: 'refresh_token',
      refresh_token: refreshToken
    });

    const response = await fetch(tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: body
    });

    if (!response.ok) {
      throw new Error(`Token refresh failed: ${response.status}`);
    }

    const tokens = await response.json();
    
    // Update stored tokens
    await chrome.storage.local.set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      expiresAt: Date.now() + (tokens.expires_in * 1000)
    });

    return tokens.access_token;
  }
}

// Initialize auth service
const authService = new DexcomExtensionAuth();

// Handle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'authenticate') {
    authService.startAuthFlow()
      .then(tokens => sendResponse({ success: true, tokens }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'getAccessToken') {
    authService.getAccessToken()
      .then(token => sendResponse({ success: true, token }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
});
```

### Popup HTML (popup.html)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dexcom CGM Dashboard</title>
    <link rel="stylesheet" href="style.css">
    <style>
        body {
            width: 400px;
            height: 600px;
            margin: 0;
            padding: 0;
        }
        .extension-popup {
            height: 100vh;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div id="root" class="extension-popup"></div>
    
    <!-- Load React and other dependencies -->
    <script src="libs/react.production.min.js"></script>
    <script src="libs/react-dom.production.min.js"></script>
    <script src="libs/chart.min.js"></script>
    
    <!-- Extension-specific popup script -->
    <script src="popup.js"></script>
</body>
</html>
```

### Popup Script (popup.js)
```javascript
// Chrome Extension Popup Script
// Adapted version of the main app for extension popup

const { useState, useEffect } = React;

// Extension-specific authentication hook
function useExtensionAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await chrome.runtime.sendMessage({ action: 'getAccessToken' });
      setIsAuthenticated(response.success);
    } catch (error) {
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const authenticate = async () => {
    setIsLoading(true);
    try {
      const response = await chrome.runtime.sendMessage({ action: 'authenticate' });
      setIsAuthenticated(response.success);
      return response.success;
    } catch (error) {
      console.error('Authentication failed:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    await chrome.storage.local.clear();
    setIsAuthenticated(false);
  };

  return { isAuthenticated, isLoading, authenticate, logout };
}

// Extension-specific API service
class ExtensionDexcomService {
  async getAccessToken() {
    const response = await chrome.runtime.sendMessage({ action: 'getAccessToken' });
    if (!response.success) {
      throw new Error(response.error);
    }
    return response.token;
  }

  async makeAuthenticatedRequest(endpoint, options = {}) {
    const token = await this.getAccessToken();
    const baseUrl = 'https://sandbox-api.dexcom.com'; // or production
    
    const response = await fetch(`${baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        ...options.headers
      }
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    return response.json();
  }

  async getGlucoseData(startDate, endDate) {
    const params = new URLSearchParams({
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString()
    });

    return this.makeAuthenticatedRequest(`/v3/users/self/egvs?${params}`);
  }

  async getDeviceInfo() {
    return this.makeAuthenticatedRequest('/v3/users/self/devices');
  }
}

// Main Extension Popup Component
function ExtensionApp() {
  const { isAuthenticated, isLoading, authenticate, logout } = useExtensionAuth();
  const [glucoseData, setGlucoseData] = useState([]);
  const [currentGlucose, setCurrentGlucose] = useState(null);

  const dexcomService = new ExtensionDexcomService();

  useEffect(() => {
    if (isAuthenticated) {
      loadGlucoseData();
    }
  }, [isAuthenticated]);

  const loadGlucoseData = async () => {
    try {
      const endDate = new Date();
      const startDate = new Date(endDate.getTime() - 24 * 60 * 60 * 1000); // Last 24 hours
      
      const data = await dexcomService.getGlucoseData(startDate, endDate);
      setGlucoseData(data.egvs || []);
      
      if (data.egvs && data.egvs.length > 0) {
        setCurrentGlucose(data.egvs[data.egvs.length - 1]);
      }
    } catch (error) {
      console.error('Failed to load glucose data:', error);
    }
  };

  if (isLoading) {
    return React.createElement('div', { className: 'loading-container' }, 
      React.createElement('div', { className: 'loading-spinner' },
        React.createElement('div', { className: 'spinner' }),
        React.createElement('p', null, 'Loading...')
      )
    );
  }

  if (!isAuthenticated) {
    return React.createElement('div', { className: 'auth-container' },
      React.createElement('div', { className: 'auth-content' },
        React.createElement('h2', null, 'Connect to Dexcom'),
        React.createElement('p', null, 'Connect your Dexcom account to view your glucose data.'),
        React.createElement('button', { 
          className: 'btn btn--primary',
          onClick: authenticate 
        }, 'Connect to Dexcom')
      )
    );
  }

  return React.createElement('div', { className: 'extension-dashboard' },
    React.createElement('header', { className: 'extension-header' },
      React.createElement('h1', null, 'Dexcom CGM'),
      React.createElement('button', { 
        className: 'btn btn--secondary btn--small',
        onClick: logout 
      }, 'Disconnect')
    ),
    
    React.createElement('main', { className: 'extension-main' },
      currentGlucose && React.createElement('div', { className: 'current-glucose' },
        React.createElement('div', { className: 'glucose-value' },
          React.createElement('span', { className: 'value' }, currentGlucose.realtimeValue),
          React.createElement('span', { className: 'unit' }, 'mg/dL')
        ),
        React.createElement('div', { className: 'glucose-trend' },
          React.createElement('span', { className: `trend trend--${currentGlucose.trend}` }, 
            getTrendIcon(currentGlucose.trend)
          ),
          React.createElement('span', { className: 'trend-rate' }, 
            `${currentGlucose.trendRate > 0 ? '+' : ''}${currentGlucose.trendRate} mg/dL/min`
          )
        )
      ),
      
      React.createElement('div', { className: 'quick-stats' },
        React.createElement('div', { className: 'stat-item' },
          React.createElement('span', { className: 'stat-label' }, 'Time in Range'),
          React.createElement('span', { className: 'stat-value' }, calculateTimeInRange(glucoseData) + '%')
        ),
        React.createElement('div', { className: 'stat-item' },
          React.createElement('span', { className: 'stat-label' }, 'Avg Glucose'),
          React.createElement('span', { className: 'stat-value' }, calculateAverage(glucoseData) + ' mg/dL')
        )
      ),
      
      React.createElement('button', { 
        className: 'btn btn--primary btn--block',
        onClick: () => chrome.tabs.create({ url: chrome.runtime.getURL('dashboard.html') })
      }, 'Open Full Dashboard')
    )
  );
}

// Utility functions
function getTrendIcon(trend) {
  const icons = {
    'doubleUp': '⬆⬆',
    'singleUp': '⬆',
    'fortyFiveUp': '↗',
    'flat': '→',
    'fortyFiveDown': '↘',
    'singleDown': '⬇',
    'doubleDown': '⬇⬇'
  };
  return icons[trend] || '→';
}

function calculateTimeInRange(data) {
  if (!data || data.length === 0) return 0;
  const inRange = data.filter(d => d.realtimeValue >= 70 && d.realtimeValue <= 180).length;
  return Math.round((inRange / data.length) * 100);
}

function calculateAverage(data) {
  if (!data || data.length === 0) return 0;
  const sum = data.reduce((acc, d) => acc + d.realtimeValue, 0);
  return Math.round(sum / data.length);
}

// Render the app
ReactDOM.render(React.createElement(ExtensionApp), document.getElementById('root'));
```

## Key Differences from Web Version

### 1. Authentication Flow
- Uses `chrome.identity.launchWebAuthFlow` instead of window redirects
- Stores tokens in `chrome.storage` instead of localStorage
- Handles authentication in background service worker

### 2. Data Storage
- Uses `chrome.storage.local` for persistent token storage
- Uses `chrome.storage.session` for temporary OAuth state
- No dependency on localStorage or sessionStorage

### 3. UI Constraints
- Popup has fixed dimensions (400x600px recommended)
- Full dashboard opens in new tab
- Simplified UI for popup view

### 4. Permissions
- Requires `identity` permission for OAuth
- Requires `storage` permission for token storage
- Requires host permissions for Dexcom API endpoints

## Development Workflow

1. **Setup Development Environment**
   ```bash
   mkdir dexcom-extension
   cd dexcom-extension
   npm init -y
   npm install --save-dev web-ext webpack babel
   ```

2. **Build Process**
   - Bundle React components for production
   - Minify CSS and JavaScript
   - Copy assets to build directory

3. **Testing**
   - Load unpacked extension in Chrome Developer Mode
   - Test OAuth flow in sandbox environment
   - Verify API calls and data display

4. **Publishing**
   - Submit to Chrome Web Store
   - Provide privacy policy for health data handling
   - Include detailed permissions justification

## Security Considerations

1. **Token Security**
   - Store tokens in chrome.storage.local (encrypted by Chrome)
   - Never log sensitive authentication data
   - Implement proper token refresh logic

2. **API Rate Limiting**
   - Respect Dexcom's 60,000 calls/hour limit
   - Implement exponential backoff for failed requests
   - Cache data appropriately to minimize API calls

3. **Content Security Policy**
   - Use strict CSP in manifest
   - Avoid inline scripts and eval()
   - Load external libraries from CDN or bundle locally

4. **Data Privacy**
   - Follow HIPAA guidelines for health data
   - Implement data retention policies
   - Provide clear privacy policy to users

This guide provides a comprehensive roadmap for converting the React-based Dexcom dashboard into a Chrome extension while maintaining all core functionality and ensuring compatibility with Manifest V3 requirements.