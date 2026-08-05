# Dexcom API Integration Technical Specification

## Overview
This document provides detailed technical specifications for integrating with the Dexcom Developer API, including authentication flows, data models, and best practices for health data applications.

## API Authentication

### OAuth 2.0 with PKCE Implementation
The Dexcom API uses OAuth 2.0 with Proof Key for Code Exchange (PKCE) for secure authentication. This is required for all production applications.

#### Authentication Flow Steps:

1. **Generate PKCE Parameters**
   ```javascript
   // Generate code verifier (43-128 characters)
   function generateCodeVerifier() {
     const array = new Uint8Array(32);
     window.crypto.getRandomValues(array);
     return Array.from(array, dec => ('0' + dec.toString(16)).substr(-2)).join('');
   }

   // Generate code challenge (SHA256 hash of verifier, base64 encoded)
   async function generateCodeChallenge(verifier) {
     const encoder = new TextEncoder();
     const data = encoder.encode(verifier);
     const digest = await window.crypto.subtle.digest('SHA-256', data);
     return btoa(String.fromCharCode(...new Uint8Array(digest)))
       .replace(/\+/g, '-')
       .replace(/\//g, '_')
       .replace(/=/g, '');
   }
   ```

2. **Authorization URL Construction**
   ```javascript
   const authUrl = new URL(`${baseUrl}/v2/oauth2/login`);
   authUrl.searchParams.set('client_id', clientId);
   authUrl.searchParams.set('redirect_uri', redirectUri);
   authUrl.searchParams.set('response_type', 'code');
   authUrl.searchParams.set('scope', 'offline_access');
   authUrl.searchParams.set('code_challenge', codeChallenge);
   authUrl.searchParams.set('code_challenge_method', 'S256');
   authUrl.searchParams.set('state', state);
   ```

3. **Token Exchange**
   ```javascript
   async function exchangeCodeForToken(authCode, codeVerifier) {
     const response = await fetch(`${baseUrl}/v2/oauth2/token`, {
       method: 'POST',
       headers: {
         'Content-Type': 'application/x-www-form-urlencoded'
       },
       body: new URLSearchParams({
         client_id: clientId,
         client_secret: clientSecret, // Only for confidential clients
         grant_type: 'authorization_code',
         code: authCode,
         redirect_uri: redirectUri,
         code_verifier: codeVerifier
       })
     });
     
     return response.json();
   }
   ```

### Environment Configuration

#### Production Endpoints:
- **US**: `https://api.dexcom.com`
- **EU**: `https://api.dexcom.eu`
- **Japan**: `https://api.dexcom.jp`

#### Sandbox Endpoint:
- **Sandbox**: `https://sandbox-api.dexcom.com`

#### Rate Limiting:
- Maximum 60,000 API calls per application per hour
- HTTP 429 response when limit exceeded
- Implement exponential backoff for failed requests

## Data Models and Types

### Estimated Glucose Values (EGV)
```typescript
interface EGV {
  systemTime: string;          // ISO 8601 UTC timestamp
  displayTime: string;         // Device local time
  realtimeValue: number;       // Current glucose reading (mg/dL)
  smoothedValue: number;       // Smoothed glucose value
  status: 'ok' | 'low' | 'high' | 'unknown';
  trend: 'doubleUp' | 'singleUp' | 'fortyFiveUp' | 'flat' | 
         'fortyFiveDown' | 'singleDown' | 'doubleDown';
  trendRate: number;           // Rate of change (mg/dL/min)
}
```

### Device Information
```typescript
interface DeviceInfo {
  alertScheduleList: AlertSchedule[];
  displayDevice: {
    transmitterGeneration: 'g6' | 'g7' | 'one' | 'oneplus';
    displayApp: string;
    softwareNumber: string;
  };
}

interface AlertSchedule {
  alertScheduleName: string;
  isEnabled: boolean;
  isDefaultSchedule: boolean;
  alertSettings: AlertSetting[];
}

interface AlertSetting {
  alertName: 'high' | 'low' | 'urgentLow' | 'rateOfChange';
  value: number;
  unit: 'mg/dL' | 'mmol/L';
  snooze: number; // minutes
}
```

### Events Data
```typescript
interface UserEvent {
  systemTime: string;
  displayTime: string;
  eventType: 'exercise' | 'food' | 'insulin' | 'carbs' | 'health';
  eventSubType?: string;
  value?: number;
  unit?: string;
  description?: string;
}
```

### Calibrations
```typescript
interface Calibration {
  systemTime: string;
  displayTime: string;
  glucose: number;
  unit: 'mg/dL' | 'mmol/L';
}
```

## API Endpoints Implementation

### Authenticated Request Helper
```javascript
class DexcomAPIService {
  constructor(accessToken, baseUrl = 'https://sandbox-api.dexcom.com') {
    this.accessToken = accessToken;
    this.baseUrl = baseUrl;
  }

  async makeAuthenticatedRequest(endpoint, options = {}) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
        ...options.headers
      }
    });

    if (!response.ok) {
      if (response.status === 429) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }
      if (response.status === 401) {
        throw new Error('Authentication failed. Token may be expired.');
      }
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }
}
```

### EGV Data Retrieval
```javascript
async function getGlucoseData(startDate, endDate, maxCount = 1440) {
  // Validate date range (max 30 days)
  const daysDiff = (endDate - startDate) / (1000 * 60 * 60 * 24);
  if (daysDiff > 30) {
    throw new Error('Date range cannot exceed 30 days');
  }

  const params = new URLSearchParams({
    startDate: startDate.toISOString(),
    endDate: endDate.toISOString()
  });

  if (maxCount) {
    params.set('maxCount', maxCount.toString());
  }

  return this.makeAuthenticatedRequest(`/v3/users/self/egvs?${params}`);
}
```

### Data Range Query
```javascript
async function getDataRange() {
  const response = await this.makeAuthenticatedRequest('/v3/users/self/dataRange');
  return {
    start: new Date(response.start.systemTime),
    end: new Date(response.end.systemTime),
    availableDays: response.availableDays
  };
}
```

## Data Processing and Calculations

### Time in Range (TIR) Calculation
```javascript
function calculateTimeInRange(egvData, targetLow = 70, targetHigh = 180) {
  if (!egvData || egvData.length === 0) return 0;

  const ranges = {
    veryLow: 0,    // < 54 mg/dL
    low: 0,        // 54-69 mg/dL
    target: 0,     // 70-180 mg/dL
    high: 0,       // 181-250 mg/dL
    veryHigh: 0    // > 250 mg/dL
  };

  egvData.forEach(reading => {
    const value = reading.realtimeValue;
    if (value < 54) ranges.veryLow++;
    else if (value < targetLow) ranges.low++;
    else if (value <= targetHigh) ranges.target++;
    else if (value <= 250) ranges.high++;
    else ranges.veryHigh++;
  });

  const total = egvData.length;
  return {
    veryLow: Math.round((ranges.veryLow / total) * 100),
    low: Math.round((ranges.low / total) * 100),
    target: Math.round((ranges.target / total) * 100),
    high: Math.round((ranges.high / total) * 100),
    veryHigh: Math.round((ranges.veryHigh / total) * 100)
  };
}
```

### Glucose Statistics
```javascript
function calculateGlucoseStatistics(egvData) {
  if (!egvData || egvData.length === 0) return null;

  const values = egvData.map(d => d.realtimeValue).sort((a, b) => a - b);
  const n = values.length;

  const mean = values.reduce((sum, val) => sum + val, 0) / n;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
  const standardDeviation = Math.sqrt(variance);

  return {
    count: n,
    mean: Math.round(mean),
    median: values[Math.floor(n / 2)],
    min: values[0],
    max: values[n - 1],
    standardDeviation: Math.round(standardDeviation * 10) / 10,
    coefficientOfVariation: Math.round((standardDeviation / mean) * 100 * 10) / 10,
    q25: values[Math.floor(n * 0.25)],
    q75: values[Math.floor(n * 0.75)]
  };
}
```

### Glucose Management Indicator (GMI)
```javascript
function calculateGMI(meanGlucose) {
  // GMI = 3.31 + 0.02392 × mean glucose (mg/dL)
  return Math.round((3.31 + 0.02392 * meanGlucose) * 10) / 10;
}
```

## Error Handling Best Practices

### API Error Response Handling
```javascript
class DexcomAPIError extends Error {
  constructor(message, statusCode, response) {
    super(message);
    this.name = 'DexcomAPIError';
    this.statusCode = statusCode;
    this.response = response;
  }
}

async function handleAPIResponse(response) {
  if (!response.ok) {
    const errorBody = await response.text();
    
    switch (response.status) {
      case 400:
        throw new DexcomAPIError('Bad request - check parameters', 400, errorBody);
      case 401:
        throw new DexcomAPIError('Unauthorized - token may be expired', 401, errorBody);
      case 404:
        throw new DexcomAPIError('Resource not found', 404, errorBody);
      case 429:
        throw new DexcomAPIError('Rate limit exceeded', 429, errorBody);
      case 500:
        throw new DexcomAPIError('Internal server error', 500, errorBody);
      default:
        throw new DexcomAPIError(`HTTP ${response.status}: ${response.statusText}`, response.status, errorBody);
    }
  }
  
  return response.json();
}
```

### Retry Logic with Exponential Backoff
```javascript
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
  let lastError;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (attempt === maxRetries) break;
      
      // Don't retry on authentication errors
      if (error.statusCode === 401) break;
      
      // Exponential backoff with jitter
      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError;
}
```

## Security Considerations

### Token Storage Security
```javascript
// For web applications - use secure httpOnly cookies
class SecureTokenStorage {
  static async storeTokens(tokens) {
    // Store in secure, httpOnly cookie via backend
    await fetch('/api/auth/store-tokens', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tokens)
    });
  }

  static async getAccessToken() {
    const response = await fetch('/api/auth/token', {
      credentials: 'include'
    });
    
    if (!response.ok) {
      throw new Error('Failed to retrieve access token');
    }
    
    const { accessToken } = await response.json();
    return accessToken;
  }
}

// For Chrome extensions - use chrome.storage.local
class ExtensionTokenStorage {
  static async storeTokens(tokens) {
    await chrome.storage.local.set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      expiresAt: Date.now() + (tokens.expires_in * 1000)
    });
  }

  static async getAccessToken() {
    const { accessToken, refreshToken, expiresAt } = 
      await chrome.storage.local.get(['accessToken', 'refreshToken', 'expiresAt']);
    
    if (!accessToken) {
      throw new Error('No access token available');
    }
    
    // Check if token needs refresh
    if (Date.now() >= expiresAt - 60000) {
      return this.refreshToken(refreshToken);
    }
    
    return accessToken;
  }
}
```

### Data Privacy and HIPAA Compliance
- Never log sensitive health data in console or error logs
- Implement proper data retention policies
- Use HTTPS for all API communications
- Sanitize data before displaying in UI
- Implement proper user consent flows

## Integration with Other Health APIs

### Extensible Architecture for Multiple Health APIs
```javascript
class HealthDataIntegrator {
  constructor() {
    this.connectedServices = new Map();
  }

  // Register health service integrations
  registerService(name, serviceInstance) {
    this.connectedServices.set(name, serviceInstance);
  }

  // Unified data retrieval
  async getHealthData(startDate, endDate, dataTypes = []) {
    const results = {};
    
    for (const [serviceName, service] of this.connectedServices) {
      try {
        if (dataTypes.includes('glucose') && serviceName === 'dexcom') {
          results.glucose = await service.getGlucoseData(startDate, endDate);
        }
        if (dataTypes.includes('activity') && serviceName === 'fitbit') {
          results.activity = await service.getActivityData(startDate, endDate);
        }
        if (dataTypes.includes('heart_rate') && serviceName === 'apple_health') {
          results.heartRate = await service.getHeartRateData(startDate, endDate);
        }
      } catch (error) {
        console.error(`Failed to fetch data from ${serviceName}:`, error);
        results[serviceName + '_error'] = error.message;
      }
    }
    
    return results;
  }
}

// Usage example
const healthIntegrator = new HealthDataIntegrator();
healthIntegrator.registerService('dexcom', new DexcomAPIService(dexcomToken));
healthIntegrator.registerService('fitbit', new FitbitAPIService(fitbitToken));

const healthData = await healthIntegrator.getHealthData(
  new Date('2024-01-01'),
  new Date('2024-01-07'),
  ['glucose', 'activity']
);
```

This technical specification provides the foundation for building robust, secure, and scalable health data applications that integrate with the Dexcom API and can be extended to work with other health data sources.