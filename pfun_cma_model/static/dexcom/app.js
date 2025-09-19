// Dexcom Health Dashboard - React Application with TypeScript patterns
// Compatible with Chrome Extension Manifest V3

const { useState, useEffect, useCallback, useMemo } = React;

// Constants and Configuration
const DEXCOM_CONFIG = {
  endpoints: {
    production: {
      us: "https://api.dexcom.com",
      eu: "https://api.dexcom.eu",
      jp: "https://api.dexcom.jp"
    },
    sandbox: "https://sandbox-api.dexcom.com"
  },
  oauth: {
    authorization: "/v2/oauth2/login",
  },
  api: {
    egvs: "/users/self/egvs",
    events: "/users/self/events",
    devices: "/users/self/devices",
    dataRange: "/users/self/dataRange",
    calibrations: "/users/self/calibrations",
    alerts: "/users/self/alerts"
  },
  scope: "offline_access" // currently the only acceptable 'scope'
};

const GLUCOSE_RANGES = {
  veryLow: 54,
  low: 70,
  targetLow: 70,
  targetHigh: 180,
  high: 250,
  veryHigh: 400
};

const CHART_COLORS = {
  veryLow: "#C92A2A",
  low: "#FF6B6B", 
  target: "#51CF66",
  high: "#FFD43B",
  veryHigh: "#FF6B6B"
};

// Utility Functions
const generateCodeVerifier = () => {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return btoa(String.fromCharCode.apply(null, array))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
};

const generateCodeChallenge = async (verifier) => {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return btoa(String.fromCharCode.apply(null, new Uint8Array(digest)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
};

const getGlucoseCategory = (value) => {
  if (value < GLUCOSE_RANGES.veryLow) return 'very-low';
  if (value < GLUCOSE_RANGES.low) return 'low';
  if (value <= GLUCOSE_RANGES.targetHigh) return 'target';
  if (value < GLUCOSE_RANGES.high) return 'high';
  return 'very-high';
};

const getTrendArrow = (trend) => {
  const trendMap = {
    'doubleUp': '⇈',
    'singleUp': '↗',
    'fortyFiveUp': '↗',
    'flat': '→',
    'fortyFiveDown': '↘',
    'singleDown': '↘',
    'doubleDown': '⇊'
  };
  return trendMap[trend] || '→';
};

const calculateTimeInRange = (egvData) => {
  if (!egvData || egvData.length === 0) {
    return { veryLow: 0, low: 0, target: 0, high: 0, veryHigh: 0 };
  }

  const total = egvData.length;
  const counts = egvData.reduce((acc, reading) => {
    const category = getGlucoseCategory(reading.realtimeValue);
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {});

  return {
    veryLow: Math.round(((counts['very-low'] || 0) / total) * 100),
    low: Math.round(((counts['low'] || 0) / total) * 100),
    target: Math.round(((counts['target'] || 0) / total) * 100),
    high: Math.round(((counts['high'] || 0) / total) * 100),
    veryHigh: Math.round(((counts['very-high'] || 0) / total) * 100)
  };
};

// Authentication Service
const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [accessToken, setAccessToken] = useState(null);
  const [refreshToken, setRefreshToken] = useState(null);
  const [environment, setEnvironment] = useState('sandbox');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const getBaseUrl = useCallback(() => {
    return environment === 'sandbox' 
      ? DEXCOM_CONFIG.endpoints.sandbox 
      : DEXCOM_CONFIG.endpoints.production.us;
  }, [environment]);

  const startAuthFlow = useCallback(async (env = 'sandbox') => {
    try {
      setIsLoading(true);
      setError(null);
      setEnvironment(env);
      
      const baseUrl = env === 'sandbox' 
        ? DEXCOM_CONFIG.endpoints.sandbox 
        : DEXCOM_CONFIG.endpoints.production.us;
      
      const authUrl = new URL(baseUrl + DEXCOM_CONFIG.oauth.authorization);
      // The client_id will be added by the backend
      authUrl.searchParams.set('redirect_uri', window.location.origin + "/dexcom/auth/callback");
      authUrl.search_params.set('response_type', 'code');
      authUrl.searchParams.set('scope', DEXCOM_CONFIG.scope);
      authUrl.searchParams.set('state', Math.random().toString(36).substring(7));
      
      window.location.href = authUrl.toString();
      
    } catch (err) {
      setError('Failed to start authentication: ' + err.message);
      setIsLoading(false);
    }
  }, []);

  const exchangeCodeForToken = useCallback(async (code) => {
    try {
      setIsLoading(true);
      
      const response = await fetch("/dexcom/token", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          code: code,
          redirect_uri: window.location.origin + "/dexcom/auth/callback"
        })
      });
      
      if (!response.ok) {
        throw new Error('Token exchange failed');
      }
      
      const data = await response.json();
      
      // At this point, the backend has the tokens and they are in the session.
      // We just need to set the frontend state to authenticated.
      setIsAuthenticated(true);
      setEnvironment(sessionStorage.getItem('dexcom_environment') || 'sandbox');
      
    } catch (err) {
      setError('Token exchange failed: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    setAccessToken(null);
    setRefreshToken(null);
    setIsAuthenticated(false);
    setError(null);
    sessionStorage.clear();
  }, []);

  return {
    isAuthenticated,
    accessToken,
    refreshToken,
    environment,
    isLoading,
    error,
    startAuthFlow,
    exchangeCodeForToken,
    logout,
    getBaseUrl
  };
};

// Dexcom API Service
const useDexcomData = (auth) => {
  const [egvData, setEgvData] = useState([]);
  const [deviceData, setDeviceData] = useState(null);
  const [dataRange, setDataRange] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchEGVs = useCallback(async (startDate, endDate) => {
    if (!auth.isAuthenticated) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const url = new URL(window.location.origin + '/dexcom' + DEXCOM_CONFIG.api.egvs);
      if (startDate) url.searchParams.set('startDate', startDate);
      if (endDate) url.searchParams.set('endDate', endDate);
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const data = await response.json();
      setEgvData(data.egvs || []);
      
    } catch (err) {
      setError('Failed to fetch glucose data: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  }, [auth]);

  const fetchDeviceInfo = useCallback(async () => {
    if (!auth.isAuthenticated) return;
    
    try {
      const response = await fetch(window.location.origin + '/dexcom' + DEXCOM_CONFIG.api.devices);
      
      if (!response.ok) {
        throw new Error(`Device API request failed: ${response.status}`);
      }
      
      const data = await response.json();
      setDeviceData(data);
      
    } catch (err) {
      console.error('Failed to fetch device info:', err);
    }
  }, [auth]);

  // Auto-fetch data when authenticated
  useEffect(() => {
    if (auth.isAuthenticated) {
      const endDate = new Date().toISOString();
      const startDate = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(); // Last 24 hours
      
      fetchEGVs(startDate, endDate);
      fetchDeviceInfo();
    }
  }, [auth.isAuthenticated, fetchEGVs, fetchDeviceInfo]);

  return {
    egvData,
    deviceData,
    dataRange,
    isLoading,
    error,
    fetchEGVs,
    fetchDeviceInfo
  };
};

// Chart Components
const GlucoseChart = ({ data }) => {
  const chartRef = React.useRef(null);
  const chartInstanceRef = React.useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const ctx = chartRef.current.getContext('2d');

    // Destroy existing chart
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const chartData = {
      labels: data.map(d => new Date(d.displayTime).toLocaleTimeString()),
      datasets: [{
        label: 'Glucose (mg/dL)',
        data: data.map(d => d.realtimeValue),
        borderColor: '#1FB8CD',
        backgroundColor: 'rgba(31, 184, 205, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: data.map(d => {
          const category = getGlucoseCategory(d.realtimeValue);
          return CHART_COLORS[category] || '#1FB8CD';
        }),
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 4
      }]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const dataPoint = data[context.dataIndex];
              return [
                `Glucose: ${context.parsed.y} mg/dL`,
                `Trend: ${getTrendArrow(dataPoint.trend)}`,
                `Rate: ${dataPoint.trendRate} mg/dL/min`
              ];
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          min: 40,
          max: 300,
          grid: {
            color: function(context) {
              const value = context.tick.value;
              if (value === GLUCOSE_RANGES.low || value === GLUCOSE_RANGES.targetHigh) {
                return 'rgba(255, 193, 7, 0.5)'; // Target range borders
              }
              return 'rgba(0, 0, 0, 0.1)';
            }
          },
          ticks: {
            callback: function(value) {
              return value + ' mg/dL';
            }
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      },
      elements: {
        point: {
          hoverRadius: 6
        }
      }
    };

    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: chartData,
      options: options
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, [data]);

  if (!data || data.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <p>No glucose data available</p>
      </div>
    );
  }

  return <canvas ref={chartRef}></canvas>;
};

const TimeInRangeChart = ({ data }) => {
  const timeInRange = useMemo(() => calculateTimeInRange(data), [data]);

  return (
    <div className="time-in-range">
      <div className="range-bar">
        <div 
          className="range-segment very-low" 
          style={{ flexBasis: `${timeInRange.veryLow}%` }}
        >
          {timeInRange.veryLow > 5 ? `${timeInRange.veryLow}%` : ''}
        </div>
        <div 
          className="range-segment low" 
          style={{ flexBasis: `${timeInRange.low}%` }}
        >
          {timeInRange.low > 5 ? `${timeInRange.low}%` : ''}
        </div>
        <div 
          className="range-segment target" 
          style={{ flexBasis: `${timeInRange.target}%` }}
        >
          {timeInRange.target > 5 ? `${timeInRange.target}%` : ''}
        </div>
        <div 
          className="range-segment high" 
          style={{ flexBasis: `${timeInRange.high}%` }}
        >
          {timeInRange.high > 5 ? `${timeInRange.high}%` : ''}
        </div>
        <div 
          className="range-segment very-high" 
          style={{ flexBasis: `${timeInRange.veryHigh}%` }}
        >
          {timeInRange.veryHigh > 5 ? `${timeInRange.veryHigh}%` : ''}
        </div>
      </div>
      
      <div className="range-legend">
        <div className="range-legend-item">
          <div className="range-color" style={{ backgroundColor: CHART_COLORS.veryLow }}></div>
          <span>Very Low (&lt;54): {timeInRange.veryLow}%</span>
        </div>
        <div className="range-legend-item">
          <div className="range-color" style={{ backgroundColor: CHART_COLORS.low }}></div>
          <span>Low (54-69): {timeInRange.low}%</span>
        </div>
        <div className="range-legend-item">
          <div className="range-color" style={{ backgroundColor: CHART_COLORS.target }}></div>
          <span>Target (70-180): {timeInRange.target}%</span>
        </div>
        <div className="range-legend-item">
          <div className="range-color" style={{ backgroundColor: CHART_COLORS.high }}></div>
          <span>High (181-250): {timeInRange.high}%</span>
        </div>
        <div className="range-legend-item">
          <div className="range-color" style={{ backgroundColor: CHART_COLORS.veryHigh }}></div>
          <span>Very High (&gt;250): {timeInRange.veryHigh}%</span>
        </div>
      </div>
    </div>
  );
};

// Main Dashboard Components
const MetricsPanel = ({ data }) => {
  const currentReading = data && data.length > 0 ? data[data.length - 1] : null;
  const averageGlucose = data && data.length > 0 
    ? Math.round(data.reduce((sum, d) => sum + d.realtimeValue, 0) / data.length)
    : 0;
  
  const timeInRange = useMemo(() => calculateTimeInRange(data), [data]);

  if (!currentReading) {
    return (
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="empty-state">
            <p>No glucose data available</p>
          </div>
        </div>
      </div>
    );
  }

  const glucoseCategory = getGlucoseCategory(currentReading.realtimeValue);

  return (
    <div className="metrics-grid">
      <div className="metric-card current-glucose">
        <h3>Current Glucose</h3>
        <div className={`metric-value glucose-value ${glucoseCategory}`}>
          {currentReading.realtimeValue}
          <span className="metric-unit">mg/dL</span>
        </div>
        <div className="metric-trend">
          <span className={`trend-arrow ${currentReading.trend.includes('Up') ? 'up' : currentReading.trend.includes('Down') ? 'down' : 'flat'}`}>
            {getTrendArrow(currentReading.trend)}
          </span>
          <span>{currentReading.trendRate} mg/dL/min</span>
        </div>
      </div>

      <div className="metric-card">
        <h3>Average Glucose</h3>
        <div className="metric-value">
          {averageGlucose}
          <span className="metric-unit">mg/dL</span>
        </div>
        <div className="metric-trend">
          <span>Last 24 hours</span>
        </div>
      </div>

      <div className="metric-card">
        <h3>Time in Range</h3>
        <div className="metric-value">
          {timeInRange.target}
          <span className="metric-unit">%</span>
        </div>
        <div className="metric-trend">
          <span>Target: 70-180 mg/dL</span>
        </div>
      </div>

      <div className="metric-card">
        <h3>Glucose Variability</h3>
        <div className="metric-value">
          {data && data.length > 0 
            ? Math.round((Math.sqrt(data.reduce((sum, d) => sum + Math.pow(d.realtimeValue - averageGlucose, 2), 0) / data.length) / averageGlucose) * 100)
            : 0}
          <span className="metric-unit">% CV</span>
        </div>
        <div className="metric-trend">
          <span>Coefficient of Variation</span>
        </div>
      </div>
    </div>
  );
};

const DeviceInfo = ({ deviceData }) => {
  if (!deviceData) {
    return (
      <div className="metric-card device-info">
        <h3>Device Information</h3>
        <div className="empty-state">
          <p>No device data available</p>
        </div>
      </div>
    );
  }

  const { displayDevice } = deviceData;

  return (
    <div className="metric-card device-info">
      <h3>Device Information</h3>
      
      <div className="device-detail">
        <span>Transmitter:</span>
        <span>{displayDevice?.transmitterGeneration?.toUpperCase() || 'Unknown'}</span>
      </div>
      
      <div className="device-detail">
        <span>Platform:</span>
        <span>{displayDevice?.displayApp || 'Unknown'}</span>
      </div>
      
      <div className="device-detail">
        <span>Software Version:</span>
        <span>{displayDevice?.softwareNumber || 'Unknown'}</span>
      </div>
      
      <div className="device-detail">
        <span>Connection:</span>
        <span className="status status--success">Connected</span>
      </div>
    </div>
  );
};

const AuthModal = ({ isOpen, onClose, onAuthenticate }) => {
  if (!isOpen) return null;

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose}></div>
      <div className="modal-content">
        <div className="modal-header">
          <h3>Connect to Dexcom</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <p>Connect your Dexcom account to view your continuous glucose monitoring data.</p>
          <div className="auth-options">
            <button 
              className="btn btn--primary"
              onClick={() => onAuthenticate('sandbox')}
            >
              Connect to Dexcom Sandbox
            </button>
            <button 
              className="btn btn--secondary"
              onClick={() => onAuthenticate('production')}
            >
              Connect to Dexcom Production
            </button>
          </div>
          <div className="auth-info">
            <small>
              Sandbox mode uses test data for demonstration. Production mode requires a real Dexcom account.
              For this demo, sandbox mode will simulate the authentication flow.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Application Component
const DexcomDashboard = () => {
  const auth = useAuth();
  const dexcomData = useDexcomData(auth);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isAppLoading, setIsAppLoading] = useState(true);

  // Handle OAuth callback
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    const error = urlParams.get('error');

    if (code) {
      auth.exchangeCodeForToken(code);
      // Clean URL
      window.history.replaceState({}, document.title, window.location.pathname);
    } else if (error) {
      console.error('OAuth error:', error);
    }
  }, []);

  // Hide loading screen after initial load
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsAppLoading(false);
      const loadingScreen = document.getElementById('loading-screen');
      if (loadingScreen) {
        loadingScreen.classList.add('hidden');
      }
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  const handleAuthenticate = (environment) => {
    setShowAuthModal(false);
    auth.startAuthFlow(environment);
  };

  const handleLogout = () => {
    auth.logout();
  };

  if (isAppLoading) {
    return null; // Loading screen is shown via HTML
  }

  if (!auth.isAuthenticated) {
    return (
      <div className="dashboard">
        <div className="dashboard-header">
          <div className="dashboard-header-content">
            <div className="dashboard-title">
              <h1>Dexcom Health Dashboard</h1>
            </div>
            <div className="connection-status">
              <div className="status-indicator disconnected"></div>
              <span>Not Connected</span>
            </div>
          </div>
        </div>
        
        <div className="dashboard-content">
          <div className="empty-state">
            <div className="empty-state-icon">🔗</div>
            <h2>Connect Your Dexcom Account</h2>
            <p>To view your continuous glucose monitoring data, please connect your Dexcom account.</p>
            <button 
              className="btn btn--primary btn--lg"
              onClick={() => setShowAuthModal(true)}
            >
              Connect to Dexcom
            </button>
          </div>
        </div>

        <AuthModal 
          isOpen={showAuthModal}
          onClose={() => setShowAuthModal(false)}
          onAuthenticate={handleAuthenticate}
        />
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="dashboard-header-content">
          <div className="dashboard-title">
            <h1>Dexcom Health Dashboard</h1>
            <span className="status status--success">
              {auth.environment === 'sandbox' ? 'Sandbox Mode' : 'Connected'}
            </span>
          </div>
          <div className="connection-status">
            <div className="status-indicator"></div>
            <span>Connected to {auth.environment}</span>
            <button className="btn btn--outline btn--sm" onClick={handleLogout}>
              Disconnect
            </button>
          </div>
        </div>
      </div>
      
      <div className="dashboard-content">
        {auth.error && (
          <div className="error-state">
            <div className="status status--error">Authentication Error</div>
            <p>{auth.error}</p>
          </div>
        )}

        {dexcomData.error && (
          <div className="error-state">
            <div className="status status--error">Data Error</div>
            <p>{dexcomData.error}</p>
          </div>
        )}

        <MetricsPanel data={dexcomData.egvData} />

        <div className="charts-section">
          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-header">
                <h3>Glucose Trend</h3>
                <div className="chart-controls">
                  <span className="status status--info">Last 24 Hours</span>
                  {dexcomData.isLoading && <div className="spinner"></div>}
                </div>
              </div>
              <div className="chart-container">
                <GlucoseChart data={dexcomData.egvData} />
              </div>
            </div>

            <div className="chart-card">
              <div className="chart-header">
                <h3>Time in Range</h3>
              </div>
              <div className="chart-container small">
                <TimeInRangeChart data={dexcomData.egvData} />
              </div>
              <DeviceInfo deviceData={dexcomData.deviceData} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Global functions for modal controls (referenced in HTML)
window.closeAuthModal = () => {
  const modal = document.getElementById('auth-modal');
  if (modal) modal.classList.add('hidden');
};

window.closeErrorModal = () => {
  const modal = document.getElementById('error-modal');
  if (modal) modal.classList.add('hidden');
};

window.startDexcomAuth = (environment) => {
  // This would trigger authentication in the React app
  console.log('Starting Dexcom authentication for:', environment);
};

// Render the application
ReactDOM.render(<DexcomDashboard />, document.getElementById('root'));