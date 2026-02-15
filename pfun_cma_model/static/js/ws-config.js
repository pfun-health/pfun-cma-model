// pfun_cma_model/static/js/ws-config.js
// WebSocket Configuration
window.wsConfig = {
    scheme: window.location.protocol === 'https:' ? 'wss' : 'ws',
    host: window.location.hostname,
    port: window.location.port || (window.location.protocol === 'https:' ? 443 : 80)
};
window.wsUrl = `${window.wsConfig.scheme}://${window.wsConfig.host}:${window.wsConfig.port}`;
