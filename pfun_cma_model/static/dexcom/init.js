// pfun_cma_model/static/dexcom/init.js
document.addEventListener("DOMContentLoaded", () => {
    // Auth Modal Handlers
    const authModal = document.getElementById('auth-modal');
    if (authModal) {
        // Close handlers
        const closeAuth = () => {
            if (typeof window.closeAuthModal === 'function') {
                window.closeAuthModal();
            } else {
                authModal.classList.add('hidden');
            }
        };

        const overlay = authModal.querySelector('.modal-overlay');
        if (overlay) overlay.addEventListener('click', closeAuth);

        const closeBtn = authModal.querySelector('.modal-close');
        if (closeBtn) closeBtn.addEventListener('click', closeAuth);

        // Start Auth handlers
        const sandboxBtn = authModal.querySelector('[data-env="sandbox"]');
        if (sandboxBtn) {
            sandboxBtn.addEventListener('click', () => {
                if (typeof window.startDexcomAuth === 'function') {
                    window.startDexcomAuth('sandbox');
                }
            });
        }

        const prodBtn = authModal.querySelector('[data-env="production"]');
        if (prodBtn) {
            prodBtn.addEventListener('click', () => {
                if (typeof window.startDexcomAuth === 'function') {
                    window.startDexcomAuth('production');
                }
            });
        }
    }

    // Error Modal Handlers
    const errorModal = document.getElementById('error-modal');
    if (errorModal) {
        const closeError = () => {
            if (typeof window.closeErrorModal === 'function') {
                window.closeErrorModal();
            } else {
                errorModal.classList.add('hidden');
            }
        };

        const overlay = errorModal.querySelector('.modal-overlay');
        if (overlay) overlay.addEventListener('click', closeError);

        const closeBtn = errorModal.querySelector('.modal-close');
        if (closeBtn) closeBtn.addEventListener('click', closeError);

        const okBtn = errorModal.querySelector('.btn--primary'); // The OK button
        if (okBtn) okBtn.addEventListener('click', closeError);
    }
});
