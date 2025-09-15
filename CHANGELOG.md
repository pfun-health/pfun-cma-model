# CHANGELOG.md

## 2025-09-15

------------

### chore: reorganize assets and documentation; add CHANGELOG.md

**Add new asset files for Dexcom dashboard, including HTML, CSS, JS, and documentation templates.** Reorganize static files by moving run-at-time CSS/JS to
appropriate directory and removing obsolete run-at-time demo template. Introduce CHANGELOG.md to improve project transparency. Removed `pfun_cma_model/templates/run-at-time-demo.html` as it is no longer needed.

------------

### chore: implement theme toggler

**(Gemini)**
I've implemented the theme toggler for you. Here's a summary of the changes:
- `pfun_cma_model/static/style.css`: I've merged the dark and light theme styles from `pfun_cma_model/static/dexcom/style.css' into this file. This makes the theme variables available globally.
- `pfun_cma_model/templates/base.html`: I've added a "Toggle Theme" button to the navigation bar. I've included a script that manages theme switching. It automatically detects the user's preferred color scheme and allows them to toggle between dark and light modes. The chosen theme is saved in the browser's local storage, so it will be remembered on their next visit.
