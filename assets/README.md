# Assets

## Logo

Drop your Myntra logo here named **`myntra_logo.png`** (also supported:
`.svg`, `.jpg`, `.jpeg`, `.webp`).

The nav bar in `app.py` automatically picks it up via `_logo_html()` and embeds
it (base64) so it works on Streamlit Cloud without serving static files.

If no logo file is present, the nav bar falls back to a CSS "Myntra" wordmark,
so the app always renders.
