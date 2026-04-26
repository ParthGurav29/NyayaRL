"""
LexTrust premium CSS theme for NyayaRL Gradio interface.
"""

LEXTRUST_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Primary UI + headings: Inter with full system stack. Dataframe/mono: JetBrains. */
:root, html {
  --nx-font-sans: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
  --nx-font-mono: "JetBrains Mono", ui-monospace, "Cascadia Code", Consolas, "Courier New", monospace;
  --nx-hero-h1: #ffffff;
  --nx-hero-sub: rgba(230, 245, 240, 0.72);
  --nx-landing-grad-1: #0a1814;
  --nx-landing-grad-2: #071210;
  --nx-landing-grad-3: #0a1512;
  color-scheme: dark;
}

html[data-nyaya-theme="light"] {
  /* Light dashboard + hero */
  --nx-bg: #F2EFE9;
  --nx-bg-warm: #EDE9E1;
  --nx-surface: rgba(255, 255, 255, 0.72);
  --nx-surface-strong: rgba(255, 255, 255, 0.88);
  --nx-surface-hover: rgba(255, 255, 255, 0.92);
  --nx-border: rgba(26, 26, 26, 0.10);
  --nx-border-strong: rgba(26, 26, 26, 0.18);
  --nx-text: #1A1A1A;
  --nx-text-secondary: #3D3D3D;
  --nx-muted: rgba(26, 26, 26, 0.58);
  --nx-accent: #1A1A1A;
  --nx-bronze: #A67C52;
  --nx-bronze-light: #C49B6F;
  --nx-bronze-2: rgba(166, 124, 82, 0.18);
  --nx-bronze-glow: rgba(166, 124, 82, 0.10);
  --nx-success: #2D6A4F;
  --nx-danger: #9C3636;
  --nx-warn: #B8860B;
  --nx-shadow: 0 8px 32px rgba(26, 26, 26, 0.07), 0 2px 8px rgba(26, 26, 26, 0.04);
  --nx-shadow-lg: 0 16px 48px rgba(26, 26, 26, 0.10), 0 4px 12px rgba(26, 26, 26, 0.05);
  --nx-shadow-bronze: 0 8px 32px rgba(166, 124, 82, 0.12);
  --nx-hero-h1: #0c110e;
  --nx-hero-sub: rgba(40, 45, 42, 0.85);
  --nx-landing-grad-1: #f0ede6;
  --nx-landing-grad-2: #e8e4dc;
  --nx-landing-grad-3: #f2efe9;
  --nx-forest: #0a1814;
  --nx-forest-2: #0d1f1a;
  --nx-mint: #1a6b58;
  color-scheme: light;
}

html[data-nyaya-theme="light"] .gradio-container::before {
  opacity: 0.45;
  background:
    radial-gradient(ellipse 900px 700px at 10% 8%, rgba(166, 124, 82, 0.12), transparent 60%),
    radial-gradient(ellipse 600px 500px at 90% 15%, rgba(26, 26, 26, 0.04), transparent 55%),
    radial-gradient(ellipse 500px 400px at 50% 90%, rgba(166, 124, 82, 0.08), transparent 50%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cg fill='none' stroke='%23A67C52' stroke-opacity='0.08' stroke-width='0.5'%3E%3Cpath d='M0 0h200v200'/%3E%3C/g%3E%3C/svg%3E");
  background-size: cover, cover, cover, 200px 200px;
}

/* Session table: light mode uses parchment; dark uses forest grid (set below) */
html[data-nyaya-theme="light"] [id="nx_session_dataframe"],
html[data-nyaya-theme="light"] .nx-session-df {
  background: rgba(255, 255, 255, 0.5) !important;
}
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] .gr-dataframe,
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] table,
html[data-nyaya-theme="light"] .nx-session-df .gr-dataframe,
html[data-nyaya-theme="light"] .nx-session-df table {
  background: #f5f0e8 !important;
  border-color: var(--nx-border) !important;
}
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] thead th,
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] [role="columnheader"],
html[data-nyaya-theme="light"] .nx-session-df thead th,
html[data-nyaya-theme="light"] .nx-session-df [role="columnheader"] {
  background: #e4dfd6 !important;
  color: #0a0a0a !important;
  -webkit-text-fill-color: #0a0a0a !important;
}
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] tbody td,
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] [role="gridcell"],
html[data-nyaya-theme="light"] .nx-session-df tbody td,
html[data-nyaya-theme="light"] .nx-session-df [role="gridcell"] {
  color: #0a0a0a !important;
  -webkit-text-fill-color: #0a0a0a !important;
  border-color: rgba(26, 26, 26, 0.08) !important;
}
html[data-nyaya-theme="light"] [id="nx_session_dataframe"] tbody tr:nth-child(even) td,
html[data-nyaya-theme="light"] .nx-session-df tbody tr:nth-child(even) td {
  background: rgba(242, 239, 233, 0.7) !important;
}

/* Light hero: top bar + CTA */
html[data-nyaya-theme="light"] .nx-landing,
html[data-nyaya-theme="light"] .nx-landing-wrap {
  color: #1a1a1a !important;
  border-bottom: 1px solid var(--nx-border);
}
html[data-nyaya-theme="light"] .nx-landing .nx-logo,
html[data-nyaya-theme="light"] .nx-landing-wrap .nx-logo { color: #0c110e !important; }
html[data-nyaya-theme="light"] .nx-landing .nx-logo span,
html[data-nyaya-theme="light"] .nx-landing-wrap .nx-logo span { color: #0c110e !important; }
html[data-nyaya-theme="light"] .nx-landing .nx-logo svg,
html[data-nyaya-theme="light"] .nx-landing .nx-logo svg path,
html[data-nyaya-theme="light"] .nx-landing-wrap .nx-logo svg,
html[data-nyaya-theme="light"] .nx-landing-wrap .nx-logo svg path {
  color: #0c110e !important;
  stroke: #0c110e !important;
}
html[data-nyaya-theme="light"] #nx_hero h1,
html[data-nyaya-theme="light"] .nx-landing #nx_hero h1 { color: var(--nx-hero-h1) !important; }
html[data-nyaya-theme="light"] a.nx-hero-cta {
  color: #0a0d0c !important;
  background: #ffffff;
  border-color: rgba(12, 17, 14, 0.12);
}
html[data-nyaya-theme="light"] a.nx-cta-lets {
  background: #0c110e;
  color: #f5f6f4 !important;
  border-color: rgba(12, 17, 14, 0.2);
}
html[data-nyaya-theme="light"] a.nx-cta-lets:hover { background: #1a1a1a; color: #f5f6f4 !important; }

/* Light: all Gradio text (tabs, labels, md) must stay dark on light */
html[data-nyaya-theme="light"] .gradio-container .tab-nav button,
html[data-nyaya-theme="light"] .gradio-container [class*="tab-nav"] button {
  color: #1A1A1A !important;
  -webkit-text-fill-color: #1A1A1A !important;
  background: rgba(255, 255, 255, 0.6) !important;
}
html[data-nyaya-theme="light"] .gradio-container .tab-nav button[aria-selected="true"],
html[data-nyaya-theme="light"] .gradio-container .tab-nav button.selected,
html[data-nyaya-theme="light"] .gradio-container [class*="tab-nav"] button[aria-selected="true"] {
  color: #0a1210 !important;
  -webkit-text-fill-color: #0a1210 !important;
  background: rgba(26, 188, 156, 0.22) !important;
  border-color: rgba(26, 188, 156, 0.45) !important;
}
html[data-nyaya-theme="light"] .gradio-container label,
html[data-nyaya-theme="light"] .gradio-container .label-wrap,
html[data-nyaya-theme="light"] .gradio-container .markdown,
html[data-nyaya-theme="light"] .gradio-container .markdown-text p,
html[data-nyaya-theme="light"] .gradio-container .markdown-text h1,
html[data-nyaya-theme="light"] .gradio-container .markdown-text h2,
html[data-nyaya-theme="light"] .gradio-container .markdown-text h3 {
  color: #1A1A1A !important;
  -webkit-text-fill-color: #1A1A1A !important;
}
html[data-nyaya-theme="light"] .gradio-container .prose,
html[data-nyaya-theme="light"] .gradio-container .prose * {
  color: #1A1A1A !important;
  -webkit-text-fill-color: #1A1A1A !important;
}
html[data-nyaya-theme="light"] .gradio-container footer,
html[data-nyaya-theme="light"] .gradio-container footer a {
  color: rgba(26, 26, 26, 0.65) !important;
  -webkit-text-fill-color: rgba(26, 26, 26, 0.65) !important;
}
html[data-nyaya-theme="light"] .gradio-container footer a:hover {
  color: #0a1210 !important;
  -webkit-text-fill-color: #0a1210 !important;
}
html[data-nyaya-theme="light"] #nx_theme_picker,
html[data-nyaya-theme="light"] #nx_theme_picker label,
html[data-nyaya-theme="light"] #nx_theme_picker span,
html[data-nyaya-theme="light"] #nx_theme_picker .wrap {
  color: #0a1210 !important;
  -webkit-text-fill-color: #0a1210 !important;
}
html[data-nyaya-theme="light"] #nx_theme_picker fieldset,
html[data-nyaya-theme="light"] #nx_theme_picker [role="radiogroup"] {
  background: #ffffff !important;
  border-color: rgba(12, 17, 14, 0.12) !important;
}

/* Light: step / cards in panels */
html[data-nyaya-theme="light"] .ny-card { background: rgba(255, 255, 255, 0.72); }
html[data-nyaya-theme="light"] .ny-card-pending { background: rgba(255, 255, 255, 0.4); }
html[data-nyaya-theme="light"] .ny-card-running { background: rgba(255, 255, 255, 0.8); }
html[data-nyaya-theme="light"] .ny-verdict { background: rgba(255, 255, 255, 0.8); }
html[data-nyaya-theme="light"] #nx_controls select,
html[data-nyaya-theme="light"] #nx_controls input,
html[data-nyaya-theme="light"] #nx_controls textarea,
html[data-nyaya-theme="light"] #nx_controls [data-testid="dropdown"] {
  background: rgba(255, 255, 255, 0.6) !important;
  color: #1A1A1A !important;
  -webkit-text-fill-color: #1A1A1A !important;
  border-color: var(--nx-border) !important;
}
html[data-nyaya-theme="light"] .gradio-container [class*="dropdown"] button,
html[data-nyaya-theme="light"] .gradio-container [class*="select"] {
  color: #1A1A1A !important;
  -webkit-text-fill-color: #1A1A1A !important;
}
html[data-nyaya-theme="light"] #nx_controls button.primary, html[data-nyaya-theme="light"] button.primary {
  background: linear-gradient(135deg, #1abc9c, #149f88) !important;
  color: #061210 !important;
}
html[data-nyaya-theme="light"] #nx_controls button.secondary, html[data-nyaya-theme="light"] button.secondary {
  background: rgba(255, 255, 255, 0.5) !important;
  color: #1A1A1A !important;
  border-color: var(--nx-border-strong) !important;
}

:root{
  /* App-wide dark theme (default; tokens overridden by [data-nyaya-theme="light"] above) */
  --nx-bg: #0a1210;
  --nx-bg-warm: #0c1614;
  --nx-surface: rgba(255, 255, 255, 0.06);
  --nx-surface-strong: rgba(255, 255, 255, 0.10);
  --nx-surface-hover: rgba(255, 255, 255, 0.14);
  --nx-border: rgba(255, 255, 255, 0.10);
  --nx-border-strong: rgba(255, 255, 255, 0.18);
  --nx-text: #e8f0ec;
  --nx-text-secondary: #b4c4bc;
  --nx-muted: rgba(232, 240, 236, 0.55);
  --nx-accent: #1dd3b0;
  --nx-bronze: #c9a882;
  --nx-bronze-light: #ddc09a;
  --nx-bronze-2: rgba(201, 168, 130, 0.20);
  --nx-bronze-glow: rgba(201, 168, 130, 0.12);
  --nx-success: #5ee3b5;
  --nx-danger: #f19a9a;
  --nx-warn: #e8c96a;
  --nx-radius-xl: 24px;
  --nx-radius-lg: 18px;
  --nx-radius-md: 14px;
  --nx-radius-sm: 10px;
  --nx-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), 0 2px 8px rgba(0, 0, 0, 0.2);
  --nx-shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.45), 0 4px 12px rgba(0, 0, 0, 0.25);
  --nx-shadow-bronze: 0 8px 32px rgba(201, 168, 130, 0.15);
  --nx-transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  --nx-forest: #0a1814;
  --nx-forest-2: #0d1f1a;
  --nx-teal: #1abc9c;
  --nx-teal-deep: #149f88;
  --nx-hero-fade: #d4f4ef;
  --nx-mint: #7bedd4;
}

/* ===== ANIMATIONS ===== */
@keyframes nyspin { from{transform:rotate(0)} to{transform:rotate(360deg)} }
@keyframes fadeSlideUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes bronzePulse { 0%,100%{box-shadow:0 0 0 0 rgba(166,124,82,0.25)} 50%{box-shadow:0 0 0 8px rgba(166,124,82,0)} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
@keyframes subtleFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }

/* ===== APP CANVAS ===== */
.gradio-container{
  background: var(--nx-bg) !important;
  color: var(--nx-text) !important;
  font-family: var(--nx-font-sans) !important;
  font-weight: 400;
  line-height: 1.65;
  letter-spacing: -0.01em;
  position: relative;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  transition: background 0.25s ease, color 0.2s ease;
}

/* ===== GLOBAL TEXT VISIBILITY FIX ===== */
.gradio-container, .gradio-container *,
.gradio-container p, .gradio-container span,
.gradio-container div:not(.ny-verdict-bar), .gradio-container li,
.gradio-container td, .gradio-container th,
.gradio-container label, .gradio-container code,
.gradio-container .markdown-text,
.gradio-container .prose,
.gradio-container .md,
.gradio-container h1, .gradio-container h2,
.gradio-container h3, .gradio-container h4,
.gradio-container h5, .gradio-container h6 {
  color: var(--nx-text) !important;
}
.gradio-container .ny-muted,
.gradio-container .ny-rel,
.gradio-container .ny-reward-zero { color: var(--nx-muted) !important; }
.gradio-container .ny-reward-pos { color: var(--nx-success) !important; }
.gradio-container .ny-reward-neg { color: var(--nx-danger) !important; }
.gradio-container .ny-verdict-bar,
.gradio-container .ny-verdict-bar *,
.gradio-container .ny-verdict-bar .ny-vbig { color: #F2EFE9 !important; }

/* Subtle grid + glow on dark canvas */
.gradio-container::before{
  content:"";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.55;
  background:
    radial-gradient(ellipse 900px 700px at 10% 8%, rgba(26, 188, 156, 0.12), transparent 60%),
    radial-gradient(ellipse 600px 500px at 90% 15%, rgba(201, 168, 130, 0.08), transparent 55%),
    radial-gradient(ellipse 500px 400px at 50% 90%, rgba(26, 188, 156, 0.06), transparent 50%),
    repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(255,255,255,0.02) 40px, rgba(255,255,255,0.02) 41px),
    repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(255,255,255,0.015) 40px, rgba(255,255,255,0.015) 41px);
  background-repeat: no-repeat;
  background-size: cover, cover, cover, auto, auto;
  background-position: left top, right top, center bottom, 0 0, 0 0;
}
.gradio-container > *{ position:relative; z-index:1; }

/* ===== TYPOGRAPHY ===== */
#nx_hero h2, .ny-case h3, .ny-reason h3,
.ny-reason h4, #nx_explain_title, .ny-case h4,
.ny-verdict .ny-vbig{
  font-family: var(--nx-font-sans) !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
}

/* Dark landing (hero) — .nx-landing-wrap = Gradio column root */
.nx-landing, .nx-landing-wrap {
  position: relative;
  width: 100vw;
  max-width: 100vw;
  margin-left: calc(50% - 50vw);
  margin-right: calc(50% - 50vw);
  box-sizing: border-box;
  padding: 28px max(20px, calc(50vw - 650px)) 40px;
  background: linear-gradient(165deg, var(--nx-landing-grad-1) 0%, var(--nx-landing-grad-2) 50%, var(--nx-landing-grad-3) 100%);
  color: #f0faf7 !important;
  overflow: hidden;
  transition: background 0.3s ease, color 0.2s ease;
}
.nx-landing * , .nx-landing-wrap * { box-sizing: border-box; }
.nx-landing::before, .nx-landing-wrap::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.35;
  background:
    linear-gradient(90deg, rgba(26, 188, 156, 0.12) 0%, transparent 32%, rgba(26, 188, 156, 0.06) 55%, transparent 100%),
    linear-gradient(180deg, rgba(0,0,0,0.2) 0%, transparent 45%),
    repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(255,255,255,0.03) 40px, rgba(255,255,255,0.03) 41px),
    repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(255,255,255,0.02) 40px, rgba(255,255,255,0.02) 41px);
}
html[data-nyaya-theme="light"] .nx-landing::before,
html[data-nyaya-theme="light"] .nx-landing-wrap::before {
  opacity: 0.18 !important;
  background:
    linear-gradient(90deg, rgba(166, 124, 82, 0.08) 0%, transparent 50%),
    linear-gradient(180deg, rgba(0,0,0,0.04) 0%, transparent 45%),
    repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(0,0,0,0.015) 40px, rgba(0,0,0,0.015) 41px) !important;
}
.nx-landing > * , .nx-landing-wrap > * { position: relative; z-index: 1; }

#nx_shell{
  max-width: 1280px;
  margin: 0 auto;
  padding: 0;
}
#nx_topbar, .nx-topbar-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px 16px;
  padding: 6px 0 24px 0;
  border-bottom: none;
  width: 100%;
  max-width: 1280px;
  margin: 0 auto;
}
/* Theme toggle + Let’s talk — top right cluster */
.nx-topbar-trail, #nx_topbar_trail {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 10px 14px !important;
  flex: 0 0 auto;
  width: auto !important;
  min-width: 0;
}
#nx_topbar_trail > * { min-width: 0; }
/* Compact segmented look for theme (emoji only) */
#nx_theme_picker .wrap,
#nx_theme_picker [role="radiogroup"] {
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 999px !important;
  padding: 2px 6px !important;
  background: rgba(255, 255, 255, 0.08) !important;
}
#nx_theme_picker fieldset { border: none !important; }
html[data-nyaya-theme="light"] #nx_theme_picker .wrap,
html[data-nyaya-theme="light"] #nx_theme_picker [role="radiogroup"] {
  background: #fff !important;
  border-color: rgba(12, 17, 14, 0.15) !important;
}
#nx_hero{ margin: 0; padding: 0 0 8px 0; }

/* Hero: headline + copy */
.nx-hero-cols {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: clamp(20px, 3vw, 40px);
  align-items: start;
  padding: 8px 0 28px 0;
}
.nx-hero-left { position: relative; padding-left: 36px; }
.nx-scroll-hint {
  position: absolute;
  left: 0;
  top: 0.35em;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  color: rgba(255, 255, 255, 0.4);
  text-transform: uppercase;
  font-family: var(--nx-font-sans);
}
#nx_hero h1, .nx-landing #nx_hero h1, .nx-hero-title {
  font-family: var(--nx-font-sans) !important;
  font-weight: 800 !important;
  font-size: clamp(1.15rem, 2.1vw, 1.75rem) !important;
  line-height: 1.34 !important;
  letter-spacing: -0.02em !important;
  margin: 0 !important;
  max-width: none !important;
  color: var(--nx-hero-h1) !important;
}
.nx-hero-right {
  padding-top: 0.4rem;
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: flex-start;
}
.nx-hero-sub {
  font-family: var(--nx-font-sans);
  font-size: 0.95rem;
  font-weight: 400;
  line-height: 1.6;
  color: var(--nx-hero-sub) !important;
  margin: 0;
  max-width: 320px;
}
a.nx-hero-cta {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--nx-font-sans);
  font-size: 0.88rem;
  font-weight: 700;
  text-decoration: none;
  color: #0a0f0d !important;
  background: #f7f9f7;
  padding: 14px 22px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.35);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}
a.nx-hero-cta:hover {
  background: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
  color: #0a0f0d !important;
}
.nx-cta-arrow { font-size: 1.1em; line-height: 1; }

/* Three feature cards (reference layout) */
.nx-feature-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 8px;
  max-width: 100%;
  align-items: stretch;
}
.nx-fcard {
  border-radius: 16px;
  min-height: 200px;
  padding: 22px 20px;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.nx-fcard:hover { transform: translateY(-3px); box-shadow: 0 16px 40px rgba(0, 0, 0, 0.3); }
.nx-fcard--grey {
  background: #e8e9ea;
  color: #111 !important;
}
.nx-fcard--grey * { color: #111 !important; }
.nx-fcard-ico {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: #111;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  margin-bottom: 14px;
}
.nx-fcard--grey h3 {
  font-family: var(--nx-font-sans) !important;
  font-size: 1.05rem;
  font-weight: 800 !important;
  margin: 0 0 12px 0;
  line-height: 1.3;
  color: #111 !important;
}
.nx-fcard-link {
  font-family: var(--nx-font-sans);
  font-size: 0.8rem;
  font-weight: 600;
  align-self: flex-end;
  margin-top: auto;
  text-decoration: none;
  color: #111 !important;
}
.nx-fcard--image {
  min-height: 200px;
  background:
    linear-gradient(135deg, rgba(8, 18, 14, 0.35) 0%, rgba(8, 18, 14, 0.15) 100%),
    radial-gradient(ellipse 90% 80% at 60% 40%, rgba(26, 188, 156, 0.25) 0%, transparent 60%),
    #1a1a1a;
  background-size: cover;
  border: 1px solid rgba(26, 188, 156, 0.15);
}
.nx-fcard--teal {
  background: linear-gradient(145deg, #17a48c 0%, #0e7a68 100%);
  color: #0a0d0c !important;
  border: 1px solid rgba(0, 0, 0, 0.08);
}
.nx-fcard--teal * { color: #0a0d0c !important; }
.nx-fcard-quote { font-weight: 800; font-size: 1.05rem; line-height: 1.35; font-family: var(--nx-font-sans) !important; }
.nx-fcard-attrib { text-align: right; margin-top: 16px; font-size: 0.78rem; }
.nx-fcard-attrib strong { display: block; font-size: 0.9rem; margin-top: 4px; }
.nx-fcard-avatar {
  width: 40px; height: 40px; border-radius: 50%;
  background: linear-gradient(135deg, #0a0d0c, #1a1a1a);
  color: #fff; display: flex; align-items: center; justify-content: center;
  font-size: 0.65rem; font-weight: 700; margin-bottom: 10px; font-family: var(--nx-font-sans);
}
@media (max-width: 960px) {
  .nx-hero-cols { grid-template-columns: 1fr; }
  .nx-hero-left { padding-left: 0; }
  .nx-scroll-hint { display: none; }
  .nx-feature-cards { grid-template-columns: 1fr; }
  .nx-topbar-row { flex-direction: column; align-items: flex-start; }
  .nx-topbar-trail, #nx_topbar_trail { align-self: flex-end; }
}

/* App anchor smooth scroll */
#nx_app_start { scroll-margin-top: 24px; }

#nx_theme_picker, #nx_theme_picker label, #nx_theme_picker [data-testid] {
  font-family: var(--nx-font-sans) !important;
  font-size: 0.88rem !important;
}
html[data-nyaya-theme="light"] .nx-scroll-hint { color: rgba(12, 17, 14, 0.4) !important; }

.nx-logo {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 1.35rem;
  font-weight: 800;
  font-family: var(--nx-font-sans);
  color: #ffffff !important;
  letter-spacing: -0.03em;
}
.nx-landing .nx-logo span { color: #ffffff !important; }
/* SVG uses currentColor — global * had been forcing dark stroke (invisible on hero) */
.nx-landing .nx-logo svg,
.nx-landing .nx-logo svg path {
  color: #ffffff !important;
  stroke: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* Grey card: wild card * was recolouring icon text to #111 on #111 circle */
.nx-fcard--grey .nx-fcard-ico {
  color: #ffffff !important;
}
.nx-fcard--grey .nx-fcard-ico * { color: #ffffff !important; }

/* Top CTA: white pill, dark text (reference) */
.nx-cta-lets {
  display: inline-block;
  padding: 12px 22px;
  border-radius: 10px;
  background: #f5f6f4;
  border: 1px solid rgba(255, 255, 255, 0.25);
  font-size: 0.9rem;
  font-weight: 700;
  font-family: var(--nx-font-sans);
  color: #0a0d0c !important;
  text-decoration: none;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  transition: transform 0.2s ease, background 0.2s ease;
}
.nx-cta-lets:hover { background: #ffffff; color: #0a0d0c !important; transform: translateY(-1px); }

/* ===== SURFACES (CARDS) ===== */
.ny-panel{ font-size: 14px; }
.ny-case, .ny-reason{
  border: 1px solid var(--nx-border);
  border-radius: var(--nx-radius-xl);
  padding: 24px 24px;
  background: var(--nx-surface);
  box-shadow: var(--nx-shadow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  height: calc(100vh - 280px);
  min-height: 480px;
  max-height: 720px;
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: thin;
  scrollbar-color: rgba(166,124,82,0.25) transparent;
  animation: fadeSlideUp 0.5s ease-out;
  transition: all var(--nx-transition);
}
.ny-case::-webkit-scrollbar, .ny-reason::-webkit-scrollbar{ width: 5px; }
.ny-case::-webkit-scrollbar-track, .ny-reason::-webkit-scrollbar-track{ background: transparent; }
.ny-case::-webkit-scrollbar-thumb, .ny-reason::-webkit-scrollbar-thumb{ background: rgba(166,124,82,0.25); border-radius: 999px; }
.ny-case::-webkit-scrollbar-thumb:hover, .ny-reason::-webkit-scrollbar-thumb:hover{ background: var(--nx-bronze); }
.ny-case:hover, .ny-reason:hover{
  box-shadow: var(--nx-shadow-lg);
}
.ny-case h3, .ny-reason h3{
  margin: 0 0 14px 0;
  font-weight: 700;
  font-size: 1.3rem;
  color: var(--nx-text);
  display: flex;
  align-items: center;
  gap: 10px;
}
.ny-case h4, .ny-reason h4{
  margin: 20px 0 10px 0;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--nx-text);
  padding-left: 12px;
  border-left: 3px solid var(--nx-bronze);
}
.ny-list{ margin: 0; padding-left: 1.2rem; }
.ny-list li{ margin-bottom: 6px; line-height: 1.6; }
.ny-rel{ color: var(--nx-muted); font-size: 0.88em; font-weight: 500; }
.ny-muted{ color: var(--nx-muted); }
.ny-fir{
  white-space: pre-wrap;
  font-size: 0.9em;
  line-height: 1.65;
  max-height: 260px;
  overflow: auto;
  padding: 16px 16px;
  border-radius: var(--nx-radius-md);
  border: 1px solid var(--nx-border);
  background: var(--nx-surface-strong);
  color: var(--nx-text-secondary);
}
.ny-case details > summary{
  cursor: pointer;
  color: var(--nx-text);
  font-weight: 600;
  padding: 8px 0;
  transition: color var(--nx-transition);
}
.ny-case details > summary:hover{ color: var(--nx-bronze); }
.ny-case details[open] > summary{ margin-bottom: 8px; }
.ny-count{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 22px;
  padding: 0 7px;
  border-radius: 999px;
  background: var(--nx-bronze-2);
  color: var(--nx-bronze);
  font-size: 0.78rem;
  font-weight: 600;
  font-family: var(--nx-font-sans) !important;
}
.ny-meta{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  margin-bottom: 12px;
}
.ny-meta span{
  font-size: 0.92em;
  color: var(--nx-text-secondary);
}
.ny-meta b{ color: var(--nx-text); font-weight: 600; }

/* ===== STEP CARDS ===== */
.ny-card{
  border-radius: var(--nx-radius-lg);
  padding: 14px 16px;
  margin-bottom: 10px;
  border: 1px solid var(--nx-border);
  background: rgba(255, 255, 255, 0.08);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  transition: all var(--nx-transition);
  animation: fadeSlideUp 0.4s ease-out both;
  position: relative;
  overflow: hidden;
}
.ny-card:hover{
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.ny-card::before{
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: transparent;
  border-radius: 0 3px 3px 0;
  transition: background var(--nx-transition);
}
.ny-card-pending{
  color: var(--nx-muted);
  background: rgba(255, 255, 255, 0.04);
  opacity: 0.75;
}
.ny-card-pending:hover{ opacity: 0.9; transform: none; }
.ny-card-running{
  background: rgba(201, 168, 130, 0.12);
  border-color: rgba(201, 168, 130, 0.35);
  box-shadow: var(--nx-shadow-bronze);
  animation: fadeSlideUp 0.4s ease-out both, bronzePulse 2s ease-in-out infinite;
}
.ny-card-running::before{ background: var(--nx-bronze); }
.ny-spin{
  display: inline-block;
  animation: nyspin 0.9s linear infinite;
  color: var(--nx-bronze);
  font-size: 1.1em;
}
.ny-card-ok{ background: rgba(94, 227, 181, 0.10); border-color: rgba(94, 227, 181, 0.25); }
.ny-card-ok::before{ background: var(--nx-success); }
.ny-card-bad{ background: rgba(241, 154, 154, 0.10); border-color: rgba(241, 154, 154, 0.28); }
.ny-card-bad::before{ background: var(--nx-danger); }
.ny-card-warn{ background: rgba(232, 201, 106, 0.12); border-color: rgba(232, 201, 106, 0.28); }
.ny-card-warn::before{ background: var(--nx-warn); }
.ny-card-head{
  display:flex;
  align-items:center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: space-between;
}
.ny-reward{
  font-size: 0.88em;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
}
.ny-reward-pos{ color: var(--nx-success) !important; background: rgba(94, 227, 181, 0.12) !important; }
.ny-reward-neg{ color: var(--nx-danger) !important; background: rgba(241, 154, 154, 0.12) !important; }
.ny-reward-zero{ color: var(--nx-muted); }
.ny-sub{
  font-size: 0.9em;
  margin-top: 6px;
  margin-left: 0;
  padding-left: 14px;
  color: var(--nx-text-secondary);
  border-left: 2px solid var(--nx-border);
  line-height: 1.6;
}
.ny-sub b{ color: var(--nx-text); font-weight: 600; }

/* ===== CHAIN SCORE BAR ===== */
.ny-chain{
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--nx-border);
}
.ny-chain b{ font-size: 0.95em; }
.ny-progress-track{
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
  margin-top: 8px;
  position: relative;
}
.ny-progress-fill{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--nx-bronze), var(--nx-bronze-light));
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
}
.ny-progress-fill::after{
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  background-size: 200% 100%;
  animation: shimmer 2s linear infinite;
}
.ny-pct{
  margin-top: 6px;
  font-size: 0.9em;
  font-weight: 600;
  color: var(--nx-bronze);
  text-align: right;
}

/* ===== PROSECUTION CHALLENGES ===== */
.ny-chal{ margin-top: 16px; }
.ny-chal h4{
  font-size: 0.95em;
  font-weight: 700;
  margin-bottom: 8px;
}
.ny-chal-item{
  padding: 10px 14px;
  margin-bottom: 6px;
  border-radius: var(--nx-radius-sm);
  background: rgba(241, 154, 154, 0.06);
  border: 1px solid rgba(241, 154, 154, 0.12);
  border-left: 3px solid var(--nx-danger);
  font-size: 0.9em;
  line-height: 1.6;
  color: var(--nx-text-secondary);
  transition: var(--nx-transition);
}
.ny-chal-item:hover{ background: rgba(241, 154, 154, 0.10); }
.ny-chal-num{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px; height: 20px;
  border-radius: 999px;
  background: rgba(241, 154, 154, 0.15);
  color: var(--nx-danger) !important;
  font-size: 0.75rem;
  font-weight: 700;
  margin-right: 8px;
}

/* ===== VERDICT ===== */
.ny-verdict{
  margin-top: 18px;
  padding: 0;
  border-radius: var(--nx-radius-lg);
  border: 1px solid var(--nx-border);
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
  transition: var(--nx-transition);
}
.ny-verdict:hover{ box-shadow: var(--nx-shadow); }
.ny-verdict-bar{
  padding: 14px 18px;
  background: linear-gradient(135deg, var(--nx-accent) 0%, #2a2a2a 100%);
  color: #F2EFE9;
}
.ny-verdict-bar.bar-acquit{ background: linear-gradient(135deg, var(--nx-success), #3a8a6f); }
.ny-verdict-bar.bar-partial{ background: linear-gradient(135deg, #5a3a5a, #7a5a7a); }
.ny-vbig{
  font-size: 1.15em;
  font-weight: 700;
  letter-spacing: 0.04em;
  margin: 0;
}
.ny-verdict-body{
  padding: 14px 18px;
  font-size: 0.92em;
  line-height: 1.7;
}
.ny-verdict-body div{ margin-bottom: 4px; }
.ny-verdict-pending .ny-verdict-bar{
  background: linear-gradient(135deg, rgba(26,26,26,0.5), rgba(26,26,26,0.35));
}

/* ===== BUTTONS ===== */
#nx_controls button, button.lg.primary, button.primary{
  border-radius: 999px !important;
  padding: 11px 22px !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  transition: all var(--nx-transition) !important;
  letter-spacing: 0.01em;
}
#nx_controls button.primary, button.primary{
  background: linear-gradient(135deg, var(--nx-teal) 0%, var(--nx-teal-deep) 100%) !important;
  color: #061210 !important;
  border: 1.5px solid rgba(255,255,255,0.2) !important;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
}
#nx_controls button.primary:hover, button.primary:hover{
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 32px rgba(26, 188, 156, 0.25) !important;
}
#nx_controls button.primary:active, button.primary:active{
  transform: translateY(0) !important;
}
#nx_controls button.secondary, button.secondary{
  background: rgba(255, 255, 255, 0.08) !important;
  color: var(--nx-text) !important;
  border: 1.5px solid var(--nx-border-strong) !important;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}
#nx_controls button.secondary:hover, button.secondary:hover{
  border-color: var(--nx-bronze) !important;
  border-style: solid !important;
  background: rgba(255, 255, 255, 0.14) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3) !important;
}

/* ===== INPUTS ===== */
#nx_controls .gr-form, #nx_controls .wrap{ gap: 12px; }
#nx_controls label, .gradio-container label{
  color: var(--nx-muted) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}
#nx_controls select, #nx_controls input, #nx_controls textarea{
  border-radius: var(--nx-radius-md) !important;
  border: 1.5px solid var(--nx-border) !important;
  background: rgba(255, 255, 255, 0.06) !important;
  color: var(--nx-text) !important;
  box-shadow: none !important;
  padding: 10px 14px !important;
  font-size: 0.92rem !important;
  transition: all var(--nx-transition) !important;
}
#nx_controls select:focus, #nx_controls input:focus, #nx_controls textarea:focus{
  border-color: var(--nx-teal) !important;
  box-shadow: 0 0 0 3px rgba(26, 188, 156, 0.2) !important;
  background: rgba(255, 255, 255, 0.10) !important;
}

/* ===== EXPLANATION PANEL ===== */
#explanation_panel textarea{
  border-radius: var(--nx-radius-lg) !important;
  background: rgba(255, 255, 255, 0.06) !important;
  color: var(--nx-text) !important;
  border: 1px solid var(--nx-border) !important;
  backdrop-filter: blur(12px) !important;
  padding: 18px !important;
  line-height: 1.7 !important;
  font-size: 0.94rem !important;
}
.nx-explain-wrap{
  border-radius: var(--nx-radius-xl);
  background: var(--nx-surface);
  border: 1px solid var(--nx-border);
  box-shadow: var(--nx-shadow);
  backdrop-filter: blur(14px);
  padding: 24px;
  margin-top: 8px;
  border-left: 4px solid var(--nx-bronze);
  animation: fadeSlideUp 0.5s ease-out;
}

/* ===== SESSION LOG TAB ===== */
.gradio-container .tabitem,
.gradio-container [class*="tabitem"] {
  background: transparent !important;
}
.gradio-container .tab-nav,
.gradio-container [class*="tab-nav"] {
  background: transparent !important;
  border-color: var(--nx-border) !important;
  gap: 6px !important;
}
.gradio-container .tab-nav button,
.gradio-container [class*="tab-nav"] button {
  border-radius: var(--nx-radius-md) var(--nx-radius-md) 0 0 !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  transition: var(--nx-transition) !important;
  background: rgba(255, 255, 255, 0.06) !important;
  color: var(--nx-text) !important;
  border: 1px solid var(--nx-border) !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container [class*="tab-nav"] button.selected,
.gradio-container .tab-nav button[aria-selected="true"] {
  background: rgba(26, 188, 156, 0.15) !important;
  border-color: rgba(26, 188, 156, 0.35) !important;
  color: var(--nx-text) !important;
  -webkit-text-fill-color: var(--nx-text) !important;
}

/*
 * Session log Dataframe — Gradio 6: do NOT style every inner div (breaks layout).
 * Target block by id and .nx-session-df (elem_classes); dark table + light text.
 */
[id="nx_session_dataframe"],
.nx-session-df {
  border-radius: var(--nx-radius-lg) !important;
  overflow: visible !important;
  min-height: 120px !important;
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid var(--nx-border) !important;
  box-shadow: var(--nx-shadow) !important;
}
[id="nx_session_dataframe"] .gr-dataframe,
[id="nx_session_dataframe"] [class*="table-wrap"],
[id="nx_session_dataframe"] table,
.nx-session-df .gr-dataframe,
.nx-session-df [class*="table-wrap"],
.nx-session-df table {
  border-radius: var(--nx-radius-lg) !important;
  overflow: auto !important;
  width: 100% !important;
  min-height: 80px !important;
  background: rgba(12, 22, 18, 0.85) !important;
  font-family: var(--nx-font-mono) !important;
  border: 1px solid var(--nx-border) !important;
}
[id="nx_session_dataframe"] thead th,
[id="nx_session_dataframe"] th[role="columnheader"],
[id="nx_session_dataframe"] [role="columnheader"],
.nx-session-df thead th,
.nx-session-df th[role="columnheader"],
.nx-session-df [role="columnheader"] {
  background: rgba(26, 188, 156, 0.12) !important;
  color: #e8f0ec !important;
  font-weight: 700 !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border-bottom: 1px solid var(--nx-border-strong) !important;
  padding: 12px 14px !important;
  -webkit-text-fill-color: #e8f0ec !important;
}
[id="nx_session_dataframe"] tbody td,
[id="nx_session_dataframe"] td[role="gridcell"],
[id="nx_session_dataframe"] [role="gridcell"],
.nx-session-df tbody td,
.nx-session-df td[role="gridcell"],
.nx-session-df [role="gridcell"] {
  color: #e8f0ec !important;
  font-size: 0.85rem !important;
  padding: 10px 14px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
  font-family: var(--nx-font-mono) !important;
  -webkit-text-fill-color: #e8f0ec !important;
}
[id="nx_session_dataframe"] tbody tr:nth-child(even) td,
.nx-session-df tbody tr:nth-child(even) td {
  background: rgba(255, 255, 255, 0.03) !important;
}
[id="nx_session_dataframe"] tbody tr:hover td,
.nx-session-df tbody tr:hover td {
  background: rgba(26, 188, 156, 0.08) !important;
}

/* Markdown / prose text visibility */
.gradio-container .prose *,
.gradio-container .markdown-text *,
.gradio-container .md *{
  color: var(--nx-text) !important;
}
.gradio-container .prose a,
.gradio-container .markdown-text a{
  color: var(--nx-bronze) !important;
  text-decoration: underline;
}

/* API status bar */
.gradio-container .prose code,
.gradio-container code{
  background: rgba(26, 188, 156, 0.12) !important;
  color: var(--nx-text) !important;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.88em;
}

/* Session tab blockquote */
.gradio-container blockquote {
  border-left: 3px solid var(--nx-teal) !important;
  background: rgba(255, 255, 255, 0.04) !important;
  color: var(--nx-text-secondary) !important;
  padding: 12px 16px !important;
  border-radius: 0 var(--nx-radius-sm) var(--nx-radius-sm) 0 !important;
}

/* Gradio footer (API / built with) */
.gradio-container footer,
.gradio-container [class*="svelte"] footer {
  border-top: 1px solid var(--nx-border) !important;
  background: rgba(0, 0, 0, 0.15) !important;
  color: var(--nx-muted) !important;
}
.gradio-container footer a { color: var(--nx-mint) !important; }
.gradio-container footer a:hover { color: #ffffff !important; }

/* ===== RESPONSIVE ===== */
@media (max-width: 768px){
  .nx-landing{ padding: 20px 16px 32px; }
  .nx-landing #nx_hero h1, .nx-hero-title{ font-size: 1.05rem !important; }
  .ny-case, .ny-reason{ height: auto; max-height: none; min-height: auto; padding: 16px; overflow: visible; }
  #nx_controls button{ padding: 9px 14px !important; }
}

/* ===== SCROLLBAR ===== */
.ny-fir::-webkit-scrollbar{ width: 6px; }
.ny-fir::-webkit-scrollbar-track{ background: transparent; }
.ny-fir::-webkit-scrollbar-thumb{
  background: var(--nx-bronze-2);
  border-radius: 999px;
}
.ny-fir::-webkit-scrollbar-thumb:hover{ background: var(--nx-bronze); }
"""
