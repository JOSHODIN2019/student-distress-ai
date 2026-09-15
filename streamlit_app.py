import streamlit as st
import streamlit.components.v1 as components
import joblib, os, json, pandas as pd

st.set_page_config(
    page_title="Student Distress Analyzer",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_CONFIG = [
    {"id": "lr", "name": "Logistic Reg", "short": "LR", "file": "logistic_regression_model.pkl"},
    {"id": "rf", "name": "Random Forest",           "short": "RF", "file": "random_forest_model.pkl"},
]
for m in MODEL_CONFIG:
    m["available"] = os.path.exists(os.path.join(BASE, m["file"]))

try:
    with open(os.path.join(BASE, "model_metrics.json")) as f:
        METRICS = json.load(f)
except Exception:
    METRICS = {}

CATEGORICAL = ["Gender", "Sleep Duration", "Dietary Habits"]
NUMERICAL   = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction",
               "Work/Study Hours", "Financial Stress"]
FEATURES    = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
               "Sleep Duration", "Dietary Habits", "Work/Study Hours", "Financial Stress"]

@st.cache_resource(show_spinner="Initialising models…")
def load_resources():
    models = {}
    for m in MODEL_CONFIG:
        if m["available"]:
            models[m["id"]] = joblib.load(os.path.join(BASE, m["file"]))
    return (
        models,
        joblib.load(os.path.join(BASE, "scaler.pkl")),
        joblib.load(os.path.join(BASE, "label_encoders.pkl")),
    )

loaded_models, scaler, encoders = load_resources()

# ── Session state ──────────────────────────────────────────────────────────────
available_ids = [m["id"] for m in MODEL_CONFIG if m["available"]]
def _init(k, v):
    if k not in st.session_state: st.session_state[k] = v

_init("mdl",    available_ids[0] if available_ids else "lr")
_init("result", None)
_init("ap", 3); _init("ss", 3); _init("wsh", 6); _init("fs", 3)

mdl        = st.session_state.mdl
active_cfg = next((m for m in MODEL_CONFIG if m["id"] == mdl), MODEL_CONFIG[0])
met        = METRICS.get(mdl, {})

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
/* ─ Chrome removal ──────────────────────────────────────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"],
section[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display:none !important; }

/* ─ Reset ───────────────────────────────────────────────────────────────── */
html, body { margin:0; padding:0; overflow:hidden; height:100vh; }
* { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif !important;
    box-sizing:border-box; }
.block-container { padding:0 !important; max-width:100% !important; }
[data-testid="stMain"] { padding:0 !important; }
.main, section.main { padding-top:0 !important; }

/* ─ 2-stripe: sidebar white, rest soft gray ─────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right,
        #ffffff 0%, #ffffff 20%,
        #edf0f5 20%, #edf0f5 100%
    ) !important;
    min-height:100vh !important;
}

/* ─ Lock page to viewport — no page scroll ──────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    height:100vh !important;
    overflow:clip !important;
}
[data-testid="stBottom"] { display:none !important; }

/* ─ Remove gap between header row and main layout row ──────────────────── */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    gap:0 !important;
}

/* ─ Outer column row ───────────────────────────────────────────────────── */
[data-role="main-layout"] {
    align-items:stretch !important;
    gap:0 !important;
}

/* ─ Each column: JS sets exact heights; columns scroll internally if needed */
[data-role="sidebar"],
[data-role="form"],
[data-role="result"] {
    align-self:stretch !important;
    overflow-y:auto !important;
    overflow-x:hidden !important;
    scrollbar-width:thin;
}

/* ─ Sidebar sits above header row content ───────────────────────────────── */
[data-role="sidebar"] {
    position:relative !important;
    z-index:2 !important;
}

/* ─ Sidebar ─────────────────────────────────────────────────────────────── */
[data-role="sidebar"] {
    background:#ffffff !important;
    border-right:1px solid #e8ecf0 !important;
    padding-left:12px !important;
    padding-right:12px !important;
}
[data-role="sidebar"] p,
[data-role="sidebar"] span,
[data-role="sidebar"] label { color:#374151 !important; }
[data-role="sidebar"] [data-testid="stWidgetLabel"] p {
    color:#94a3b8 !important; font-size:10px !important;
    font-weight:700 !important; letter-spacing:.1em !important;
    text-transform:uppercase !important;
}
[data-role="sidebar"] .stButton button {
    background:#f8fafc !important;
    border:1px solid #e8ecf0 !important;
    color:#64748b !important; border-radius:8px !important;
    font-size:12.5px !important; font-weight:500 !important;
    padding:9px 12px !important; transition:all .15s !important;
    text-align:left !important; width:100% !important;
}
[data-role="sidebar"] .stButton button:hover:not(:disabled) {
    background:#f1f5f9 !important;
    color:#1e293b !important; border-color:#cbd5e1 !important;
}
[data-role="sidebar"] .stButton button[data-testid="baseButton-primary"] {
    background:rgba(16,163,127,.08) !important;
    border:1px solid rgba(16,163,127,.3) !important;
    color:#10a37f !important; font-weight:700 !important;
}
[data-role="sidebar"] .stButton button:disabled {
    background:#f8fafc !important;
    border-color:#f1f5f9 !important;
    color:#cbd5e1 !important; cursor:not-allowed !important; opacity:.6 !important;
}

/* ─ Model toggle buttons in sidebar ─────────────────────────────────────── */
[data-role="sidebar"] [data-testid="stHorizontalBlock"] {
    gap:4px !important;
    background:#f1f5f9 !important;
    border:1px solid #e2e8f0 !important;
    border-radius:10px !important;
    padding:3px !important;
}
[data-role="sidebar"] [data-testid="stHorizontalBlock"] .stButton button {
    text-align:center !important;
    justify-content:center !important;
    padding:6px 4px !important;
    font-size:12px !important;
    border-radius:8px !important;
    display:flex !important;
    align-items:center !important;
    width:100% !important;
    border:none !important;
}
/* Force child p/span to inherit the button's JS-applied color */
[data-role="sidebar"] [data-testid="stHorizontalBlock"] .stButton button p,
[data-role="sidebar"] [data-testid="stHorizontalBlock"] .stButton button span,
[data-role="sidebar"] [data-testid="stHorizontalBlock"] .stButton button div {
    color:inherit !important;
    transition:color .25s ease !important;
}

/* ─ Form column: soft gray background, card floats inside ───────────────── */
[data-role="form"] {
    background:#edf0f5 !important;
    border-right:none !important;
    padding:40px 14px 20px !important;
    align-items:flex-start !important;
}
/* Header row column — sits above the card in the gray zone */
[data-role="form-header"] {
    background:#edf0f5 !important;
    padding:0 14px 0 !important;
}
[data-role="form-header-result"] {
    background:#edf0f5 !important;
    padding:0 !important;
}
/* The card */
[data-role="form"] > div:first-child {
    background:#ffffff !important;
    border-radius:16px !important;
    border:1.5px solid #e2e8f0 !important;
    box-shadow:0 4px 24px rgba(15,23,42,.09), 0 1px 4px rgba(15,23,42,.04) !important;
    overflow:clip !important;
    padding:0 !important;
    width:100% !important;
}

/* Remove ALL default Streamlit gaps inside the form */
[data-role="form"] [data-testid="stVerticalBlock"] { gap:0 !important; }

/* Each row of field columns */
[data-role="form"] [data-testid="stHorizontalBlock"] {
    gap:12px !important;
    padding:0 20px !important;
}
/* Column inner div: top padding only (bottom spacing via inter-row spacers) */
[data-role="form"] [data-testid="stColumn"] > div { padding:10px 0 0 !important; }

/* ─ Form field label normalization ──────────────────────────────────────── */
[data-role="form"] [data-testid="stHorizontalBlock"] [data-testid="stWidgetLabel"] {
    margin:0 0 4px !important; padding:0 !important;
}

/* ─ Result column: same card treatment as form ──────────────────────────── */
[data-role="result"] {
    background:#edf0f5 !important;
    padding:40px 14px 20px 0 !important;
    align-items:flex-start !important;
}
[data-role="result"] > div:first-child {
    background:#ffffff !important;
    border-radius:16px !important;
    border:1.5px solid #e2e8f0 !important;
    box-shadow:0 4px 24px rgba(15,23,42,.09), 0 1px 4px rgba(15,23,42,.04) !important;
    overflow:hidden !important;
    padding:0 !important;
    width:100% !important;
    max-height:300px !important;
    transition:max-height .9s cubic-bezier(.4,0,.2,1) !important;
}
/* Expand when prediction result is present — capped to viewport */
[data-role="result"] > div:first-child:has(.result-card) {
    max-height:calc(100vh - 100px) !important;
    overflow-y:auto !important;
}
[data-role="result"] [data-testid="stVerticalBlock"] { gap:0 !important; }

/* ─ Global widget label ─────────────────────────────────────────────────── */
[data-testid="stWidgetLabel"] p {
    font-size:12.5px !important; font-weight:600 !important;
    color:#374151 !important; letter-spacing:0 !important;
    text-transform:none !important; margin-bottom:4px !important;
}
[data-role="sidebar"] [data-testid="stWidgetLabel"] p {
    font-size:10px !important; color:#475569 !important;
    text-transform:uppercase !important; letter-spacing:.1em !important;
}

/* ─ Select box ──────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div {
    border:1.5px solid #e2e8f0 !important; border-radius:10px !important;
    background:#fff !important; font-size:13px !important;
    color:#1e293b !important; min-height:42px !important;
    transition:border-color .15s, box-shadow .15s !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color:#10a37f !important;
    box-shadow:0 0 0 3px rgba(16,163,127,.1) !important;
}

/* ─ Number input — border lives on the container, buttons stay in flow ──── */
/* Container becomes the visible "box" */
[data-testid="stNumberInputContainer"] {
    display:flex !important; align-items:stretch !important;
    border:1.5px solid #e2e8f0 !important; border-radius:10px !important;
    background:#fff !important; overflow:hidden !important;
    transition:border-color .15s, box-shadow .15s !important;
    min-height:42px !important;
}
[data-testid="stNumberInputContainer"]:focus-within {
    border-color:#10a37f !important;
    box-shadow:0 0 0 3px rgba(16,163,127,.1) !important;
}
/* Input: no border of its own — inherits the container's visual box */
[data-testid="stNumberInput"] input {
    flex:1 !important; min-width:0 !important;
    border:none !important; outline:none !important; box-shadow:none !important;
    background:transparent !important; font-size:13px !important;
    color:#1e293b !important; padding:0 10px !important;
}
[data-testid="stNumberInput"] input:focus {
    border:none !important; outline:none !important; box-shadow:none !important;
}
/* Button wrapper div sits to the right inside the container */
[data-testid="stNumberInputContainer"] > div {
    display:flex !important; align-items:stretch !important;
    border-left:1.5px solid #e2e8f0 !important; flex-shrink:0 !important;
}
/* Each ± button */
[data-testid="stNumberInputContainer"] button {
    width:34px !important; border:none !important; border-radius:0 !important;
    background:transparent !important; color:#64748b !important;
    display:flex !important; align-items:center !important;
    justify-content:center !important; cursor:pointer !important;
    transition:background .12s !important;
    border-left:1px solid #e2e8f0 !important;
}
[data-testid="stNumberInputContainer"] button:first-child {
    border-left:none !important;
}
[data-testid="stNumberInputContainer"] button:hover {
    background:#f1f5f9 !important;
}


/* ─ Primary button (Analyze) ────────────────────────────────────────────── */
.stButton button[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#10a37f,#0d8f6e) !important;
    border:none !important; color:#fff !important; font-weight:700 !important;
    border-radius:10px !important; box-shadow:0 4px 14px rgba(16,163,127,.3) !important;
    font-size:14px !important; letter-spacing:.01em !important;
    transition:all .18s !important;
}
.stButton button[data-testid="baseButton-primary"]:hover {
    box-shadow:0 6px 20px rgba(16,163,127,.45) !important;
    transform:translateY(-1px) !important;
}
.stButton button[data-testid="baseButton-primary"]:active {
    transform:translateY(0) !important;
}

/* ─ Secondary button (Reset) ────────────────────────────────────────────── */
.stButton button[data-testid="baseButton-secondary"] {
    background:#fff !important; border:1.5px solid #e2e8f0 !important;
    color:#64748b !important; font-weight:600 !important;
    border-radius:10px !important; font-size:13px !important;
    transition:all .15s !important;
}
.stButton button[data-testid="baseButton-secondary"]:hover {
    border-color:#cbd5e1 !important; color:#374151 !important;
    background:#f8fafc !important;
}

/* ─ Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:#d1d5db; border-radius:4px; }
[data-role="sidebar"]::-webkit-scrollbar-thumb { background:#1e293b; }

/* ─ Result panel animations ─────────────────────────────────────────────── */
@keyframes resultCardIn {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes statNumIn {
  from { opacity:0; transform:scale(.85); }
  to   { opacity:1; transform:scale(1); }
}
.result-card {
  animation: resultCardIn .4s cubic-bezier(.4,0,.2,1) both;
}
.result-card:nth-child(2) { animation-delay:.08s; }
.result-card:nth-child(3) { animation-delay:.16s; }

/* ─ Analyze button loading state ────────────────────────────────────────── */
@keyframes pulse {
  0%,100% { opacity:1; }
  50%      { opacity:.6; }
}

/* ════════════════════════════════════════════════════════════════════════════
   MOBILE  (≤ 768 px)
   ════════════════════════════════════════════════════════════════════════════ */
@media (max-width:768px) {

/* Re-enable scroll */
html, body { overflow:auto !important; height:auto !important; }
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { height:auto !important; overflow:auto !important; }

/* Solid gray bg — no white sidebar stripe */
[data-testid="stAppViewContainer"] { background:#edf0f5 !important; }

/* Header row: hide blank flanking columns, title fills width */
[data-testid="stHorizontalBlock"]:has([data-role="form-header"])
  > [data-testid="stColumn"]:first-child,
[data-role="form-header-result"] { display:none !important; }
[data-role="form-header"] {
    flex:1 1 100% !important; max-width:100% !important;
    padding:16px 16px 10px !important;
}

/* Main layout: stack columns vertically */
[data-role="main-layout"] {
    flex-direction:column !important;
    height:auto !important; min-height:auto !important;
}

/* Hide sidebar on mobile */
[data-role="sidebar"] { display:none !important; }

/* Form and result: full width, natural height */
[data-role="form"],
[data-role="result"] {
    flex:1 1 100% !important;
    width:100% !important; max-width:100% !important;
    height:auto !important; min-height:auto !important; max-height:none !important;
    margin-top:0 !important;
}
[data-role="form"]   { padding:12px 12px 16px !important; }
[data-role="result"] { padding:0 12px 16px !important; }

/* Result card: lift max-height cap on mobile */
[data-role="result"] > div:first-child,
[data-role="result"] > div:first-child:has(.result-card) {
    max-height:none !important; overflow-y:visible !important;
}

/* Form field rows: tighter horizontal padding */
[data-role="form"] [data-testid="stHorizontalBlock"] {
    padding:0 12px !important; gap:8px !important;
}
}

/* Very small screens (≤ 480 px): stack 2-col field pairs vertically */
@media (max-width:480px) {
[data-role="form"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
[data-role="form"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex:1 1 100% !important; min-width:0 !important; width:100% !important;
}
}
</style>""", unsafe_allow_html=True)

# ── JS: tag 3-column layout for CSS targeting ─────────────────────────────────
components.html("""<script>
function tagLayout() {
    try {
        var doc = parent.document;
        var hbs = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
        var blocks = [];
        for (var i = 0; i < hbs.length; i++) {
            var cols = hbs[i].querySelectorAll(':scope > [data-testid="stColumn"]');
            if (cols.length === 3) blocks.push({hb: hbs[i], cols: cols});
        }
        if (blocks.length === 0) return false;
        // Last 3-col block is the main layout
        var main = blocks[blocks.length - 1];
        function sa(el, val) {
            if (el.getAttribute('data-role') !== val) el.setAttribute('data-role', val);
        }
        sa(main.hb, 'main-layout');
        sa(main.cols[0], 'sidebar');
        sa(main.cols[1], 'form');
        sa(main.cols[2], 'result');
        if (blocks.length >= 2) {
            var hdr = blocks[blocks.length - 2];
            sa(hdr.cols[1], 'form-header');
            sa(hdr.cols[2], 'form-header-result');
        }
        return true;
    } catch(e) {}
    return false;
}

function cleanSliders() {
    try {
        var doc = parent.document;
        var form = doc.querySelector('[data-role="form"]');
        if (!form) return;
        // Hide native value bubble and tick marks on all form sliders
        form.querySelectorAll('[data-testid="stSlider"]').forEach(function(s) {
            s.querySelectorAll('p, [data-testid="stTickBarMin"], [data-testid="stTickBarMax"]')
             .forEach(function(el){ el.style.cssText='display:none!important'; });
            // Hide the floating value div above thumb (Streamlit renders it as a sibling div)
            var track = s.querySelector('[role="slider"]');
            if (track) {
                var parent_div = track.parentElement;
                if (parent_div) {
                    parent_div.querySelectorAll('div:not([role="slider"])').forEach(function(d){
                        if(d.textContent.trim().length < 5 && !isNaN(d.textContent.trim())) {
                            d.style.cssText='display:none!important';
                        }
                    });
                }
            }
        });
    } catch(e) {}
}

function isMobile() {
    try { return parent.innerWidth <= 768; } catch(e) { return false; }
}

function alignSidebar() {
    try {
        var doc = parent.document;
        var sidebar = doc.querySelector('[data-role="sidebar"]');
        if (!sidebar) return;
        if (isMobile()) { sidebar.style.marginTop = '0'; return; }
        var headerCol = doc.querySelector('[data-role="form-header"]');
        if (!headerCol) return;
        var row = headerCol.closest('[data-testid="stHorizontalBlock"]');
        if (!row) return;
        sidebar.style.marginTop = '-' + row.offsetHeight + 'px';
    } catch(e) {}
}

function setLayoutHeight() {
    try {
        var doc = parent.document;
        var el = doc.getElementById('sda-layout-h');
        if (isMobile()) {
            if (el) el.textContent = '';
            return;
        }
        var hrowEl = doc.querySelector('[data-role="form-header"]');
        if (!hrowEl) return;
        var hrow = hrowEl.closest('[data-testid="stHorizontalBlock"]');
        if (!hrow) return;
        var headerH = hrow.offsetHeight;
        var vh = Math.min(parent.innerHeight, doc.documentElement.clientHeight);
        var mainH = Math.max(vh - headerH - 4, 380);
        if (!el) { el = doc.createElement('style'); el.id = 'sda-layout-h'; doc.head.appendChild(el); }
        el.textContent =
            '[data-role="main-layout"]{height:'+mainH+'px!important;min-height:'+mainH+'px!important;}' +
            '[data-role="sidebar"],[data-role="form"],[data-role="result"]{height:'+mainH+'px!important;min-height:'+mainH+'px!important;max-height:'+mainH+'px!important;}';
    } catch(e) {}
}

function applyBtnStyle(btn, isActive) {
    var color = isActive ? '#fff' : '#64748b';
    var fw    = isActive ? '700' : '500';
    var bg    = isActive ? 'linear-gradient(135deg,#10a37f,#0d8f6e)' : 'transparent';
    var sh    = isActive ? '0 2px 6px rgba(16,163,127,.3)' : 'none';
    // Incoming button fades in smoothly; outgoing snaps off instantly (avoids double-green overlap)
    var tr    = isActive ? 'background .2s ease, color .2s ease, box-shadow .2s ease' : 'none';
    btn.style.setProperty('transition', tr, 'important');
    btn.style.setProperty('border', 'none', 'important');
    btn.style.setProperty('background', bg, 'important');
    btn.style.setProperty('color', color, 'important');
    btn.style.setProperty('font-weight', fw, 'important');
    btn.style.setProperty('box-shadow', sh, 'important');
    btn.querySelectorAll('p,span,div').forEach(function(el) {
        el.style.setProperty('color', color, 'important');
        el.style.setProperty('transition', isActive ? 'color .2s ease' : 'none', 'important');
    });
}

function styleModelBtns() {
    try {
        var doc = parent.document;
        var marker = doc.getElementById('sda-active-mdl');
        if (!marker) return;
        var activeMdl = marker.textContent.trim();
        var sidebar = doc.querySelector('[data-role="sidebar"]');
        if (!sidebar) return;
        var hb = sidebar.querySelector('[data-testid="stHorizontalBlock"]');
        if (!hb) return;
        var btns = hb.querySelectorAll('button');
        btns.forEach(function(btn) {
            var text = btn.textContent.trim().toLowerCase();
            var isActive = (activeMdl === 'lr' && text.indexOf('logistic') !== -1) ||
                           (activeMdl === 'rf' && text.indexOf('forest') !== -1);
            // Only mutate DOM when state differs — avoids constant repaints
            var alreadyActive = btn.style.getPropertyValue('background').indexOf('10a37f') !== -1;
            if (isActive !== alreadyActive) applyBtnStyle(btn, isActive);
            // Wire click listener once
            if (!btn._sdaWired) {
                btn._sdaWired = true;
                (function(b) {
                    b.addEventListener('click', function() {
                        // Apply to current elements immediately for instant visual response
                        var clickedLR = b.textContent.trim().toLowerCase().indexOf('logistic') !== -1;
                        var container = b.closest('[data-testid="stHorizontalBlock"]');
                        if (container) {
                            container.querySelectorAll('button').forEach(function(x) {
                                applyBtnStyle(x, clickedLR
                                    ? x.textContent.trim().toLowerCase().indexOf('logistic') !== -1
                                    : x.textContent.trim().toLowerCase().indexOf('forest') !== -1);
                            });
                        }
                        // MutationObserver handles re-styling after Streamlit rerender
                    });
                })(btn);
            }
        });
    } catch(e) {}
}

(function retry(){ if(!tagLayout()) setTimeout(retry,80); else { alignSidebar(); setLayoutHeight(); styleModelBtns(); } })();
setInterval(function(){ tagLayout(); cleanSliders(); alignSidebar(); setLayoutHeight(); styleModelBtns(); }, 200);
</script>""", height=1)

# ══════════════════════════════════════════════════════════════════════════════
# FORM HEADER ROW (above the card, in the gray zone)
# ══════════════════════════════════════════════════════════════════════════════
_, _h_form, _ = st.columns([1.7, 5, 2.1], gap="small")
with _h_form:
    st.markdown("""
<div style="padding:0 0 10px;">
  <h2 style="font-size:20px;font-weight:800;color:#0f172a;letter-spacing:-.4px;
    line-height:1.2;margin:0;">Student Distress Assessment</h2>
  <p style="font-size:14px;color:#94a3b8;margin:0;line-height:1.5;">
    Enter the student's details below — the model will instantly predict their distress risk level.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_sb, col_form, col_result = st.columns([1.7, 5, 2.1], gap="small")

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with col_sb:
    st.markdown("""
<div style="padding:24px 0 12px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <div style="width:36px;height:36px;background:linear-gradient(135deg,#10a37f,#059669);
      border-radius:10px;display:flex;align-items:center;justify-content:center;
      flex-shrink:0;box-shadow:0 4px 12px rgba(16,163,127,.25);">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.44-3.16Z"/>
        <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.44-3.16Z"/>
      </svg>
    </div>
    <div>
      <div style="font-size:9.5px;font-weight:800;letter-spacing:.18em;color:#94a3b8;
        text-transform:uppercase;margin-bottom:1px;">Student</div>
      <div style="font-size:15.5px;font-weight:800;color:#0f172a;letter-spacing:-.2px;
        line-height:1.15;white-space:nowrap;">Distress Analyzer</div>
    </div>
  </div>
</div>
<div style="height:1px;background:#e8ecf0;margin:0 0 16px;"></div>
<div style="padding:0 0 8px;">
  <div style="font-size:9px;font-weight:800;letter-spacing:.14em;color:#94a3b8;
    text-transform:uppercase;margin-bottom:10px;">Prediction Model</div>
</div>
""", unsafe_allow_html=True)

    def _switch_model(new_mdl):
        st.session_state.mdl = new_mdl
        st.session_state.result = None

    bt1, bt2 = st.columns(2, gap="small")
    with bt1:
        if st.button("Logistic Reg", key="btn_lr", use_container_width=True,
                     type="primary" if mdl == "lr" else "secondary",
                     disabled=not any(m["id"]=="lr" and m["available"] for m in MODEL_CONFIG)):
            _switch_model("lr")
            st.rerun()
    with bt2:
        if st.button("Random Forest", key="btn_rf", use_container_width=True,
                     type="primary" if mdl == "rf" else "secondary",
                     disabled=not any(m["id"]=="rf" and m["available"] for m in MODEL_CONFIG)):
            _switch_model("rf")
            st.rerun()
    # Hidden marker so JS knows which model is active
    st.markdown(f'<div id="sda-active-mdl" style="display:none">{mdl}</div>', unsafe_allow_html=True)

    acc_v  = f"{met.get('accuracy','–')}%"  if met else "–"
    f1_v   = f"{met.get('f1','–')}%"        if met else "–"
    prec_v = f"{met.get('precision','–')}%" if met else "–"
    rec_v  = f"{met.get('recall','–')}%"    if met else "–"

    st.markdown(f"""
<div style="padding-top:16px;border-top:1px solid #e8ecf0;margin-top:14px;">
  <div style="font-size:9px;font-weight:800;letter-spacing:.14em;color:#94a3b8;
    text-transform:uppercase;margin-bottom:10px;">Model Performance</div>
  <div style="display:flex;align-items:center;gap:7px;margin-bottom:14px;">
    <div style="width:7px;height:7px;border-radius:50%;background:#10a37f;
      box-shadow:0 0 0 3px rgba(16,163,127,.2);flex-shrink:0;"></div>
    <span style="font-size:12px;font-weight:600;color:#64748b;">Active:</span>
    <span style="font-size:12px;font-weight:700;color:#10a37f;">{active_cfg['name']}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:9px;color:#94a3b8;margin-bottom:3px;letter-spacing:.04em;">Accuracy</div>
      <div style="font-size:14px;font-weight:800;color:#0f172a;font-variant-numeric:tabular-nums;">{acc_v}</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:9px;color:#94a3b8;margin-bottom:3px;letter-spacing:.04em;">F1 Score</div>
      <div style="font-size:14px;font-weight:800;color:#0f172a;font-variant-numeric:tabular-nums;">{f1_v}</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:9px;color:#94a3b8;margin-bottom:3px;letter-spacing:.04em;">Precision</div>
      <div style="font-size:14px;font-weight:800;color:#0f172a;font-variant-numeric:tabular-nums;">{prec_v}</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:9px;color:#94a3b8;margin-bottom:3px;letter-spacing:.04em;">Recall</div>
      <div style="font-size:14px;font-weight:800;color:#0f172a;font-variant-numeric:tabular-nums;">{rec_v}</div>
    </div>
  </div>
</div>
<div style="height:1px;background:#e8ecf0;margin:16px 0;"></div>
<div style="margin:0 0 24px;">
  <div style="font-size:9px;font-weight:800;letter-spacing:.14em;color:#94a3b8;
    text-transform:uppercase;margin-bottom:10px;">Training Data</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:13px;font-weight:900;color:#0f172a;font-variant-numeric:tabular-nums;">27,895</div>
      <div style="font-size:9px;color:#94a3b8;margin-top:2px;">Records</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:13px;font-weight:900;color:#0f172a;">9</div>
      <div style="font-size:9px;color:#94a3b8;margin-top:2px;">Features</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:13px;font-weight:900;color:#10a37f;">80%</div>
      <div style="font-size:9px;color:#94a3b8;margin-top:2px;">Train split</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e8ecf0;
      border-radius:8px;padding:8px 10px;">
      <div style="font-size:13px;font-weight:900;color:#10a37f;">20%</div>
      <div style="font-size:9px;color:#94a3b8;margin-top:2px;">Test split</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# FORM COLUMN
# ──────────────────────────────────────────────────────────────────────────────
with col_form:

    def section_label(title, icon="", first=False):
        top = "14px" if first else "20px"
        st.markdown(
            f'<div style="padding:{top} 24px 10px;background:#ffffff;">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:.12em;'
            f'color:#94a3b8;text-transform:uppercase;">{icon}{title}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    def field_pad(top=10, bottom=10):
        """Vertical padding wrapper around a field row."""
        st.markdown(
            f'<div style="padding:{top}px 0 {bottom}px;"></div>',
            unsafe_allow_html=True
        )

    def field_label(text):
        """Plain label with the same fixed height as slider_label."""
        st.markdown(
            f'<div style="display:flex;align-items:center;height:28px;">'
            f'<span style="font-size:12.5px;font-weight:600;color:#374151;">{text}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    def slider_label(text, val, min_v=1, max_v=5):
        """Custom label row showing slider name + current value badge."""
        pct = (val - min_v) / (max_v - min_v)
        if pct < 0.4:
            badge_bg, badge_color = "#f0fdf4", "#16a34a"
        elif pct < 0.7:
            badge_bg, badge_color = "#fffbeb", "#d97706"
        else:
            badge_bg, badge_color = "#fef2f2", "#dc2626"
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;height:28px;">'
            f'<span style="font-size:12.5px;font-weight:600;color:#374151;">{text}</span>'
            f'<span style="background:{badge_bg};color:{badge_color};font-size:11px;'
            f'font-weight:700;border-radius:6px;padding:2px 8px;">{val} / {max_v}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    def slider_label_wide(text, val, min_v=0, max_v=16, unit="hrs"):
        pct = (val - min_v) / (max_v - min_v)
        if pct < 0.4:
            badge_bg, badge_color = "#f0fdf4", "#16a34a"
        elif pct < 0.7:
            badge_bg, badge_color = "#fffbeb", "#d97706"
        else:
            badge_bg, badge_color = "#fef2f2", "#dc2626"
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;height:28px;">'
            f'<span style="font-size:12.5px;font-weight:600;color:#374151;">{text}</span>'
            f'<span style="background:{badge_bg};color:{badge_color};font-size:11px;'
            f'font-weight:700;border-radius:6px;padding:2px 8px;">{val} {unit}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── DEMOGRAPHICS ──────────────────────────────────────────────────────────
    section_label("Demographics", first=True)
    d1, d2 = st.columns(2, gap="small")
    with d1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    with d2:
        age = st.number_input("Age", 15, 45, 21, key="age")

    # ── ACADEMIC PERFORMANCE ──────────────────────────────────────────────────
    section_label("Academic Performance")
    a1, a2 = st.columns(2, gap="small")
    with a1:
        academic_pressure = st.number_input("Academic Pressure (1–5)", min_value=1, max_value=5, step=1, key="ap")
    with a2:
        cgpa = st.number_input("CGPA (0.0 – 5.0)", 0.0, 5.0, 3.50, 0.01, format="%.2f", key="cgpa")

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    a3, a4 = st.columns(2, gap="small")
    with a3:
        study_satisfaction = st.number_input("Study Satisfaction (1–5)", min_value=1, max_value=5, step=1, key="ss")
    with a4:
        st.empty()

    # ── LIFESTYLE ─────────────────────────────────────────────────────────────
    section_label("Lifestyle")
    l1, l2 = st.columns(2, gap="small")
    with l1:
        dietary_habits = st.selectbox("Dietary Habits",
            ["Healthy", "Moderate", "Unhealthy", "Others"], key="diet")
    with l2:
        work_study_hours = st.number_input("Work / Study Hours (0–16)", min_value=0, max_value=16, step=1, key="wsh")

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    l3, l4 = st.columns(2, gap="small")
    with l3:
        financial_stress = st.number_input("Financial Stress (1–5)", min_value=1, max_value=5, step=1, key="fs")
    with l4:
        sleep_duration = st.selectbox("Sleep Duration", [
            "Less than 5 hours", "5-6 hours", "7-8 hours",
            "More than 8 hours", "Others"
        ], key="sleep")

    # ── ACTION BAR ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="height:1px;background:#e2e8f0;margin:36px -24px 0;"></div>
<div style="height:28px;"></div>
""", unsafe_allow_html=True)

    ab1, ab2 = st.columns([1, 3], gap="small")
    with ab1:
        reset_clicked = st.button("↺ Reset", key="reset_btn", use_container_width=True)
    with ab2:
        analyze = st.button(
            f"Analyze with {active_cfg['name']}  →",
            key="analyze_btn", use_container_width=True, type="primary"
        )
    st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RESULT COLUMN
# ──────────────────────────────────────────────────────────────────────────────
with col_result:

    r = st.session_state.result

    # Header — always visible (compact state)
    st.markdown("""
<div style="padding:20px 20px 16px;background:#fff;border-radius:16px 16px 0 0;
  border-bottom:1px solid #f1f5f9;">
  <h3 style="font-size:18px;font-weight:800;color:#0f172a;letter-spacing:-.3px;
    line-height:1;margin:0;">Prediction Result</h3>
  <p style="font-size:14px;color:#9ca3af;margin:0;line-height:1;">AI-powered distress risk assessment</p>
</div>
""", unsafe_allow_html=True)

    if r is None:
        # Empty state — compact card with icon + hint
        st.markdown("""
<div style="padding:24px 20px 28px;text-align:center;">
  <div style="width:50px;height:50px;background:#f1f5f9;border-radius:50%;
    margin:0 auto 14px;display:flex;align-items:center;justify-content:center;">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 7V5a2 2 0 0 1 2-2h2"/>
      <path d="M17 3h2a2 2 0 0 1 2 2v2"/>
      <path d="M21 17v2a2 2 0 0 1-2 2h-2"/>
      <path d="M7 21H5a2 2 0 0 1-2-2v-2"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  </div>
  <div style="font-size:14.5px;font-weight:700;color:#1e293b;margin-bottom:5px;
    letter-spacing:-.15px;">No prediction yet</div>
  <div style="font-size:12.5px;color:#94a3b8;line-height:1.6;">
    Fill in the student profile and click<br>
    <span style="color:#10a37f;font-weight:600;">Analyze</span> to get a distress result
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        # ── Result display ────────────────────────────────────────────────────
        prob_d   = r["prob_d"]
        prob_nd  = r["prob_nd"]
        mdl_name = r["model"]

        if prob_d >= 70:
            risk_level = "HIGH RISK"
            risk_bg, risk_text = "#fff5f5", "#dc2626"
            ring_col = "#ef4444"
            badge_bg, badge_border = "#fef2f2", "#fecaca"
        elif prob_d >= 45:
            risk_level = "MEDIUM RISK"
            risk_bg, risk_text = "#fffbeb", "#b45309"
            ring_col = "#f59e0b"
            badge_bg, badge_border = "#fefce8", "#fde68a"
        else:
            risk_level = "LOW RISK"
            risk_bg, risk_text = "#f0fdf4", "#15803d"
            ring_col = "#10a37f"
            badge_bg, badge_border = "#f0fdf4", "#bbf7d0"

        offset = round(251.2 * (1 - prob_d / 100), 1)

        if prob_d >= 70:
            advice = ("Significant distress indicators detected. "
                      "Professional counselling, workload review, and campus support resources "
                      "are strongly recommended.")
        elif prob_d >= 45:
            advice = ("Moderate distress indicators present. "
                      "An early check-in with an academic advisor and monitoring of "
                      "key stressors is recommended.")
        else:
            advice = ("Low distress indicators detected. "
                      "Maintaining current healthy habits is advised. "
                      "Continue to monitor each semester proactively.")

        st.markdown(f"""
<div style="padding:16px 16px 0;">

  <!-- Risk ring card -->
  <div class="result-card" style="background:{risk_bg};border:1.5px solid {badge_border};
    border-radius:14px;padding:24px 20px 18px;text-align:center;margin-bottom:12px;">
    <div style="position:relative;width:128px;height:128px;
      display:inline-flex;align-items:center;justify-content:center;margin-bottom:12px;">
      <svg width="128" height="128" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(0,0,0,.07)" stroke-width="9"/>
        <circle cx="50" cy="50" r="40" fill="none" stroke="{ring_col}" stroke-width="9"
          stroke-dasharray="251.2" stroke-dashoffset="{offset}"
          stroke-linecap="round" transform="rotate(-90 50 50)">
          <animate attributeName="stroke-dashoffset"
            from="251.2" to="{offset}"
            dur="1.1s" calcMode="spline"
            keyTimes="0;1" keySplines=".4,0,.2,1"
            fill="freeze"/>
        </circle>
      </svg>
      <div style="position:absolute;display:flex;flex-direction:column;
        align-items:center;justify-content:center;">
        <div style="font-size:28px;font-weight:900;color:{ring_col};
          font-variant-numeric:tabular-nums;letter-spacing:-1px;line-height:1;
          animation:statNumIn .6s .3s cubic-bezier(.4,0,.2,1) both;">
          {prob_d:.0f}%</div>
        <div style="font-size:8.5px;font-weight:700;color:{ring_col};
          opacity:.7;letter-spacing:.08em;margin-top:1px;">DISTRESS</div>
      </div>
    </div>
    <div style="display:inline-block;background:{badge_bg};border:1.5px solid {badge_border};
      border-radius:20px;padding:4px 16px;margin-bottom:6px;">
      <span style="font-size:11px;font-weight:800;color:{risk_text};letter-spacing:.06em;">
        {risk_level}</span>
    </div>
    <div style="font-size:11px;color:#9ca3af;margin-top:2px;">via {mdl_name}</div>
  </div>

  <!-- Probability bars -->
  <div class="result-card" style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;
    padding:16px 16px 14px;margin-bottom:12px;">
    <div style="font-size:9.5px;font-weight:800;letter-spacing:.12em;color:#94a3b8;
      text-transform:uppercase;margin-bottom:12px;">Probability Breakdown</div>
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
        font-size:12.5px;margin-bottom:5px;">
        <span style="font-weight:600;color:#ef4444;">Distressed</span>
        <span style="font-weight:800;color:#0f172a;font-variant-numeric:tabular-nums;">
          {prob_d:.1f}%</span>
      </div>
      <div style="background:#f1f5f9;border-radius:6px;height:8px;overflow:hidden;">
        <div style="background:linear-gradient(to right,#ef4444,#f87171);
          height:100%;border-radius:6px;
          animation:barGrow .9s .1s cubic-bezier(.4,0,.2,1) both;
          --bar-w:{prob_d:.1f}%;">
        </div>
      </div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;
        font-size:12.5px;margin-bottom:5px;">
        <span style="font-weight:600;color:#10a37f;">Not Distressed</span>
        <span style="font-weight:800;color:#0f172a;font-variant-numeric:tabular-nums;">
          {prob_nd:.1f}%</span>
      </div>
      <div style="background:#f1f5f9;border-radius:6px;height:8px;overflow:hidden;">
        <div style="background:linear-gradient(to right,#10a37f,#34d399);
          height:100%;border-radius:6px;
          animation:barGrow .9s .2s cubic-bezier(.4,0,.2,1) both;
          --bar-w:{prob_nd:.1f}%;">
        </div>
      </div>
    </div>
  </div>

  <!-- Interpretation -->
  <div class="result-card" style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;
    padding:16px 16px 14px;margin-bottom:16px;">
    <div style="font-size:9.5px;font-weight:800;letter-spacing:.12em;color:#94a3b8;
      text-transform:uppercase;margin-bottom:8px;">Recommendation</div>
    <div style="font-size:13px;color:#374151;line-height:1.75;">{advice}</div>
  </div>

</div>
<style>
@keyframes barGrow {{
  from {{ width:0; }}
  to   {{ width:var(--bar-w); }}
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if reset_clicked:
    st.session_state.result = None
    st.rerun()

if analyze:
    model_obj = loaded_models.get(mdl)
    if model_obj is None:
        st.error(f"Model '{active_cfg['name']}' is not loaded. Please select another model.")
    else:
        with st.spinner(f"Running {active_cfg['name']}…"):
            row = {
                "Gender":             gender,
                "Age":                float(age),
                "Academic Pressure":  int(academic_pressure),
                "CGPA":               float(cgpa),
                "Study Satisfaction": int(study_satisfaction),
                "Sleep Duration":     sleep_duration,
                "Dietary Habits":     dietary_habits,
                "Work/Study Hours":   float(work_study_hours),
                "Financial Stress":   int(financial_stress),
            }
            df = pd.DataFrame([row])
            for col in CATEGORICAL:
                df[col] = encoders[col].transform(df[col])
            df = df[FEATURES]
            df[NUMERICAL] = scaler.transform(df[NUMERICAL])

            label  = int(model_obj.predict(df)[0])
            probas = model_obj.predict_proba(df)[0]
            prob_d = round(float(probas[1]) * 100, 2)

            st.session_state.result = {
                "label":      label,
                "label_text": "Distressed" if label == 1 else "Not Distressed",
                "confidence": round(float(probas[label]) * 100, 2),
                "prob_d":     prob_d,
                "prob_nd":    round(float(probas[0]) * 100, 2),
                "model":      active_cfg["name"],
            }
        st.rerun()
