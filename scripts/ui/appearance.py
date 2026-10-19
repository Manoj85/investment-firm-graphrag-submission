"""Presentation-only themes; System follows the browser color preference."""

LIGHT = "--page:#f7f9fb;--panel:#ffffff;--input:#edf2f6;--ink:#142b40;--muted:#526578;--line:#dce5ec;--accent:#0f766e;"
DARK = "--page:#0f111a;--panel:#191e2c;--input:#242c3d;--ink:#e4eaf3;--muted:#afbdd0;--line:#3b4861;--accent:#70d8c8;"


def appearance_css(choice):
    values = DARK if choice == "Dark" else LIGHT
    system = "@media (prefers-color-scheme:dark) {:root {" + DARK + "}}" if choice == "System" else ""
    return "<style>:root {" + values + "}" + system + """
.st-key-corpus_inventory [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
.st-key-corpus_inventory [data-testid="stColumn"] {min-width:145px;flex:1 1 145px;}
.stApp, [data-testid="stHeader"] {background:var(--page); color:var(--ink);}
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4,
[data-testid="stMain"] p, [data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] {color:var(--ink);}
[data-testid="stMain"] [data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"] {color:var(--muted);}
.intro {color:var(--muted);} .eyebrow {color:var(--accent);}
.st-key-corpus_inventory [data-testid="stMetric"], [data-testid="stForm"] {
 background:var(--panel); border-color:var(--line);}
[data-testid="stMetricValue"] {color:var(--ink);}
[data-testid="stMain"] details, [data-testid="stForm"] [data-testid="stExpander"] details {
 background:var(--panel); border-color:var(--line); color:var(--ink);}
[data-testid="stTextArea"] textarea, [data-baseweb="select"] > div,
[data-baseweb="popover"] ul {background:var(--input); color:var(--ink);}
[data-testid="stTextArea"] textarea {caret-color:var(--ink);}
[data-testid="stTabs"] button[role="tab"] {color:var(--muted);}
[data-testid="stTabs"] button[aria-selected="true"] {color:var(--accent);}
[data-testid="stMain"] [data-testid="stBaseButton-secondary"],
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] p {
 background:var(--panel); color:var(--ink); border-color:var(--line);}
[data-testid="stSidebar"] [data-baseweb="select"] {color:var(--ink);}
[data-testid="stMain"] hr {border-color:var(--line);}
[data-testid="stMain"] a {color:var(--accent);}
</style>"""
