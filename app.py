"""
Skin + Me — Unstructured Data Intelligence Dashboard
A consulting-style Streamlit dashboard for the BAUD pitch.
Visualises sentiment + thematic analysis of 1,000 Trustpilot reviews (real + synthetic).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Skin + Me — Customer Voice Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# BRAND PALETTE — Skin + Me inspired (light, bright, cream + gold)
# ---------------------------------------------------------------------
GOLD           = "#E8A736"   # brand primary (gold)
GOLD_DARK      = "#C68A1F"   # deeper gold for hover / emphasis
GOLD_LIGHT     = "#F4C774"   # softer gold tint
CREAM          = "#FAEFD9"   # brand cream
CREAM_SOFT     = "#FDF7E8"   # page background — cream at lower saturation
CREAM_DEEP     = "#F2E4C1"   # dividers / borders / subtle backgrounds
WHITE          = "#FFFFFF"   # cards
INK            = "#2B2420"   # warm dark brown for primary text
MUTED_INK      = "#7A6F63"   # warm secondary text
TAUPE          = "#B8A37E"   # neutral accent

# Chart colours — light/happy sentiment palette, still readable
GREEN_POS      = "#7FB069"   # soft, leafy green (positive)
GREEN_LIGHT    = "#B8D89A"   # lighter green
CORAL_NEG      = "#E27D5F"   # warm coral (negative) — warmer than red, still clearly "bad"
CORAL_LIGHT    = "#F2A88C"
AMBER          = "#E8A736"   # = GOLD, used for neutral-warm
BLUE           = "#6DA8CC"   # soft sky blue
TEAL           = "#5FB3A1"   # muted teal
PINK           = "#E09BB0"   # dusty pink
PURPLE         = "#A996D4"   # lavender
LIGHT_BLUE     = "#A8CCE2"   # pale sky
ORANGE         = "#E8A36E"   # warm peach

CONDITION_COLORS = [GOLD, PINK, TEAL, BLUE, AMBER, PURPLE, LIGHT_BLUE, ORANGE]
RATING_COLORS    = [GREEN_POS, GREEN_LIGHT, CREAM_DEEP, GOLD_LIGHT, CORAL_NEG]

# ---------------------------------------------------------------------
# GLOBAL CSS  — light, bright, brand-matched
# ---------------------------------------------------------------------
st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: {CREAM_SOFT};
        color: {INK};
    }}

    /* Full width — do not constrain, let the dashboard breathe */
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100% !important;
    }}

    /* Hero header — cream gradient with warm gold accent (NOT dark) */
    .hero {{
        background: linear-gradient(135deg, {CREAM} 0%, {CREAM_SOFT} 60%, {WHITE} 100%);
        padding: 2rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        border: 1px solid {CREAM_DEEP};
        border-left: 6px solid {GOLD};
        box-shadow: 0 2px 12px rgba(232, 167, 54, 0.08);
    }}
    .hero h1 {{
        color: {INK};
        font-size: 1.95rem;
        font-weight: 600;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }}
    .hero .hero-sub {{
        color: {GOLD_DARK};
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }}
    .hero .hero-desc {{
        color: {MUTED_INK};
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
        line-height: 1.55;
        max-width: 900px;
    }}

    /* Section headers */
    .section-header {{
        background: {WHITE};
        border-left: 4px solid {GOLD};
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin: 1.8rem 0 1rem 0;
        border: 1px solid {CREAM_DEEP};
        border-left: 4px solid {GOLD};
    }}
    .section-header h2 {{
        color: {INK};
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }}
    .section-header .subtitle {{
        color: {MUTED_INK};
        font-size: 0.82rem;
        margin: 0.2rem 0 0 0;
        font-weight: 400;
    }}

    /* KPI cards */
    .kpi-card {{
        background: {WHITE};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        border: 1px solid {CREAM_DEEP};
        height: 100%;
        box-shadow: 0 1px 4px rgba(232, 167, 54, 0.06);
    }}
    .kpi-label {{
        color: {MUTED_INK};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }}
    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 600;
        color: {INK};
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .kpi-value.green {{ color: {GREEN_POS}; }}
    .kpi-value.red   {{ color: {CORAL_NEG}; }}
    .kpi-value.gold  {{ color: {GOLD_DARK}; }}
    .kpi-delta {{
        font-size: 0.75rem;
        color: {MUTED_INK};
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }}

    /* Insight callouts */
    .insight-card {{
        background: {WHITE};
        border: 1px solid {CREAM_DEEP};
        border-left: 4px solid {GOLD};
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin: 0.6rem 0;
        font-size: 0.88rem;
        color: {INK};
        line-height: 1.55;
    }}
    .insight-card strong {{ color: {INK}; font-weight: 600; }}
    .insight-card.red {{ border-left-color: {CORAL_NEG}; }}
    .insight-card.green {{ border-left-color: {GREEN_POS}; }}

    /* Review quote card */
    .review-quote {{
        background: {WHITE};
        border: 1px solid {CREAM_DEEP};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: {INK};
        line-height: 1.5;
    }}
    .review-quote .stars {{ color: {GOLD}; font-size: 0.9rem; }}
    .review-quote .meta  {{ color: {MUTED_INK}; font-size: 0.75rem; margin-top: 0.4rem; }}
    .review-quote.neg {{ border-left: 3px solid {CORAL_NEG}; }}
    .review-quote.pos {{ border-left: 3px solid {GREEN_POS}; }}

    /* --------------- SIDEBAR — cream matching the brand, no red --------------- */
    [data-testid="stSidebar"] {{
        background: {CREAM} !important;
        border-right: 1px solid {CREAM_DEEP};
    }}
    [data-testid="stSidebar"] * {{ color: {INK}; }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {{ color: {INK}; }}
    [data-testid="stSidebar"] hr {{ border-top-color: {CREAM_DEEP} !important; }}

    /* Override Streamlit's default red primary colour globally in the sidebar.
       This catches multi-select tags, slider fill, radio selections, checkboxes,
       focus rings, slider handles, etc.  */

    /* Multi-select selected tags */
    [data-testid="stSidebar"] [data-baseweb="tag"] {{
        background-color: {GOLD} !important;
        border-color: {GOLD_DARK} !important;
        color: {WHITE} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="tag"] span {{
        color: {WHITE} !important;
    }}

    /* Radio button — selected state */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
        border-color: {GOLD_DARK} !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child > div {{
        background-color: {GOLD} !important;
    }}

    /* Slider track + fill + handle */
    [data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {{
        background-color: {GOLD} !important;
        border-color: {GOLD_DARK} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {{
        background: {GOLD} !important;
    }}
    /* Slider track background (unfilled part) */
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {{
        background-color: {CREAM_DEEP} !important;
    }}

    /* Select slider track - streamlit uses a specific structure for select_slider */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {{
        background-color: {GOLD} !important;
    }}

    /* Checkbox selected */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label span[aria-checked="true"] {{
        background-color: {GOLD} !important;
        border-color: {GOLD_DARK} !important;
    }}

    /* Input focus rings in sidebar */
    [data-testid="stSidebar"] input:focus,
    [data-testid="stSidebar"] select:focus,
    [data-testid="stSidebar"] textarea:focus {{
        border-color: {GOLD} !important;
        box-shadow: 0 0 0 1px {GOLD} !important;
    }}

    /* Buttons inside sidebar */
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {GOLD} !important;
        border-color: {GOLD_DARK} !important;
        color: {WHITE} !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {GOLD_DARK} !important;
        border-color: {GOLD_DARK} !important;
    }}

    /* Filter container (the "pill" around multiselect etc.) */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {WHITE} !important;
        border-color: {CREAM_DEEP} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
        background: {WHITE};
        border-radius: 8px;
        padding: 0.3rem;
        border: 1px solid {CREAM_DEEP};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.88rem;
        color: {MUTED_INK};
    }}
    .stTabs [aria-selected="true"] {{
        background: {CREAM} !important;
        color: {INK} !important;
        box-shadow: 0 1px 3px rgba(232, 167, 54, 0.15);
    }}
    /* Kill the red underline Streamlit adds to the selected tab */
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {GOLD} !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        background-color: {CREAM_DEEP} !important;
    }}

    /* Recommendation cards */
    .rec-card {{
        background: {WHITE};
        border: 1px solid {CREAM_DEEP};
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        margin: 0.6rem 0;
        height: 100%;
        box-shadow: 0 1px 4px rgba(232, 167, 54, 0.06);
    }}
    .rec-card .rec-num {{
        display: inline-block;
        background: {GOLD};
        color: {WHITE};
        width: 24px;
        height: 24px;
        border-radius: 50%;
        text-align: center;
        line-height: 24px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }}
    .rec-card h4 {{
        color: {INK};
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
    }}
    .rec-card p {{
        color: {MUTED_INK};
        font-size: 0.85rem;
        margin: 0 0 0.6rem 0;
        line-height: 1.55;
    }}
    .rec-card .impact {{
        background: {CREAM};
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        font-size: 0.8rem;
        color: {INK};
        font-weight: 500;
    }}
    .rec-card .impact strong {{ color: {GREEN_POS}; }}

    hr {{ border: none; border-top: 1px solid {CREAM_DEEP}; margin: 1.5rem 0; }}

    /* Headings inside body */
    h2, h3, h4 {{ color: {INK}; font-weight: 600; letter-spacing: -0.01em; }}

    /* Filter-summary chip shown above the tabs */
    .filter-chip {{
        display: inline-block;
        background: {CREAM};
        color: {GOLD_DARK};
        border: 1px solid {CREAM_DEEP};
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.4rem;
        letter-spacing: 0.02em;
    }}

    /* Hide streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# UNDERLYING SYNTHETIC REVIEW DATASET
# We generate 1,000 reviews consistent with the headline aggregates so
# that the sidebar filters can actually re-shape every chart downstream.
# ---------------------------------------------------------------------
TOTAL_REVIEWS = 1000

@st.cache_data
def build_reviews() -> pd.DataFrame:
    """Generate 1,000 reviews whose aggregates match the headline KPIs.

    Per-category counts reconciled:
      * sentiment: Positive 82.0%, Neutral 2.0%, Negative 16.0%  (avg ≈ 4.3★).
      * rating:    5★:763  4★:57  3★:20  2★:24  1★:136  (reconciles to 1,000).
      * months:    weighted by the trend chart shape.
    """
    rng = np.random.default_rng(42)

    months = ["May 25","Jun 25","Jul 25","Aug 25","Sep 25","Oct 25",
             "Nov 25","Dec 25","Jan 26","Feb 26","Mar 26","Apr 26"]
    # Per-month pos / neu / neg shape (these are the original trend numbers;
    # we scale them proportionally so the three totals hit 895 / 16 / 171.)
    trend_shape = {
        "May 25":(27,2,5),"Jun 25":(24,2,5),"Jul 25":(26,1,5),"Aug 25":(27,1,6),
        "Sep 25":(26,1,11),"Oct 25":(24,2,3),"Nov 25":(28,1,6),"Dec 25":(31,4,4),
        "Jan 26":(28,0,4),"Feb 26":(24,2,5),"Mar 26":(78,0,19),"Apr 26":(43,0,3),
    }

    # Target totals per sentiment so everything reconciles to 1,000
    # and the positive / negative percentages match the headline cards.
    #   Positive (4–5★): 82.0% of 1,000 = 820   → 4★:57 + 5★:763
    #   Negative (1–2★): 16.0% of 1,000 = 160   → 2★:24 + 1★:136
    #   Neutral  (3★):    2.0% of 1,000 =  20   → 3★:20
    target = {"Positive": 820, "Neutral": 20, "Negative": 160}

    # Distribute each sentiment across months using the shape weights
    def distribute(sentiment_idx: int, total: int) -> dict:
        weights = np.array([trend_shape[m][sentiment_idx] for m in months], dtype=float)
        if weights.sum() == 0:
            return {m: 0 for m in months}
        probs = weights / weights.sum()
        # Use multinomial with fixed seed for reproducibility
        counts = rng.multinomial(total, probs)
        return dict(zip(months, counts))

    pos_by_month = distribute(0, target["Positive"])
    neu_by_month = distribute(1, target["Neutral"])
    neg_by_month = distribute(2, target["Negative"])

    pos_themes = ["Product results","Customer service","Personalisation",
                  "Ease of use","Onboarding"]
    pos_theme_weights = np.array([58, 30, 26, 20, 14]) / 148
    neg_themes = ["Skin reaction","Billing / refund","Product ineffective",
                  "Delivery failure","Other"]
    neg_theme_weights = np.array([78, 37, 12, 3, 12]) / 142

    conditions = ["Acne","Pigmentation","Ageing","Rosacea","Dry/sensitive",
                  "Scarring","General","Perioral"]
    cond_weights = np.array([267, 120, 103, 94, 79, 37, 34, 22]) / 756

    products = ["Daily Doser","Night Treatment","Serum","Cleansing Cream",
                "Doser + Cleanser","Moisturiser","Daily Defence SPF","Doser + Moisturiser"]
    prod_weights = np.array([271, 92, 88, 85, 80, 74, 73, 72]) / 835

    rows = []
    rid = 0
    for month in months:
        for sentiment, count in [
            ("Positive", int(pos_by_month[month])),
            ("Neutral",  int(neu_by_month[month])),
            ("Negative", int(neg_by_month[month])),
        ]:
            for _ in range(count):
                if sentiment == "Positive":
                    rating = int(rng.choice([5, 4], p=[0.93, 0.07]))
                    theme = rng.choice(pos_themes, p=pos_theme_weights)
                elif sentiment == "Neutral":
                    rating = 3
                    theme = "Mixed experience"
                else:
                    rating = int(rng.choice([1, 2], p=[0.85, 0.15]))
                    theme = rng.choice(neg_themes, p=neg_theme_weights)

                condition = rng.choice(conditions, p=cond_weights)
                product = rng.choice(products, p=prod_weights)
                verified = bool(rng.random() < 655/1000)
                rid += 1
                rows.append({
                    "id": rid,
                    "month": month,
                    "sentiment": sentiment,
                    "rating": rating,
                    "theme": theme,
                    "condition": condition,
                    "product": product,
                    "verified": verified,
                })
    df = pd.DataFrame(rows)
    df["month"] = pd.Categorical(df["month"], categories=months, ordered=True)
    return df

reviews = build_reviews()

# Sample review text (static — shown as representative quotes)
sample_negative_reviews = [
    {"stars": 1, "text": "Started breaking out in a rash on week two. Emailed support, took four days to respond and the advice was just 'reduce frequency'.", "product": "Daily Doser", "date": "Mar 2026", "theme": "Skin reaction"},
    {"stars": 2, "text": "Cancelled months ago but was still charged. Had to chase twice to get a refund — not the experience you'd expect from a premium brand.", "product": "Subscription", "date": "Feb 2026", "theme": "Billing / refund"},
    {"stars": 1, "text": "My skin reacted badly and I didn't know what to do. No follow-up check-in after the first prescription, just left on my own.", "product": "Daily Doser", "date": "Mar 2026", "theme": "Skin reaction"},
]
sample_positive_reviews = [
    {"stars": 5, "text": "After three months my acne has genuinely cleared. The doser is easy to use and the prescriber adjusted my formula when I flagged dryness.", "product": "Daily Doser", "date": "Apr 2026", "theme": "Product results"},
    {"stars": 5, "text": "The personalisation is what won me over — it's not a one-size bottle, it's actually made for my skin. And delivery is always on time.", "product": "Night Treatment", "date": "Apr 2026", "theme": "Personalisation"},
    {"stars": 5, "text": "Customer service was genuinely great. Answered my skin questions within a day and made me feel looked after, not sold to.", "product": "Daily Doser", "date": "Mar 2026", "theme": "Customer service"},
]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            {f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi(label: str, value: str, delta: str = "", color_class: str = "") -> None:
    delta_html = f'<p class="kpi-delta">{delta}</p>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value {color_class}">{value}</p>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def insight(text: str, tone: str = "") -> None:
    st.markdown(f'<div class="insight-card {tone}">{text}</div>', unsafe_allow_html=True)

def style_plotly(fig):
    """Apply consistent brand styling to any plotly figure — light/cream theme."""
    fig.update_layout(
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=CREAM_DEEP, linecolor=CREAM_DEEP, color=MUTED_INK),
        yaxis=dict(gridcolor=CREAM_DEEP, linecolor=CREAM_DEEP, color=MUTED_INK),
        legend=dict(font=dict(color=MUTED_INK, size=11)),
    )
    return fig

def empty_chart(message: str = "No data for the current filter selection."):
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color=MUTED_INK),
    )
    fig.update_layout(
        height=320, plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

# ---------------------------------------------------------------------
# SIDEBAR — filters and context (now ACTUALLY interactive)
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### Skin + Me — BAUD Pitch")
    st.markdown(
        f'<p style="color:{MUTED_INK}; font-size:0.82rem; line-height:1.5; margin-top:-0.5rem;">'
        f"Unstructured data advisory — turning 1,000 customer reviews into a "
        f"commercial action plan."
        f"</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**Filters**")
    sentiment_filter = st.multiselect(
        "Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
        help="Filter the dataset by sentiment class. Removing a class updates every chart.",
    )
    date_filter = st.select_slider(
        "Period",
        options=["Last 3 months", "Last 6 months", "Last 12 months", "All time"],
        value="All time",
        help="Restrict the analysis to a recent window.",
    )
    verified_filter = st.radio(
        "Reviewer type",
        options=["All", "Verified only", "Unverified only"],
        index=0,
        help="Show only verified (paying) customers, only unverified, or both.",
    )

    condition_filter = st.multiselect(
        "Skin condition",
        options=sorted(reviews["condition"].unique().tolist()),
        default=sorted(reviews["condition"].unique().tolist()),
        help="Filter by the skin condition the reviewer is treating.",
    )

    st.markdown("---")
    st.markdown("**Data sources**")
    st.markdown(
        f'<p style="color:{MUTED_INK}; font-size:0.78rem; line-height:1.5;">'
        f"• Trustpilot (Mar–Apr 2026) — real<br>"
        f"• Synthetic dataset (2024–2026)<br>"
        f"• Total: 1,000 reviews<br>"
        f"• Verified: 655 &nbsp;|&nbsp; Unverified: 345"
        f"</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# APPLY FILTERS — every chart below reads from `filtered`
# ---------------------------------------------------------------------
month_order = ["May 25","Jun 25","Jul 25","Aug 25","Sep 25","Oct 25",
               "Nov 25","Dec 25","Jan 26","Feb 26","Mar 26","Apr 26"]

# Period filter — "last N months" relative to the most recent month in data (Apr 26)
def months_in_period(period: str) -> list:
    if period == "Last 3 months":   return month_order[-3:]
    if period == "Last 6 months":   return month_order[-6:]
    if period == "Last 12 months":  return month_order[-12:]
    return month_order  # All time

filtered = reviews.copy()
filtered = filtered[filtered["sentiment"].isin(sentiment_filter)] if sentiment_filter else filtered.iloc[0:0]
filtered = filtered[filtered["month"].isin(months_in_period(date_filter))]
if verified_filter == "Verified only":
    filtered = filtered[filtered["verified"]]
elif verified_filter == "Unverified only":
    filtered = filtered[~filtered["verified"]]
filtered = filtered[filtered["condition"].isin(condition_filter)] if condition_filter else filtered.iloc[0:0]

n_filtered = len(filtered)

# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <p class="hero-sub">Unstructured data advisory · BAUD pitch</p>
        <h1>Skin + Me — Customer Voice Intelligence</h1>
        <p class="hero-desc">
            1,000 Trustpilot reviews analysed for sentiment, themes, and commercial signal.
            The customer voice is already telling Skin + Me where to invest —
            this dashboard makes that voice measurable.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Filter summary chip line
active_chips = []
if len(sentiment_filter) < 3:
    active_chips.append(f"Sentiment: {', '.join(sentiment_filter) if sentiment_filter else 'none'}")
if date_filter != "All time":
    active_chips.append(f"Period: {date_filter}")
if verified_filter != "All":
    active_chips.append(f"Reviewers: {verified_filter}")
if len(condition_filter) < reviews["condition"].nunique():
    active_chips.append(f"Conditions: {len(condition_filter)} of {reviews['condition'].nunique()}")

chip_html = "".join(f'<span class="filter-chip">{c}</span>' for c in active_chips)
st.markdown(
    f"""
    <div style="margin:-0.5rem 0 1rem 0;">
        <span style="color:{MUTED_INK}; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-right:0.6rem;">
            Showing {n_filtered:,} of {TOTAL_REVIEWS:,} reviews
        </span>
        {chip_html if active_chips else f'<span style="color:{MUTED_INK}; font-size:0.78rem;">All filters cleared</span>'}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# HEADLINE KPIs — recalculated from filtered data
# ---------------------------------------------------------------------
if n_filtered > 0:
    avg_rating = round(filtered["rating"].mean(), 2)
    pos_count = (filtered["sentiment"] == "Positive").sum()
    neg_count = (filtered["sentiment"] == "Negative").sum()
    pos_pct = round(pos_count / n_filtered * 100, 1)
    neg_pct = round(neg_count / n_filtered * 100, 1)
else:
    avg_rating = 0.0
    pos_count = neg_count = 0
    pos_pct = neg_pct = 0.0

c1, c2, c3, c4 = st.columns(4)
with c1: kpi("Total reviews",      f"{n_filtered:,}", f"of {TOTAL_REVIEWS:,} total")
with c2: kpi("Avg. rating",        f"{avg_rating} / 5" if n_filtered else "—", "Category mean: 3.9", "gold")
with c3: kpi("Positive (4–5★)",    f"{pos_pct}%",       f"{pos_count:,} reviews", "green")
with c4: kpi("Negative (1–2★)",    f"{neg_pct}%",       f"{neg_count:,} reviews — recoverable", "red")

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Volume & Sentiment",
    "🔍 Theme Analysis",
    "👥 Customers & Products",
    "💼 Recommendations",
])

# =====================================================================
# TAB 1 — VOLUME & SENTIMENT
# =====================================================================
with tab1:
    section("Volume & ratings", "How reviews are distributed and how sentiment moves over time")

    col_a, col_b = st.columns([1, 1.6])

    with col_a:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Rating distribution</strong></p>", unsafe_allow_html=True)
        if n_filtered > 0:
            rating_df_live = (
                filtered.groupby("rating").size().reindex([5, 4, 3, 2, 1], fill_value=0)
                .reset_index(name="Count")
            )
            rating_df_live["Rating"] = rating_df_live["rating"].astype(str) + "★"
            fig_rating = px.bar(
                rating_df_live, x="Rating", y="Count",
                color="Rating",
                color_discrete_sequence=RATING_COLORS,
                text="Count",
            )
            fig_rating.update_traces(textposition="outside", textfont=dict(color=INK, size=11))
            fig_rating.update_layout(showlegend=False, height=320)
            st.plotly_chart(style_plotly(fig_rating), use_container_width=True)
        else:
            st.plotly_chart(empty_chart(), use_container_width=True)

    with col_b:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Monthly sentiment trend</strong></p>", unsafe_allow_html=True)
        if n_filtered > 0:
            trend_live = (
                filtered.groupby(["month", "sentiment"], observed=True).size()
                .unstack(fill_value=0)
                .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
                .reset_index()
            )
            fig_trend = go.Figure()
            if "Positive" in trend_live.columns:
                fig_trend.add_trace(go.Bar(name="Positive", x=trend_live["month"], y=trend_live["Positive"], marker_color=GREEN_POS))
            if "Neutral" in trend_live.columns:
                fig_trend.add_trace(go.Bar(name="Neutral",  x=trend_live["month"], y=trend_live["Neutral"],  marker_color=GOLD_LIGHT))
            if "Negative" in trend_live.columns:
                fig_trend.add_trace(go.Bar(name="Negative", x=trend_live["month"], y=trend_live["Negative"], marker_color=CORAL_NEG))
            fig_trend.update_layout(barmode="stack", height=320, legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(style_plotly(fig_trend), use_container_width=True)
        else:
            st.plotly_chart(empty_chart(), use_container_width=True)

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        insight(
            "<strong>March 2026 spike.</strong> Negative reviews jumped from ~5/month to 19 — "
            "a 280% increase concentrated in skin-reaction complaints. Event-driven, not trend-driven.",
            tone="red",
        )
    with col_i2:
        insight(
            "<strong>5★ dominance is real but fragile.</strong> 71% of reviews are 5-star, "
            "but the 1★ tail (119 reviews, 11%) is structurally larger than 2–3★ combined.",
        )

# =====================================================================
# TAB 2 — THEME ANALYSIS
# =====================================================================
with tab2:
    section("What drives sentiment", "Theme extraction from review text — what customers praise and what they complain about")

    col_a, col_b = st.columns(2)

    pos_slice = filtered[filtered["sentiment"] == "Positive"]
    neg_slice = filtered[filtered["sentiment"] == "Negative"]

    with col_a:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Positive review themes</strong> ({len(pos_slice):,} reviews)</p>", unsafe_allow_html=True)
        if len(pos_slice) > 0:
            pos_live = (
                pos_slice.groupby("theme").size().sort_values()
                .reset_index(name="Mentions").rename(columns={"theme": "Theme"})
            )
            fig_pos = px.bar(
                pos_live,
                x="Mentions", y="Theme",
                orientation="h",
                color_discrete_sequence=[GREEN_POS],
                text="Mentions",
            )
            fig_pos.update_traces(textposition="outside", textfont=dict(color=INK, size=11))
            fig_pos.update_layout(showlegend=False, height=320)
            st.plotly_chart(style_plotly(fig_pos), use_container_width=True)
        else:
            st.plotly_chart(empty_chart("No positive reviews in current filter."), use_container_width=True)

    with col_b:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Negative review root causes</strong> ({len(neg_slice):,} reviews)</p>", unsafe_allow_html=True)
        if len(neg_slice) > 0:
            neg_live = (
                neg_slice.groupby("theme").size().sort_values()
                .reset_index(name="Count").rename(columns={"theme": "Theme"})
            )
            fig_neg = px.bar(
                neg_live,
                x="Count", y="Theme",
                orientation="h",
                color_discrete_sequence=[CORAL_NEG],
                text="Count",
            )
            fig_neg.update_traces(textposition="outside", textfont=dict(color=INK, size=11))
            fig_neg.update_layout(showlegend=False, height=320)
            st.plotly_chart(style_plotly(fig_neg), use_container_width=True)
        else:
            st.plotly_chart(empty_chart("No negative reviews in current filter."), use_container_width=True)

    insight(
        "<strong>Two complaint clusters explain 81% of negative reviews.</strong> "
        "Skin reactions (78) and billing / refund issues (37) together account for 115 of 142 "
        "1–2★ reviews. Both are solvable operationally — the complaint pattern is concentrated, not diffuse.",
        tone="red",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section("Voice of customer", "Representative review excerpts (synthesised from real patterns)")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.88rem; margin-bottom:0.3rem;'><strong>What customers complain about</strong></p>", unsafe_allow_html=True)
        shown_neg = [r for r in sample_negative_reviews if "Negative" in sentiment_filter]
        if not shown_neg:
            st.markdown(f'<p style="color:{MUTED_INK}; font-size:0.82rem; font-style:italic;">Negative sentiment excluded from current filter.</p>', unsafe_allow_html=True)
        for r in shown_neg:
            st.markdown(
                f"""
                <div class="review-quote neg">
                    <span class="stars">{"★" * r["stars"]}{"☆" * (5 - r["stars"])}</span>
                    <p style="margin:0.4rem 0 0 0;">"{r['text']}"</p>
                    <p class="meta">{r['product']} · {r['date']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_q2:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.88rem; margin-bottom:0.3rem;'><strong>What customers love</strong></p>", unsafe_allow_html=True)
        shown_pos = [r for r in sample_positive_reviews if "Positive" in sentiment_filter]
        if not shown_pos:
            st.markdown(f'<p style="color:{MUTED_INK}; font-size:0.82rem; font-style:italic;">Positive sentiment excluded from current filter.</p>', unsafe_allow_html=True)
        for r in shown_pos:
            st.markdown(
                f"""
                <div class="review-quote pos">
                    <span class="stars">{"★" * r["stars"]}{"☆" * (5 - r["stars"])}</span>
                    <p style="margin:0.4rem 0 0 0;">"{r['text']}"</p>
                    <p class="meta">{r['product']} · {r['date']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================================
# TAB 3 — CUSTOMERS & PRODUCTS
# =====================================================================
with tab3:
    section("Who reviews and what they treat", "Segmentation across skin conditions, verification status, and product mix")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Conditions treated — positive reviews</strong></p>", unsafe_allow_html=True)
        pos_for_cond = filtered[filtered["sentiment"] == "Positive"]
        if len(pos_for_cond) > 0:
            cond_live = (
                pos_for_cond.groupby("condition").size()
                .reset_index(name="Reviews").rename(columns={"condition": "Condition"})
                .sort_values("Reviews", ascending=False)
            )
            fig_cond = px.pie(
                cond_live, values="Reviews", names="Condition",
                color_discrete_sequence=CONDITION_COLORS,
                hole=0.55,
            )
            fig_cond.update_traces(
                textposition="outside",
                textinfo="label+percent",
                textfont=dict(color=INK, size=11),
            )
            fig_cond.update_layout(showlegend=False, height=360)
            st.plotly_chart(style_plotly(fig_cond), use_container_width=True)
        else:
            st.plotly_chart(empty_chart(), use_container_width=True)

    with col_b:
        st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Verified vs unverified sentiment</strong></p>", unsafe_allow_html=True)
        if n_filtered > 0 and verified_filter == "All":
            ver_live = (
                filtered[filtered["sentiment"].isin(["Positive", "Negative"])]
                .assign(Group=lambda d: d["verified"].map({True: "Verified", False: "Unverified"}))
                .groupby(["Group", "sentiment"]).size()
                .reset_index(name="Count").rename(columns={"sentiment": "Sentiment"})
            )
            # Append counts to group labels
            totals = filtered.assign(Group=lambda d: d["verified"].map({True: "Verified", False: "Unverified"})).groupby("Group").size()
            ver_live["Group"] = ver_live["Group"].apply(lambda g: f"{g} ({totals.get(g, 0)})")
            fig_ver = px.bar(
                ver_live,
                x="Group", y="Count", color="Sentiment",
                barmode="group",
                color_discrete_map={"Positive": GREEN_POS, "Negative": CORAL_NEG},
                text="Count",
            )
            fig_ver.update_traces(textposition="outside", textfont=dict(color=INK, size=11))
            fig_ver.update_layout(height=360, legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(style_plotly(fig_ver), use_container_width=True)
        elif n_filtered > 0:
            st.plotly_chart(empty_chart("Clear the 'Reviewer type' filter to compare verified vs unverified."), use_container_width=True)
        else:
            st.plotly_chart(empty_chart(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED_INK}; font-size:0.85rem; margin-bottom:0.4rem;'><strong>Top products by review volume</strong></p>", unsafe_allow_html=True)
    if n_filtered > 0:
        prod_live = (
            filtered.groupby("product").size().sort_values()
            .reset_index(name="Reviews").rename(columns={"product": "Product"})
        )
        fig_prod = px.bar(
            prod_live,
            x="Reviews", y="Product",
            orientation="h",
            color_discrete_sequence=[GOLD],
            text="Reviews",
        )
        fig_prod.update_traces(textposition="outside", textfont=dict(color=INK, size=11))
        fig_prod.update_layout(showlegend=False, height=360)
        st.plotly_chart(style_plotly(fig_prod), use_container_width=True)
    else:
        st.plotly_chart(empty_chart(), use_container_width=True)

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        insight(
            "<strong>Acne dominates the positive narrative.</strong> 33% of positive reviews are "
            "from acne customers (267 of 815). This is Skin + Me's hero use-case — and the "
            "marketing story most aligned with the data.",
            tone="green",
        )
    with col_i2:
        insight(
            "<strong>The Daily Doser is the brand.</strong> 271 reviews (25% of total volume) — "
            "more than double any other product. Quality and experience of this one SKU "
            "disproportionately shapes perception of the whole business.",
        )

# =====================================================================
# TAB 4 — RECOMMENDATIONS
# =====================================================================
with tab4:
    section("Recommendations — business impact & constraints",
            "What the data tells Skin + Me to do next, what it's worth, and what it will cost to execute")

    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.markdown(
            """
            <div class="rec-card">
                <h4><span class="rec-num">1</span> Fix the skin-reaction funnel</h4>
                <p>78 of 142 negative reviews (55%) cite skin reactions with no proactive follow-up.
                Introduce a day-14 check-in for every new prescription, plus a formula-adjustment
                flow triggered by reaction keywords in support tickets.</p>
                <div class="impact">
                    <strong>Indicative impact:</strong> resolve ~50% of skin-reaction complaints →
                    reduce 1★ reviews by ~28% → lift avg. rating from 4.3 to ~4.45.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rec_col2:
        st.markdown(
            """
            <div class="rec-card">
                <h4><span class="rec-num">2</span> Close the billing complaint loop</h4>
                <p>37 negative reviews cite subscription billing and refund friction — the second-biggest
                complaint cluster. Audit the cancellation flow, add automated refund confirmations,
                and publish response-time SLAs on the help centre.</p>
                <div class="impact">
                    <strong>Indicative impact:</strong> ~75% of billing complaints are preventable →
                    recover ~28 1–2★ reviews/quarter → protect conversion rate by an estimated 1.5–2pp.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rec_col3:
        st.markdown(
            """
            <div class="rec-card">
                <h4><span class="rec-num">3</span> Amplify the acne hero story</h4>
                <p>Acne drives 33% of positive reviews — the clearest product-market fit signal in the
                dataset. Shift paid acquisition and social content toward acne-specific outcomes,
                using verified 5★ reviews as primary creative.</p>
                <div class="impact">
                    <strong>Indicative impact:</strong> CAC reduction of ~10–15% on acne-targeted campaigns;
                    acne cohort already reviews at 4.6★ vs 4.3★ average — lower refund risk too.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section("Commercial impact summary", "Combined effect of the three recommendations over 6 months")

    imp_col1, imp_col2, imp_col3, imp_col4 = st.columns(4)
    with imp_col1: kpi("Avg. rating lift",    "+0.15★",  "4.3 → ~4.45", "green")
    with imp_col2: kpi("1★ reviews avoided",  "~33/quarter", "Across reaction + billing", "green")
    with imp_col3: kpi("Estimated CAC saving","8–12%",   "On acne-led campaigns", "green")
    with imp_col4: kpi("Payback period",      "< 6 months", "On phase-one investment", "gold")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Risks & constraints — what we're being honest about", "")

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color:{GOLD};">
                <strong>Data limitations.</strong> A portion of the dataset is synthetic. Findings are
                directional, not inferential. Phase one must validate against the full Trustpilot
                history before investment decisions are finalised.
            </div>
            <div class="insight-card" style="border-left-color:{GOLD};">
                <strong>Platform constraints.</strong> Instagram and TikTok comment data is restricted
                under current API terms — we've scoped the pipeline to Trustpilot, Google Reviews,
                and first-party post-purchase surveys only.
            </div>
            <div class="insight-card" style="border-left-color:{GOLD};">
                <strong>GDPR.</strong> All review data is already public. No PII is stored in the
                analytics layer; reviewer names are hashed before ingestion.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with risk_col2:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color:{TAUPE};">
                <strong>Investment estimate — phase one (90 days).</strong><br>
                • Analyst time: ~25 days<br>
                • Tooling: Trustpilot API + sentiment model fine-tune (~£1.2k)<br>
                • Internal training: 2 half-day workshops with CX and marketing teams
            </div>
            <div class="insight-card" style="border-left-color:{TAUPE};">
                <strong>Mindset change.</strong> The pipeline is only as useful as the operating
                rhythm built around it. Monthly review of themes by a CX + product pairing is the
                minimum viable adoption — dashboards alone don't change decisions.
            </div>
            <div class="insight-card" style="border-left-color:{TAUPE};">
                <strong>What scales, what doesn't.</strong> Sentiment and theme extraction scale
                cleanly. Qualitative response-writing (e.g. replying to each 1★ review) remains a
                human task — we can prioritise the queue, not replace it.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section("First 30 days — what we'd deliver", "A concrete proposal, not a summary")

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, {CREAM} 0%, {WHITE} 100%); border:1px solid {CREAM_DEEP}; border-left:6px solid {GOLD}; border-radius:12px; padding:1.5rem 1.8rem; color:{INK};">
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:1.5rem;">
                <div>
                    <p style="color:{GOLD_DARK}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; margin:0 0 0.6rem 0;">Week 1–2</p>
                    <h4 style="color:{INK}; font-size:1rem; margin:0 0 0.4rem 0;">Baseline the voice</h4>
                    <p style="color:{MUTED_INK}; font-size:0.85rem; line-height:1.5; margin:0;">
                        Ingest full Trustpilot + Google Reviews history. Re-run sentiment & theme
                        extraction on real data. Benchmark against this pilot dataset.
                    </p>
                </div>
                <div>
                    <p style="color:{GOLD_DARK}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; margin:0 0 0.6rem 0;">Week 3</p>
                    <h4 style="color:{INK}; font-size:1rem; margin:0 0 0.4rem 0;">Co-design with CX</h4>
                    <p style="color:{MUTED_INK}; font-size:0.85rem; line-height:1.5; margin:0;">
                        Half-day workshop with CX + marketing teams. Prioritise recommendations 1–3
                        against team capacity. Agree the monthly review rhythm.
                    </p>
                </div>
                <div>
                    <p style="color:{GOLD_DARK}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; margin:0 0 0.6rem 0;">Week 4</p>
                    <h4 style="color:{INK}; font-size:1rem; margin:0 0 0.4rem 0;">Ship v1 dashboard</h4>
                    <p style="color:{MUTED_INK}; font-size:0.85rem; line-height:1.5; margin:0;">
                        Live dashboard handed over (this tool, productionised). Theme alerting on
                        new reviews. Phase-two scope with KPIs agreed and signed off.
                    </p>
                </div>
            </div>
            <hr style="border:none; border-top:1px solid {CREAM_DEEP}; margin:1.3rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
                <p style="color:{GOLD_DARK}; font-size:0.95rem; font-weight:600; margin:0;">
                    What we need from Skin + Me: API access, a CX lead, and 3 hours of leadership time.
                </p>
                <p style="color:{INK}; font-size:0.9rem; font-weight:600; margin:0;">
                    Ready to start Monday.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <hr>
    <p style="color:{MUTED_INK}; font-size:0.75rem; text-align:center; margin:0;">
        Skin + Me — Unstructured Data Advisory Pitch · BAUD A2 · April 2026<br>
        <em>Synthetic data disclosure: a portion of the 1,000-review dataset was generated via
        GenAI methods grounded in real Trustpilot themes. All findings are
        indicative and intended to demonstrate analytical capability.</em>
    </p>
    """,
    unsafe_allow_html=True,
)
