import streamlit as st
import pandas as pd
import joblib
import os
import sys
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import clean_text

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Sentiment Analyser",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global Styles ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ── Hero Section ── */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}

/* ── Step Cards ── */
.step-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
}
.step-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.step-badge {
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.step-title {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-positive .metric-value { color: #34d399; }
.metric-negative .metric-value { color: #f87171; }
.metric-neutral .metric-value { color: #60a5fa; }
.metric-purple .metric-value { color: #a78bfa; }

/* ── Section Headers ── */
.section-header {
    color: #e2e8f0;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ── Insight Cards ── */
.insight-positive {
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #34d399;
    font-weight: 500;
}
.insight-negative {
    background: rgba(248, 113, 113, 0.1);
    border: 1px solid rgba(248, 113, 113, 0.3);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #f87171;
    font-weight: 500;
}

/* ── Tag Chips ── */
.tag-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0; }
.tag {
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #a78bfa;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(167, 139, 250, 0.4);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167, 139, 250, 0.8);
}

/* ── Analyse Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

/* ── Info / Warning / Success boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: none;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 26, 0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(167, 139, 250, 0.3);
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ─── Chart Theme ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#1e1e3a',
    'axes.edgecolor': '#2d2d4e',
    'axes.labelcolor': '#94a3b8',
    'text.color': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#2d2d4e',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

POS_COLOR = '#34d399'
NEG_COLOR = '#f87171'
CONF_COLOR = '#60a5fa'
PURPLE = '#a78bfa'

# ─── Constants ────────────────────────────────────────────────────
SENTIMENT_WORDS = {
    'good','great','excellent','amazing','love','best','perfect',
    'wonderful','fantastic','delicious','recommend','fresh','quality',
    'bad','terrible','awful','horrible','worst','disappointing',
    'poor','waste','disgusting','tasteless','stale','overpriced',
    'not','never','but','however','although','unfortunately',
    'happy','unhappy','satisfied','unsatisfied','enjoyed','hated',
    'loved','disliked','superb','mediocre','outstanding','dreadful'
}

STOPWORDS = {
    'the','and','is','in','it','of','to','a','an','this','was',
    'for','on','are','with','as','i','at','be','that','have',
    'had','but','not','they','from','or','my','so','we','its',
    'very','just','their','been','has','would','could','should',
    'also','more','than','about','one','get','got','like','will',
    'what','all','product','wine','beer','food','coffee','tea',
    'even','when','then','them','these','those','which','who',
    'after','before','because','some','any','each','into','over',
    'such','only','other','same','both','much','many','most',
    'did','does','how','our','your','him','her','his','she','he'
}

# ─── Load Models ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    model = joblib.load(os.path.join(base, 'logistic_model.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'tfidf_vectorizer.pkl'))
    return model, vectorizer

model, vectorizer = load_models()

# ─── Helper Functions ─────────────────────────────────────────────
def score_review_column(series):
    sample = series.dropna().astype(str).head(200)
    avg_length = sample.str.len().mean()
    avg_words = sample.str.split().apply(len).mean()
    unique_ratio = series.nunique() / max(len(series), 1)
    all_words = ' '.join(sample.tolist()).lower()
    word_tokens = set(re.findall(r'\b[a-z]+\b', all_words))
    sentiment_overlap = len(word_tokens & SENTIMENT_WORDS) / max(len(SENTIMENT_WORDS), 1)
    punct_diversity = sample.apply(
        lambda x: len(set(re.findall(r'[^\w\s]', x)))
    ).mean()
    title_penalty = 0.3 if (unique_ratio > 0.85 and sentiment_overlap < 0.1) else 1.0
    score = (
        (avg_length * 0.25) +
        (avg_words * 0.25) +
        (sentiment_overlap * 30) +
        (punct_diversity * 2)
    ) * title_penalty
    return round(score, 2)

def detect_review_columns(df, top_n=5):
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    scored = []
    for col in text_cols:
        avg_len = df[col].dropna().astype(str).str.len().mean()
        if avg_len > 15:
            score = score_review_column(df[col])
            scored.append((col, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(col, score) for col, score in scored[:top_n]]

def detect_categorical_columns(df, review_col, min_unique=2, max_unique=50):
    candidates = []
    for col in df.columns:
        if col == review_col:
            continue
        if df[col].dtype not in ['object', 'category']:
            if df[col].nunique() <= 10:
                candidates.append((col, df[col].nunique(), 'numeric'))
            continue
        n_unique = df[col].nunique()
        if n_unique < min_unique or n_unique > max_unique:
            continue
        sample = df[col].dropna().astype(str).head(50)
        avg_len = sample.str.len().mean()
        unique_ratio = n_unique / max(len(df), 1)
        if avg_len > 50 or unique_ratio > 0.5:
            continue
        candidates.append((col, n_unique, 'categorical'))
    candidates.sort(key=lambda x: x[1])
    return candidates

def detect_date_column(df, review_col):
    for col in df.columns:
        if col == review_col:
            continue
        col_lower = col.lower()
        if any(w in col_lower for w in ['date', 'time', 'year', 'month']):
            return col
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().head(10))
                return col
            except Exception:
                continue
    return None

def validate_review_column(series):
    sample = series.dropna().astype(str).head(100)
    avg_len = sample.str.len().mean()
    avg_words = sample.str.split().apply(len).mean()
    unique_ratio = series.nunique() / max(len(series), 1)
    all_words = ' '.join(sample.tolist()).lower()
    word_tokens = set(re.findall(r'\b[a-z]+\b', all_words))
    sentiment_overlap = len(word_tokens & SENTIMENT_WORDS) / max(len(SENTIMENT_WORDS), 1)
    if avg_len < 15:
        return False, f"Average length is {avg_len:.0f} characters — too short for reviews."
    if avg_words < 3:
        return False, f"Average word count is {avg_words:.1f} — may be a label column."
    if unique_ratio > 0.9 and sentiment_overlap < 0.05:
        return False, "High uniqueness and few opinion words — may be a title or ID column."
    return True, "Column looks good."

def analyse_reviews(df, column, batch_size=1000):
    reviews = df[column].fillna('').astype(str).tolist()
    predictions, confidences = [], []
    progress = st.progress(0, text="🔍 Analysing reviews...")
    total = len(reviews)
    for i in range(0, total, batch_size):
        batch = reviews[i:i+batch_size]
        cleaned = [clean_text(r) for r in batch]
        vectorised = vectorizer.transform(cleaned)
        preds = model.predict(vectorised)
        probs = model.predict_proba(vectorised)
        predictions.extend(preds.tolist())
        confidences.extend([round(max(p), 3) for p in probs])
        progress.progress(
            min((i + batch_size) / total, 1.0),
            text=f"🔍 Analysing... {min(i+batch_size, total):,} / {total:,}"
        )
    progress.empty()
    result_df = df.copy()
    result_df['_review_text'] = df[column].fillna('').astype(str)
    result_df['_review_length'] = result_df['_review_text'].str.split().apply(len)
    result_df['sentiment'] = ['Positive' if p == 1 else 'Negative' for p in predictions]
    result_df['confidence'] = confidences
    return result_df

def get_top_keywords(series, n=15):
    words = []
    for text in series.dropna():
        tokens = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    return Counter(words).most_common(n)

# ─── Chart Functions ───────────────────────────────────────────────
def make_fig(w=6, h=4):
    return plt.subplots(figsize=(w, h))

def plot_donut(pos_count, neg_count):
    fig, ax = make_fig(5, 5)
    total = pos_count + neg_count
    sizes = [pos_count, neg_count]
    colors = [POS_COLOR, NEG_COLOR]
    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.48, edgecolor='#1a1a2e', linewidth=3)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
        at.set_color('white')
    ax.text(0, 0, f'{total:,}', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#e2e8f0')
    ax.text(0, -0.25, 'reviews', ha='center', va='center',
            fontsize=9, color='#94a3b8')
    ax.legend(
        [f'Positive  {pos_count:,}', f'Negative  {neg_count:,}'],
        loc='lower center', bbox_to_anchor=(0.5, -0.08),
        ncol=2, frameon=False,
        labelcolor='#94a3b8', fontsize=9
    )
    ax.set_title('Sentiment Split', fontsize=13,
                fontweight='bold', color='#e2e8f0', pad=15)
    plt.tight_layout()
    return fig

def plot_confidence_distribution(result_df):
    fig, ax = make_fig(6, 4)
    pos_conf = result_df[result_df['sentiment']=='Positive']['confidence']
    neg_conf = result_df[result_df['sentiment']=='Negative']['confidence']
    ax.hist(pos_conf, bins=25, alpha=0.75, color=POS_COLOR,
            label=f'Positive', edgecolor='none')
    ax.hist(neg_conf, bins=25, alpha=0.75, color=NEG_COLOR,
            label=f'Negative', edgecolor='none')
    ax.axvline(x=pos_conf.mean(), color=POS_COLOR, linestyle='--',
               alpha=0.9, linewidth=1.5)
    ax.axvline(x=neg_conf.mean(), color=NEG_COLOR, linestyle='--',
               alpha=0.9, linewidth=1.5)
    ax.set_title('Confidence Distribution', fontsize=13,
                fontweight='bold', color='#e2e8f0')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, labelcolor='#94a3b8')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def plot_review_length_vs_confidence(result_df):
    fig, ax = make_fig(6, 4)
    sample = result_df.sample(min(2000, len(result_df)), random_state=42)
    pos = sample[sample['sentiment'] == 'Positive']
    neg = sample[sample['sentiment'] == 'Negative']
    ax.scatter(neg['_review_length'], neg['confidence'],
               c=NEG_COLOR, alpha=0.3, s=12, label='Negative')
    ax.scatter(pos['_review_length'], pos['confidence'],
               c=POS_COLOR, alpha=0.3, s=12, label='Positive')
    ax.set_title('Length vs Confidence', fontsize=13,
                fontweight='bold', color='#e2e8f0')
    ax.set_xlabel('Review Length (words)')
    ax.set_ylabel('Confidence Score')
    ax.set_xlim(0, min(500, int(sample['_review_length'].quantile(0.99))))
    ax.legend(frameon=False, labelcolor='#94a3b8')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def plot_category_sentiment(result_df, cat_col, top_n=10):
    counts = result_df.groupby(
        [cat_col, 'sentiment']
    ).size().unstack(fill_value=0)
    for col in ['Positive', 'Negative']:
        if col not in counts.columns:
            counts[col] = 0
    counts['Total'] = counts['Positive'] + counts['Negative']
    counts['Positive %'] = (counts['Positive'] / counts['Total'] * 100).round(1)
    counts['Negative %'] = (counts['Negative'] / counts['Total'] * 100).round(1)
    counts = counts.sort_values('Total', ascending=False).head(top_n)
    counts = counts.sort_values('Positive %', ascending=True)
    fig, ax = make_fig(10, max(4, len(counts) * 0.75))
    y = range(len(counts))
    ax.barh(y, counts['Positive %'], color=POS_COLOR,
            label='Positive', alpha=0.85, height=0.6)
    ax.barh(y, counts['Negative %'], left=counts['Positive %'],
            color=NEG_COLOR, label='Negative', alpha=0.85, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=10)
    ax.set_xlabel('Percentage (%)')
    ax.set_title(f'Sentiment by {cat_col}', fontsize=13,
                fontweight='bold', color='#e2e8f0')
    ax.legend(loc='lower right', frameon=False, labelcolor='#94a3b8')
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for i, (pos_pct, neg_pct) in enumerate(
        zip(counts['Positive %'], counts['Negative %'])
    ):
        if pos_pct > 10:
            ax.text(pos_pct/2, i, f'{pos_pct:.0f}%', va='center',
                   ha='center', color='white', fontsize=9, fontweight='bold')
        if neg_pct > 10:
            ax.text(pos_pct + neg_pct/2, i, f'{neg_pct:.0f}%',
                   va='center', ha='center', color='white',
                   fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig, counts

def plot_heatmap(result_df, cat_col1, cat_col2, top_n=8):
    top1 = result_df[cat_col1].value_counts().head(top_n).index
    top2 = result_df[cat_col2].value_counts().head(top_n).index
    filtered = result_df[
        result_df[cat_col1].isin(top1) &
        result_df[cat_col2].isin(top2)
    ]
    pivot = filtered.groupby([cat_col1, cat_col2]).apply(
        lambda x: round((x['sentiment'] == 'Positive').mean() * 100, 1)
    ).unstack(fill_value=np.nan)
    if pivot.empty:
        return None
    fig, ax = make_fig(
        max(8, len(pivot.columns) * 1.2),
        max(5, len(pivot.index) * 0.85)
    )
    im = ax.imshow(pivot.values, cmap='RdYlGn',
                   aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(
        f'Positive % — {cat_col1} × {cat_col2}',
        fontsize=13, fontweight='bold', color='#e2e8f0'
    )
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color='white' if val < 40 or val > 75 else 'black')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    cbar.set_label('Positive %', color='#94a3b8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8')
    plt.tight_layout()
    return fig

def plot_keyword_bar(keywords, title, color):
    if not keywords:
        return None
    words, counts = zip(*keywords)
    fig, ax = make_fig(7, 4.5)
    bars = ax.barh(range(len(words)), counts,
                   color=color, edgecolor='none', alpha=0.85, height=0.65)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight='bold', color='#e2e8f0')
    ax.set_xlabel('Frequency')
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9, color='#94a3b8')
    plt.tight_layout()
    return fig

def plot_sentiment_trend(result_df, date_col):
    try:
        df_t = result_df.copy()
        df_t[date_col] = pd.to_datetime(df_t[date_col], errors='coerce')
        df_t = df_t.dropna(subset=[date_col])
        if len(df_t) < 10:
            return None
        df_t['month'] = df_t[date_col].dt.to_period('M')
        trend = df_t.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        for col in ['Positive', 'Negative']:
            if col not in trend.columns:
                trend[col] = 0
        trend['Total'] = trend['Positive'] + trend['Negative']
        trend['Positive %'] = (trend['Positive'] / trend['Total'] * 100).round(1)
        if len(trend) < 2:
            return None
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 6),
            sharex=True, facecolor='#1a1a2e'
        )
        ax1.set_facecolor('#1e1e3a')
        ax2.set_facecolor('#1e1e3a')
        x = range(len(trend))
        ax1.plot(x, trend['Positive %'], color=POS_COLOR,
                linewidth=2.5, marker='o', markersize=4)
        ax1.fill_between(x, trend['Positive %'],
                        alpha=0.15, color=POS_COLOR)
        ax1.axhline(y=50, color='#94a3b8', linestyle='--',
                   alpha=0.4, linewidth=1)
        ax1.set_ylabel('Positive %', color='#94a3b8')
        ax1.set_ylim(0, 100)
        ax1.set_title('Sentiment Trend Over Time',
                     fontsize=13, fontweight='bold', color='#e2e8f0')
        ax1.yaxis.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)
        ax2.bar(x, trend['Positive'], color=POS_COLOR, alpha=0.75, label='Positive')
        ax2.bar(x, trend['Negative'], bottom=trend['Positive'],
               color=NEG_COLOR, alpha=0.75, label='Negative')
        ax2.set_ylabel('Review Count', color='#94a3b8')
        ax2.legend(frameon=False, labelcolor='#94a3b8')
        ax2.yaxis.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
        step = max(1, len(trend) // 12)
        ax2.set_xticks(list(x)[::step])
        ax2.set_xticklabels(
            trend.index.astype(str)[::step],
            rotation=45, ha='right', color='#94a3b8'
        )
        plt.tight_layout()
        return fig
    except Exception:
        return None

def plot_category_confidence_box(result_df, cat_col, top_n=10):
    top_cats = result_df[cat_col].value_counts().head(top_n).index
    filtered = result_df[result_df[cat_col].isin(top_cats)]
    grouped = [
        filtered[filtered[cat_col] == cat]['confidence'].values
        for cat in top_cats
    ]
    fig, ax = make_fig(8, max(4, len(top_cats) * 0.65))
    bp = ax.boxplot(
        grouped, vert=False, patch_artist=True,
        labels=top_cats,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='#94a3b8'),
        capprops=dict(color='#94a3b8'),
        flierprops=dict(marker='o', color=CONF_COLOR,
                       alpha=0.3, markersize=3)
    )
    for patch in bp['boxes']:
        patch.set_facecolor(CONF_COLOR)
        patch.set_alpha(0.5)
    ax.set_xlabel('Confidence Score')
    ax.set_title(f'Prediction Confidence by {cat_col}',
                fontsize=13, fontweight='bold', color='#e2e8f0')
    ax.axvline(x=0.5, color='#94a3b8', linestyle='--', alpha=0.4)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

# ─── UI ────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <h1>Review Sentiment Analyser</h1>
    <p>Upload any product review CSV and get instant AI-powered 
    sentiment insights — broken down by category, country, 
    product type, and more.</p>
</div>
""", unsafe_allow_html=True)

# Feature tags
st.markdown("""
<div class="tag-row" style="justify-content:center; margin-bottom:2rem;">
    <span class="tag">📁 CSV Upload</span>
    <span class="tag">🔍 Auto Column Detection</span>
    <span class="tag">🗂️ Category Analysis</span>
    <span class="tag">🔥 Heatmap</span>
    <span class="tag">📅 Trend Chart</span>
    <span class="tag">⬇️ Export Results</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Step 1: Upload ─────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-header">
        <span class="step-badge">Step 1</span>
        <p class="step-title">Upload Your CSV File</p>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file — up to 200MB",
    type=['csv'],
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(
            f"✅ **{uploaded_file.name}** loaded — "
            f"{len(df):,} rows · {len(df.columns)} columns"
        )
        with st.expander("👀 Preview data"):
            st.dataframe(df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ── Step 2: Review Column ──────────────────────────────────────
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <span class="step-badge">Step 2</span>
            <p class="step-title">Select Your Review Column</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    scored_cols = detect_review_columns(df)
    all_cols = df.columns.tolist()
    auto_cols = [col for col, _ in scored_cols]
    other_cols = [c for c in all_cols if c not in auto_cols]

    if scored_cols:
        score_df = pd.DataFrame(scored_cols, columns=['Column', 'Score'])
        score_df['Assessment'] = score_df['Score'].apply(
            lambda x: '✅ Strong' if x > 10
            else '⚠️ Possible' if x > 5
            else '❌ Unlikely'
        )
        with st.expander("🔍 View column rankings"):
            st.dataframe(score_df, use_container_width=True, hide_index=True)

    selected_col = st.selectbox(
        "Review column:",
        options=auto_cols + other_cols,
        help="Auto-detected columns shown first."
    )

    if selected_col:
        is_valid, reason = validate_review_column(df[selected_col])
        with st.expander("📋 Sample entries from selected column"):
            for i, s in enumerate(df[selected_col].dropna().head(3).tolist(), 1):
                st.info(f"**{i}.** {str(s)[:300]}")
        if not is_valid:
            st.warning(f"⚠️ {reason} — you can still proceed.")
        else:
            st.success("✅ This looks like a review column.")

    # ── Step 3: Category Columns ───────────────────────────────────
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <span class="step-badge">Step 3</span>
            <p class="step-title">Select Category Columns
            <span style="color:#64748b; font-weight:400; 
            font-size:0.9rem;"> — Optional</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Add category columns to compare sentiment across groups. "
        "Select two to unlock **cross-category heatmap** analysis."
    )

    use_category = st.checkbox("Add category analysis", value=True)
    selected_cat_cols = []

    if use_category:
        cat_candidates = detect_categorical_columns(df, selected_col)
        auto_cat_cols = [col for col, _, _ in cat_candidates]
        auto_labels = {
            col: f"⭐ {col}  ({n} unique)"
            for col, n, _ in cat_candidates
        }
        remaining = [
            c for c in df.columns
            if c != selected_col and c not in auto_cat_cols
        ]
        remaining_labels = {
            col: f"{col}  ({df[col].nunique()} unique)"
            for col in remaining
        }
        all_options = list(auto_labels.values()) + list(remaining_labels.values())
        reverse_map = {
            **{v: k for k, v in auto_labels.items()},
            **{v: k for k, v in remaining_labels.items()}
        }

        if auto_cat_cols:
            st.caption(f"⭐ = {len(auto_cat_cols)} auto-detected columns")
        else:
            st.info("No columns auto-detected — all columns listed below.")

        selected_labels = st.multiselect(
            "Select category columns:",
            options=all_options
        )
        selected_cat_cols = [reverse_map[l] for l in selected_labels]

        if selected_cat_cols:
            for cat_col in selected_cat_cols:
                n = df[cat_col].nunique()
                if n > 50:
                    st.warning(f"⚠️ **{cat_col}** has {n} unique values — top 10 shown in charts.")
                elif n < 2:
                    st.error(f"❌ **{cat_col}** has only {n} unique value.")

            st.markdown("**Category previews:**")
            num = len(selected_cat_cols)
            for row in range(0, num, 3):
                batch = selected_cat_cols[row:row+3]
                cols = st.columns(len(batch))
                for i, cat_col in enumerate(batch):
                    with cols[i]:
                        st.markdown(f"**{cat_col}**")
                        vc = df[cat_col].value_counts().head(6).reset_index()
                        vc.columns = [cat_col, 'Count']
                        st.dataframe(vc, use_container_width=True, hide_index=True)

    date_col = detect_date_column(df, selected_col)
    if date_col:
        st.info(f"📅 Date column detected: **{date_col}** — trend chart will be included.")

    # ── Step 4: Processing Options ─────────────────────────────────
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <span class="step-badge">Step 4</span>
            <p class="step-title">Processing Options</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(df) > 10000:
        st.warning(f"Your file has **{len(df):,} rows**.")
        process_option = st.radio(
            "How to proceed:",
            options=["Process all rows", "Process a sample", "Set a custom limit"],
            horizontal=True
        )
        if process_option == "Process a sample":
            sample_size = st.select_slider(
                "Sample size:",
                options=[5000, 10000, 25000, 50000, 100000],
                value=10000
            )
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        elif process_option == "Set a custom limit":
            custom_limit = st.number_input(
                "Max rows:",
                min_value=1000, max_value=len(df),
                value=min(50000, len(df)), step=1000
            )
            df = df.head(int(custom_limit))

    est = len(df) / 800
    st.caption(
        f"⏱️ {len(df):,} rows · "
        f"Estimated: ~{est:.0f}s" if est < 60
        else f"⏱️ {len(df):,} rows · Estimated: ~{est/60:.1f} min"
    )

    # ── Step 5: Run ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Analyse Sentiment", use_container_width=True)

    if run:
        result_df = analyse_reviews(df, selected_col)
        st.session_state['result_df'] = result_df
        st.session_state['selected_col'] = selected_col
        st.session_state['selected_cat_cols'] = selected_cat_cols
        st.session_state['date_col'] = date_col
        st.success("✅ Analysis complete — scroll down for results.")

# ── Results ────────────────────────────────────────────────────────
if 'result_df' in st.session_state:
    result_df = st.session_state['result_df']
    selected_cat_cols = st.session_state.get('selected_cat_cols', [])
    date_col = st.session_state.get('date_col')

    positive_df = result_df[result_df['sentiment'] == 'Positive']
    negative_df = result_df[result_df['sentiment'] == 'Negative']
    total = len(result_df)
    pos_count = len(positive_df)
    neg_count = len(negative_df)
    avg_conf = result_df['confidence'].mean()
    avg_len = result_df['_review_length'].mean()
    high_conf = (result_df['confidence'] >= 0.9).sum()

    st.divider()

    # ── Metric Cards ───────────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Sentiment Dashboard</p>',
               unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card metric-neutral">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Reviews</div>
        </div>
        <div class="metric-card metric-positive">
            <div class="metric-value">{pos_count:,}</div>
            <div class="metric-label">Positive · {pos_count/total:.0%}</div>
        </div>
        <div class="metric-card metric-negative">
            <div class="metric-value">{neg_count:,}</div>
            <div class="metric-label">Negative · {neg_count/total:.0%}</div>
        </div>
        <div class="metric-card metric-purple">
            <div class="metric-value">{avg_conf:.0%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        <div class="metric-card metric-neutral">
            <div class="metric-value">{avg_len:.0f}</div>
            <div class="metric-label">Avg Review Length</div>
        </div>
        <div class="metric-card metric-purple">
            <div class="metric-value">{high_conf/total:.0%}</div>
            <div class="metric-label">High Confidence (≥90%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview Charts ────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Overview</p>',
               unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(plot_donut(pos_count, neg_count))
        plt.close()
    with c2:
        st.pyplot(plot_confidence_distribution(result_df))
        plt.close()
    with c3:
        st.pyplot(plot_review_length_vs_confidence(result_df))
        plt.close()

    # ── Trend ──────────────────────────────────────────────────────
    if date_col:
        st.markdown('<p class="section-header">📅 Sentiment Trend</p>',
                   unsafe_allow_html=True)
        fig = plot_sentiment_trend(result_df, date_col)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough time data to generate a trend chart.")

    # ── Category Analysis ──────────────────────────────────────────
    if selected_cat_cols:
        st.markdown('<p class="section-header">🗂️ Category Analysis</p>',
                   unsafe_allow_html=True)

        for cat_col in selected_cat_cols:
            st.markdown(f"#### {cat_col}")
            left, right = st.columns([2, 1])
            with left:
                fig, counts = plot_category_sentiment(result_df, cat_col)
                st.pyplot(fig)
                plt.close()
            with right:
                fig = plot_category_confidence_box(result_df, cat_col)
                st.pyplot(fig)
                plt.close()

            best = counts['Positive %'].idxmax()
            worst = counts['Positive %'].idxmin()
            i1, i2 = st.columns(2)
            i1.markdown(
                f'<div class="insight-positive">😊 <strong>Best:</strong> '
                f'{best} — {counts.loc[best, "Positive %"]}% positive</div>',
                unsafe_allow_html=True
            )
            i2.markdown(
                f'<div class="insight-negative">😞 <strong>Needs attention:</strong> '
                f'{worst} — {counts.loc[worst, "Negative %"]}% negative</div>',
                unsafe_allow_html=True
            )
            with st.expander(f"📋 Full {cat_col} table"):
                st.dataframe(
                    counts[['Positive','Negative','Total',
                            'Positive %','Negative %']].reset_index(),
                    use_container_width=True, hide_index=True
                )

        if len(selected_cat_cols) >= 2:
            st.markdown(
                f'<p class="section-header">'
                f'🔥 Heatmap: {selected_cat_cols[0]} × {selected_cat_cols[1]}'
                f'</p>',
                unsafe_allow_html=True
            )
            st.caption("Each cell = positive sentiment % for that combination. Red = negative, green = positive.")
            fig = plot_heatmap(result_df, selected_cat_cols[0], selected_cat_cols[1])
            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Not enough overlapping data for a heatmap.")

    # ── Keywords ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔑 Top Keywords</p>',
               unsafe_allow_html=True)
    kw1, kw2 = st.columns(2)
    with kw1:
        fig = plot_keyword_bar(
            get_top_keywords(positive_df['_review_text']),
            '😊 Positive Keywords', POS_COLOR
        )
        if fig:
            st.pyplot(fig)
            plt.close()
    with kw2:
        fig = plot_keyword_bar(
            get_top_keywords(negative_df['_review_text']),
            '😞 Negative Keywords', NEG_COLOR
        )
        if fig:
            st.pyplot(fig)
            plt.close()

    # ── Filter & Explore ───────────────────────────────────────────
    st.markdown('<p class="section-header">🔎 Filter & Explore</p>',
               unsafe_allow_html=True)

    n_filters = 1 + len(selected_cat_cols)
    fcols = st.columns(n_filters)
    with fcols[0]:
        sentiment_filter = st.radio(
            "Sentiment:", ["All", "Positive Only", "Negative Only"],
            horizontal=True
        )
    cat_filters = {}
    for i, cat_col in enumerate(selected_cat_cols):
        with fcols[i+1]:
            opts = ["All"] + sorted(
                result_df[cat_col].dropna().unique().tolist()
            )
            cat_filters[cat_col] = st.selectbox(f"{cat_col}:", opts)

    display_df = result_df.copy()
    if sentiment_filter == "Positive Only":
        display_df = display_df[display_df['sentiment'] == 'Positive']
    elif sentiment_filter == "Negative Only":
        display_df = display_df[display_df['sentiment'] == 'Negative']
    for col, val in cat_filters.items():
        if val != "All":
            display_df = display_df[display_df[col] == val]

    st.caption(f"Showing **{len(display_df):,}** of {total:,} reviews")

    dcols = ['_review_text', 'sentiment', 'confidence', '_review_length']
    for c in selected_cat_cols:
        if c not in dcols:
            dcols.insert(1, c)

    st.dataframe(
        display_df[dcols].rename(columns={
            '_review_text': 'Review',
            'sentiment': 'Sentiment',
            'confidence': 'Confidence',
            '_review_length': 'Words'
        }),
        use_container_width=True, hide_index=True
    )

    # ── Download ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">⬇️ Download Results</p>',
               unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "📥 All Results",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_all.csv", mime="text/csv",
            use_container_width=True
        )
    with d2:
        st.download_button(
            "😊 Positive Reviews",
            data=positive_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_positive.csv", mime="text/csv",
            use_container_width=True
        )
    with d3:
        st.download_button(
            "😞 Negative Reviews",
            data=negative_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_negative.csv", mime="text/csv",
            use_container_width=True
        )

    active = sentiment_filter != "All" or any(
        v != "All" for v in cat_filters.values()
    )
    if active and len(display_df) != total:
        st.download_button(
            f"📂 Download Current View ({len(display_df):,} rows)",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_filtered.csv", mime="text/csv",
            use_container_width=True
        )

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#475569; font-size:0.8rem; padding:1rem 0;'>
        Trained on 500k+ Amazon Fine Food Reviews · 
        Logistic Regression + TF-IDF · 
        94% Accuracy · 
        <a href="https://github.com/danielakbank/review-sentiment-analyser" 
           style="color:#a78bfa; text-decoration:none;">View on GitHub ↗</a>
    </div>
    """, unsafe_allow_html=True)