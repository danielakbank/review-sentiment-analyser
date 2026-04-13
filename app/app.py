import streamlit as st
import pandas as pd
import joblib
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.patches import Patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import clean_text

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Sentiment Analyser",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e2e3e;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    model = joblib.load(os.path.join(base, 'logistic_model.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'tfidf_vectorizer.pkl'))
    return model, vectorizer

model, vectorizer = load_models()

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

POS_COLOR = '#2ecc71'
NEG_COLOR = '#e74c3c'
CONF_COLOR = '#3498db'

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
        if any(word in col_lower for word in ['date', 'time', 'year', 'month']):
            return col
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(10)
            try:
                pd.to_datetime(sample)
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
        return False, f"Average length is only {avg_len:.0f} characters — too short for reviews."
    if avg_words < 3:
        return False, f"Average word count is {avg_words:.1f} — looks like a label or ID column."
    if unique_ratio > 0.9 and sentiment_overlap < 0.05:
        return False, "Very high uniqueness and few opinion words — may be a title or ID column."
    return True, "Column looks good."

def analyse_reviews(df, column, batch_size=1000):
    reviews = df[column].fillna('').astype(str).tolist()
    predictions, confidences = [], []
    progress = st.progress(0, text="Analysing reviews...")
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
            text=f"Analysing... {min(i+batch_size, total):,}/{total:,}"
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

def plot_donut(pos_count, neg_count):
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [pos_count, neg_count]
    colors = [POS_COLOR, NEG_COLOR]
    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight('bold')
        at.set_color('white')
    ax.legend(
        [f'Positive ({pos_count:,})', f'Negative ({neg_count:,})'],
        loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2
    )
    ax.set_title('Overall Sentiment Split', fontsize=14,
                fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def plot_confidence_distribution(result_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    pos_conf = result_df[result_df['sentiment']=='Positive']['confidence']
    neg_conf = result_df[result_df['sentiment']=='Negative']['confidence']
    ax.hist(pos_conf, bins=20, alpha=0.7, color=POS_COLOR,
            label=f'Positive (n={len(pos_conf):,})', edgecolor='white')
    ax.hist(neg_conf, bins=20, alpha=0.7, color=NEG_COLOR,
            label=f'Negative (n={len(neg_conf):,})', edgecolor='white')
    ax.axvline(x=pos_conf.mean(), color='darkgreen', linestyle='--',
               alpha=0.8, linewidth=1.5, label=f'Pos mean: {pos_conf.mean():.2f}')
    ax.axvline(x=neg_conf.mean(), color='darkred', linestyle='--',
               alpha=0.8, linewidth=1.5, label=f'Neg mean: {neg_conf.mean():.2f}')
    ax.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def plot_review_length_vs_confidence(result_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sample = result_df.sample(min(2000, len(result_df)), random_state=42)
    colors = sample['sentiment'].map({'Positive': POS_COLOR, 'Negative': NEG_COLOR})
    ax.scatter(
        sample['_review_length'], sample['confidence'],
        c=colors, alpha=0.4, s=15
    )
    ax.set_title('Review Length vs Confidence', fontsize=13, fontweight='bold')
    ax.set_xlabel('Review Length (words)')
    ax.set_ylabel('Confidence Score')
    ax.set_xlim(0, min(500, sample['_review_length'].quantile(0.99)))
    ax.legend(handles=[
        Patch(color=POS_COLOR, label='Positive'),
        Patch(color=NEG_COLOR, label='Negative')
    ])
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
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.7)))
    y = range(len(counts))
    ax.barh(y, counts['Positive %'], color=POS_COLOR, label='Positive')
    ax.barh(y, counts['Negative %'], left=counts['Positive %'],
            color=NEG_COLOR, label='Negative')
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=10)
    ax.set_xlabel('Percentage (%)')
    ax.set_title(f'Sentiment by {cat_col}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    for i, (pos_pct, neg_pct) in enumerate(
        zip(counts['Positive %'], counts['Negative %'])
    ):
        if pos_pct > 8:
            ax.text(pos_pct/2, i, f'{pos_pct:.0f}%', va='center',
                   ha='center', color='white', fontsize=9, fontweight='bold')
        if neg_pct > 8:
            ax.text(pos_pct + neg_pct/2, i, f'{neg_pct:.0f}%',
                   va='center', ha='center', color='white',
                   fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig, counts

def plot_heatmap(result_df, cat_col1, cat_col2, top_n=8):
    top_cat1 = result_df[cat_col1].value_counts().head(top_n).index
    top_cat2 = result_df[cat_col2].value_counts().head(top_n).index
    filtered = result_df[
        result_df[cat_col1].isin(top_cat1) &
        result_df[cat_col2].isin(top_cat2)
    ]
    pivot = filtered.groupby([cat_col1, cat_col2]).apply(
        lambda x: round((x['sentiment'] == 'Positive').mean() * 100, 1)
    ).unstack(fill_value=np.nan)
    if pivot.empty:
        return None
    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.2),
                 max(5, len(pivot.index) * 0.8))
    )
    im = ax.imshow(pivot.values, cmap='RdYlGn',
                   aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(
        f'Positive Sentiment % — {cat_col1} × {cat_col2}',
        fontsize=13, fontweight='bold'
    )
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                       fontsize=8, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Positive %')
    plt.tight_layout()
    return fig

def plot_keyword_bar(keywords, title, color):
    if not keywords:
        return None
    words, counts = zip(*keywords)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(words)), counts, color=color,
                   edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9)
    plt.tight_layout()
    return fig

def plot_sentiment_trend(result_df, date_col):
    try:
        df_trend = result_df.copy()
        df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
        df_trend = df_trend.dropna(subset=[date_col])
        if len(df_trend) < 10:
            return None
        df_trend['month'] = df_trend[date_col].dt.to_period('M')
        trend = df_trend.groupby(
            ['month', 'sentiment']
        ).size().unstack(fill_value=0)
        for col in ['Positive', 'Negative']:
            if col not in trend.columns:
                trend[col] = 0
        trend['Total'] = trend['Positive'] + trend['Negative']
        trend['Positive %'] = (
            trend['Positive'] / trend['Total'] * 100
        ).round(1)
        if len(trend) < 2:
            return None
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        x = range(len(trend))
        # Top: sentiment % trend
        ax1.plot(x, trend['Positive %'], color=POS_COLOR,
                linewidth=2.5, marker='o', markersize=4, label='Positive %')
        ax1.fill_between(x, trend['Positive %'], alpha=0.15, color=POS_COLOR)
        ax1.axhline(y=50, color='grey', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Positive %')
        ax1.set_ylim(0, 100)
        ax1.set_title('Sentiment Trend Over Time',
                     fontsize=13, fontweight='bold')
        ax1.legend()
        # Bottom: volume
        ax2.bar(x, trend['Positive'], color=POS_COLOR,
                alpha=0.7, label='Positive')
        ax2.bar(x, trend['Negative'], bottom=trend['Positive'],
                color=NEG_COLOR, alpha=0.7, label='Negative')
        ax2.set_ylabel('Review Count')
        ax2.legend()
        step = max(1, len(trend) // 12)
        ax2.set_xticks(list(x)[::step])
        ax2.set_xticklabels(
            trend.index.astype(str)[::step],
            rotation=45, ha='right'
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
    fig, ax = plt.subplots(figsize=(10, max(4, len(top_cats) * 0.6)))
    bp = ax.boxplot(grouped, vert=False, patch_artist=True,
                   labels=top_cats)
    for patch in bp['boxes']:
        patch.set_facecolor(CONF_COLOR)
        patch.set_alpha(0.7)
    ax.set_xlabel('Confidence Score')
    ax.set_title(f'Prediction Confidence by {cat_col}',
                fontsize=13, fontweight='bold')
    ax.axvline(x=0.5, color='grey', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# ─── UI ────────────────────────────────────────────────────────────

st.title("🛍️ Review Sentiment Analyser")
st.markdown(
    "Upload any product review CSV for instant sentiment analysis, "
    "category breakdowns, and business insights."
)
st.divider()

# ── Step 1: Upload ─────────────────────────────────────────────────
st.subheader("📁 Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader(
    "Choose a CSV file (up to 200MB)", type=['csv']
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(
            f"✅ File loaded — {len(df):,} rows, {len(df.columns)} columns."
        )
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ── Step 2: Review Column ──────────────────────────────────────
    st.divider()
    st.subheader("🔍 Step 2: Select Your Review Column")

    scored_cols = detect_review_columns(df)
    all_text_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()
    score_map = {col: score for col, score in scored_cols}
    auto_detected_review_cols = [col for col, _ in scored_cols]

    # Show scoring table
    if scored_cols:
        score_df = pd.DataFrame(scored_cols, columns=['Column', 'Review Score'])
        score_df['Assessment'] = score_df['Review Score'].apply(
            lambda x: '✅ Strong match' if x > 10
            else '⚠️ Possible match' if x > 5
            else '❌ Unlikely'
        )
        st.markdown("**Auto-detected columns ranked by review likelihood:**")
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    # Always show full column selector
    st.markdown("Select your review column — auto-detected options are shown first:")
    other_cols = [c for c in all_cols if c not in auto_detected_review_cols]
    grouped_options = auto_detected_review_cols + other_cols

    selected_col = st.selectbox(
        "Review column:",
        options=grouped_options,
        help="Columns at the top were auto-detected. Scroll down to see all columns."
    )

    if selected_col:
        is_valid, reason = validate_review_column(df[selected_col])

        st.markdown("**Sample entries from selected column:**")
        for i, sample in enumerate(
            df[selected_col].dropna().head(3).tolist(), 1
        ):
            st.info(f"**{i}.** {str(sample)[:300]}")

        if not is_valid:
            st.warning(f"⚠️ {reason}")
            st.markdown(
                "This column may not contain reviews. "
                "You can still proceed or choose a different column above."
            )
        else:
            st.success("✅ This looks like a review column.")

    # ── Step 3: Category Columns ───────────────────────────────────
    st.divider()
    st.subheader("🗂️ Step 3: Select Category Columns (Optional)")
    st.markdown(
        "Category columns let you compare sentiment across groups — "
        "e.g. product type, country, rating, brand. "
        "Select multiple to unlock heatmap cross-analysis."
    )

    use_category = st.checkbox("Add category analysis", value=True)
    selected_cat_cols = []

    if use_category:
        cat_candidates = detect_categorical_columns(df, selected_col)
        auto_cat_cols = [col for col, _, _ in cat_candidates]
        auto_cat_labels = {
            col: f"⭐ {col} ({n} unique — {kind})"
            for col, n, kind in cat_candidates
        }

        # All other columns not the review col and not auto-detected
        remaining_cols = [
            c for c in df.columns
            if c != selected_col and c not in auto_cat_cols
        ]
        remaining_labels = {
            col: f"{col} ({df[col].nunique()} unique)"
            for col in remaining_cols
        }

        all_cat_options = (
            list(auto_cat_labels.values()) +
            list(remaining_labels.values())
        )
        reverse_map = {
            **{v: k for k, v in auto_cat_labels.items()},
            **{v: k for k, v in remaining_labels.items()}
        }

        if auto_cat_cols:
            st.markdown(
                f"⭐ = auto-detected suitable columns ({len(auto_cat_cols)} found). "
                "All other columns are also available below."
            )
        else:
            st.info(
                "No columns were auto-detected as categorical. "
                "All columns are listed below — select any that make sense."
            )

        selected_cat_labels = st.multiselect(
            "Select category columns:",
            options=all_cat_options,
            help="Auto-detected columns marked ⭐ are recommended."
        )
        selected_cat_cols = [reverse_map[label] for label in selected_cat_labels]

        if selected_cat_cols:
            for cat_col in selected_cat_cols:
                n_unique = df[cat_col].nunique()
                if n_unique > 50:
                    st.warning(
                        f"⚠️ **{cat_col}** has {n_unique} unique values — "
                        f"only the top 10 by volume will appear in charts."
                    )
                elif n_unique < 2:
                    st.error(
                        f"❌ **{cat_col}** has only {n_unique} unique value — "
                        f"not useful for comparison."
                    )

            st.markdown("**Category previews:**")
            num_cats = len(selected_cat_cols)
            cols_per_row = min(num_cats, 3)
            for row_start in range(0, num_cats, cols_per_row):
                row_cats = selected_cat_cols[row_start:row_start+cols_per_row]
                preview_cols = st.columns(len(row_cats))
                for i, cat_col in enumerate(row_cats):
                    with preview_cols[i]:
                        st.markdown(f"**{cat_col}**")
                        cat_counts = (
                            df[cat_col].value_counts()
                            .head(8).reset_index()
                        )
                        cat_counts.columns = [cat_col, 'Count']
                        st.dataframe(
                            cat_counts,
                            use_container_width=True,
                            hide_index=True
                        )

    # ── Date Column Detection ──────────────────────────────────────
    date_col = detect_date_column(df, selected_col)
    if date_col:
        st.info(
            f"📅 Date column detected: **{date_col}** — "
            f"a sentiment trend chart will be included."
        )

    # ── Step 4: Processing Options ─────────────────────────────────
    st.divider()
    st.subheader("⚙️ Step 4: Processing Options")

    if len(df) > 10000:
        st.warning(f"Your file has {len(df):,} rows.")
        process_option = st.radio(
            "How would you like to proceed?",
            options=["Process all rows", "Process a sample", "Set a custom limit"]
        )
        if process_option == "Process a sample":
            sample_size = st.select_slider(
                "Sample size:",
                options=[5000, 10000, 25000, 50000, 100000],
                value=10000
            )
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            st.info(f"Will process {len(df):,} rows.")
        elif process_option == "Set a custom limit":
            custom_limit = st.number_input(
                "Max rows:",
                min_value=1000,
                max_value=len(df),
                value=min(50000, len(df)),
                step=1000
            )
            df = df.head(int(custom_limit))
            st.info(f"Will process {len(df):,} rows.")
        else:
            st.info(f"Will process all {len(df):,} rows.")
    else:
        st.info(f"Will process all {len(df):,} rows.")

    est_secs = len(df) / 800
    st.caption(
        f"⏱️ Estimated time: ~{est_secs:.0f} seconds"
        if est_secs < 60
        else f"⏱️ Estimated time: ~{est_secs/60:.1f} minutes"
    )

    # ── Step 5: Run ────────────────────────────────────────────────
    st.divider()
    st.subheader("🚀 Step 5: Run Analysis")

    if st.button("Analyse Reviews", use_container_width=True):
        with st.spinner("Preparing data..."):
            result_df = analyse_reviews(df, selected_col)
        st.session_state['result_df'] = result_df
        st.session_state['selected_col'] = selected_col
        st.session_state['selected_cat_cols'] = selected_cat_cols
        st.session_state['date_col'] = date_col
        st.success("✅ Analysis complete! Scroll down to see results.")

# ── Results ────────────────────────────────────────────────────────
if 'result_df' in st.session_state:
    result_df = st.session_state['result_df']
    selected_col = st.session_state['selected_col']
    selected_cat_cols = st.session_state.get('selected_cat_cols', [])
    date_col = st.session_state.get('date_col')

    positive_df = result_df[result_df['sentiment'] == 'Positive']
    negative_df = result_df[result_df['sentiment'] == 'Negative']
    total = len(result_df)
    pos_count = len(positive_df)
    neg_count = len(negative_df)
    avg_conf = result_df['confidence'].mean()
    avg_len = result_df['_review_length'].mean()

    # ── Dashboard ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Sentiment Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("✅ Positive", f"{pos_count:,}", f"{pos_count/total:.0%}")
    c3.metric("❌ Negative", f"{neg_count:,}", f"{neg_count/total:.0%}")
    c4.metric("Avg Confidence", f"{avg_conf:.0%}")
    c5.metric("Avg Review Length", f"{avg_len:.0f} words")

    # ── Overview Charts ────────────────────────────────────────────
    st.markdown("#### Overview")
    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        st.pyplot(plot_donut(pos_count, neg_count))
        plt.close()
    with ov2:
        st.pyplot(plot_confidence_distribution(result_df))
        plt.close()
    with ov3:
        st.pyplot(plot_review_length_vs_confidence(result_df))
        plt.close()

    # ── Trend Chart ────────────────────────────────────────────────
    if date_col:
        st.markdown("#### 📅 Sentiment Trend Over Time")
        trend_fig = plot_sentiment_trend(result_df, date_col)
        if trend_fig:
            st.pyplot(trend_fig)
            plt.close()
        else:
            st.info("Not enough time data to plot a trend.")

    # ── Category Analysis ──────────────────────────────────────────
    if selected_cat_cols:
        st.divider()
        st.subheader("🗂️ Category Analysis")

        for cat_col in selected_cat_cols:
            st.markdown(f"#### Sentiment by **{cat_col}**")
            c_left, c_right = st.columns([2, 1])
            with c_left:
                fig, counts = plot_category_sentiment(result_df, cat_col)
                st.pyplot(fig)
                plt.close()
            with c_right:
                st.markdown("**Confidence by category:**")
                box_fig = plot_category_confidence_box(result_df, cat_col)
                st.pyplot(box_fig)
                plt.close()

            with st.expander(f"📋 {cat_col} summary table"):
                summary = counts[[
                    'Positive', 'Negative', 'Total',
                    'Positive %', 'Negative %'
                ]].reset_index()
                st.dataframe(
                    summary, use_container_width=True, hide_index=True
                )

            best = counts['Positive %'].idxmax()
            worst = counts['Positive %'].idxmin()
            ins1, ins2 = st.columns(2)
            ins1.success(
                f"😊 **Best:** {best} — "
                f"{counts.loc[best, 'Positive %']}% positive"
            )
            ins2.error(
                f"😞 **Needs attention:** {worst} — "
                f"{counts.loc[worst, 'Negative %']}% negative"
            )

        # ── Heatmap ────────────────────────────────────────────────
        if len(selected_cat_cols) >= 2:
            st.markdown(
                f"#### 🔥 Heatmap: "
                f"{selected_cat_cols[0]} × {selected_cat_cols[1]}"
            )
            st.markdown(
                "Each cell shows positive sentiment % for that combination. "
                "Red = more negative, green = more positive."
            )
            heatmap_fig = plot_heatmap(
                result_df, selected_cat_cols[0], selected_cat_cols[1]
            )
            if heatmap_fig:
                st.pyplot(heatmap_fig)
                plt.close()
            else:
                st.info(
                    "Not enough overlapping data between these two "
                    "columns to generate a heatmap."
                )

    # ── Keywords ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🔑 Top Keywords by Sentiment")
    kw1, kw2 = st.columns(2)
    with kw1:
        pos_kw = get_top_keywords(positive_df['_review_text'])
        fig = plot_keyword_bar(pos_kw, '😊 Positive Keywords', POS_COLOR)
        if fig:
            st.pyplot(fig)
            plt.close()
    with kw2:
        neg_kw = get_top_keywords(negative_df['_review_text'])
        fig = plot_keyword_bar(neg_kw, '😞 Negative Keywords', NEG_COLOR)
        if fig:
            st.pyplot(fig)
            plt.close()

    # ── Filter & Explore ───────────────────────────────────────────
    st.divider()
    st.subheader("🔎 Filter & Explore Reviews")

    num_filters = 1 + len(selected_cat_cols)
    filter_cols = st.columns(num_filters)

    with filter_cols[0]:
        sentiment_filter = st.radio(
            "Sentiment:",
            options=["All", "Positive Only", "Negative Only"],
            horizontal=True
        )

    cat_filters = {}
    for i, cat_col in enumerate(selected_cat_cols):
        with filter_cols[i + 1]:
            options = ["All"] + sorted(
                result_df[cat_col].dropna().unique().tolist()
            )
            cat_filters[cat_col] = st.selectbox(
                f"{cat_col}:", options=options
            )

    # Apply filters
    display_df = result_df.copy()
    if sentiment_filter == "Positive Only":
        display_df = display_df[display_df['sentiment'] == 'Positive']
    elif sentiment_filter == "Negative Only":
        display_df = display_df[display_df['sentiment'] == 'Negative']
    for cat_col, val in cat_filters.items():
        if val != "All":
            display_df = display_df[display_df[cat_col] == val]

    st.markdown(f"Showing **{len(display_df):,}** reviews")

    display_cols = ['_review_text', 'sentiment', 'confidence', '_review_length']
    for cat_col in selected_cat_cols:
        if cat_col not in display_cols:
            display_cols.insert(1, cat_col)

    st.dataframe(
        display_df[display_cols].rename(columns={
            '_review_text': 'Review',
            'sentiment': 'Sentiment',
            'confidence': 'Confidence',
            '_review_length': 'Word Count'
        }),
        use_container_width=True,
        hide_index=True
    )

    # ── Download ───────────────────────────────────────────────────
    st.divider()
    st.subheader("⬇️ Download Results")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "📥 All Results",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_all.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl2:
        st.download_button(
            "😊 Positive Reviews",
            data=positive_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_positive.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl3:
        st.download_button(
            "😞 Negative Reviews",
            data=negative_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_negative.csv",
            mime="text/csv",
            use_container_width=True
        )

    active_filters = (
        sentiment_filter != "All" or
        any(v != "All" for v in cat_filters.values())
    )
    if active_filters and len(display_df) != total:
        filter_name = sentiment_filter.replace(
            " Only", ""
        ).replace(" ", "_").lower()
        st.download_button(
            f"📂 Download current view ({len(display_df):,} rows)",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name=f"sentiment_filtered_{filter_name}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:grey;font-size:0.85em;'>"
        "Trained on Amazon Fine Food Reviews | "
        "Logistic Regression + TF-IDF | 94% Accuracy"
        "</div>",
        unsafe_allow_html=True
    )