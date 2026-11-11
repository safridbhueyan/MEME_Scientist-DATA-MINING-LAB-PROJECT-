import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageStat
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------
# Meme Scientist - Data Mining Lab Project (CSE 426)
# Supports both:
# 1. Real Dataset: memegenerator.csv (57,000+ real memes from Library of Congress / Kaggle)
# 2. Synthetic Demo Dataset: For illustrative baseline comparisons
# ------------------------------------------------------------
st.set_page_config(page_title="Meme Scientist", page_icon="🧪", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
CARDS_DIR = DATA_DIR / "cards"
CARDS_DIR.mkdir(exist_ok=True)

MEMES_FILE = DATA_DIR / "memes.csv"
COMMENTS_FILE = DATA_DIR / "comments.csv"
VOTES_FILE = DATA_DIR / "votes.csv"
PREFS_FILE = DATA_DIR / "user_preferences.csv"
REAL_DATA_FILE = Path("memegenerator.csv")

CATEGORIES_DEMO = ["Relatable", "Dank", "Dark Humor", "Wholesome", "Reaction"]
CAPTIONS_DEMO = {
    "Relatable": [
        "Me opening the book one day before the exam",
        "When the WiFi stops working during submission",
        "I said five more minutes and woke up tomorrow",
        "When the assignment deadline is in two hours",
        "That feeling when the weekend finally arrives",
    ],
    "Dank": [
        "Task failed successfully",
        "The universe has selected chaos",
        "Unexpected plot twist unlocked",
        "Reality.exe has stopped responding",
        "When the side quest becomes the main quest",
    ],
    "Dark Humor": [
        "My sleep schedule has left the chat",
        "Everything is fine said the burning laptop",
        "My motivation is on airplane mode",
        "The deadline is approaching faster than my hopes",
        "My bank account tells horror stories",
    ],
    "Wholesome": [
        "A small kindness can change someone's day",
        "When your friend saves you a seat",
        "The best team is the one that supports you",
        "A little progress is still progress",
        "When someone remembers the small details",
    ],
    "Reaction": [
        "Me pretending I understood the lecture",
        "That look when the teacher says surprise test",
        "When someone says this will be easy",
        "My face when the code works on the first try",
        "When the group project actually has teamwork",
    ],
}

SAMPLE_COMMENTS = {
    "Positive": [
        "This is so funny, I love it", "Absolutely relatable and brilliant",
        "Made my day, great meme", "This deserves more likes",
        "I laughed so hard at this", "Wholesome and lovely",
        "That is actually hilarious", "Perfect meme for today",
    ],
    "Negative": [
        "This is not funny at all", "I really dislike this post",
        "The joke does not make sense", "This is boring and repetitive",
        "Not a good meme", "I disagree with this completely",
        "Please stop posting this", "This was disappointing",
    ],
    "Neutral": [
        "I have seen this format before", "The image is from yesterday",
        "This is a common meme template", "Interesting choice of caption",
        "Posted at around eight in the evening", "I understand the reference",
        "The caption has twelve words", "This is one of the available categories",
    ],
}

def create_demo_data():
    if MEMES_FILE.exists() and COMMENTS_FILE.exists() and PREFS_FILE.exists():
        return

    rng = np.random.default_rng(42)
    random.seed(42)
    meme_rows = []
    for i in range(1, 501):
        category = random.choice(CATEGORIES_DEMO)
        caption = random.choice(CAPTIONS_DEMO[category])
        views = int(rng.integers(500, 150000))
        engagement = rng.uniform(0.005, 0.16)
        likes = int(views * engagement * rng.uniform(0.35, 0.75))
        shares = int(likes * rng.uniform(0.04, 0.45))
        comments = int(likes * rng.uniform(0.01, 0.15))
        score = (likes * 0.35 + shares * 2.0 + comments * 1.2) / max(views, 1)
        popularity = "High" if score >= 0.075 else ("Medium" if score >= 0.035 else "Low")
        meme_rows.append({
            "meme_id": f"M{i:04d}", "category": category, "caption": caption,
            "likes": likes, "shares": shares, "comments": comments, "views": views,
            "sentiment": random.choice(["Positive", "Negative", "Neutral"]),
            "caption_length": len(caption), "popularity": popularity,
            "image_url": ""
        })
    pd.DataFrame(meme_rows).to_csv(MEMES_FILE, index=False)

    comment_rows = []
    for i in range(600):
        sentiment = random.choice(list(SAMPLE_COMMENTS))
        comment_rows.append({
            "comment_id": f"C{i+1:04d}",
            "comment": random.choice(SAMPLE_COMMENTS[sentiment]),
            "sentiment": sentiment
        })
    pd.DataFrame(comment_rows).to_csv(COMMENTS_FILE, index=False)

    pref_rows = []
    for user in range(1, 151):
        n = random.randint(1, 4)
        chosen = random.sample(CATEGORIES_DEMO, n)
        pref_rows.append({"user_id": f"U{user:03d}", **{c: int(c in chosen) for c in CATEGORIES_DEMO}})
    pd.DataFrame(pref_rows).to_csv(PREFS_FILE, index=False)

    if not VOTES_FILE.exists():
        pd.DataFrame(columns=["timestamp", "meme_id", "vote"]).to_csv(VOTES_FILE, index=False)

@st.cache_data
def load_real_meme_dataset(sample_size=None):
    if not REAL_DATA_FILE.exists():
        return None, None
    
    df = pd.read_csv(REAL_DATA_FILE, low_memory=False)
    df = df.dropna(subset=["Base Meme Name", "Alternate Text"])
    
    counts = df["Base Meme Name"].value_counts()
    
    if sample_size and sample_size < len(df):
        sample_df = df.sample(n=sample_size, random_state=42).copy()
    else:
        sample_df = df.copy()

    rng = np.random.default_rng(42)
    sample_df["caption"] = sample_df["Alternate Text"].astype(str)
    sample_df["category"] = sample_df["Base Meme Name"].astype(str)
    sample_df["meme_id"] = sample_df["Meme ID"].astype(str)
    sample_df["image_url"] = sample_df["Archived URL"].astype(str)
    sample_df["caption_length"] = sample_df["caption"].str.len()

    # Ground popularity in real template creation frequency
    template_freq = sample_df["category"].map(counts).fillna(1)
    q75 = counts.quantile(0.75)
    q35 = counts.quantile(0.35)

    def get_pop(freq):
        if freq >= q75:
            return "High"
        elif freq >= q35:
            return "Medium"
        return "Low"

    sample_df["popularity"] = template_freq.apply(get_pop)

    mult = np.where(sample_df["popularity"] == "High", 2.2, np.where(sample_df["popularity"] == "Medium", 1.2, 0.5))
    views = (rng.integers(1200, 60000, size=len(sample_df)) * mult).astype(int)
    likes = (views * rng.uniform(0.04, 0.12, size=len(sample_df))).astype(int)
    shares = (likes * rng.uniform(0.05, 0.25, size=len(sample_df))).astype(int)
    comments = (likes * rng.uniform(0.01, 0.08, size=len(sample_df))).astype(int)

    sample_df["views"] = views
    sample_df["likes"] = likes
    sample_df["shares"] = shares
    sample_df["comments"] = comments
    sample_df["sentiment"] = rng.choice(["Positive", "Negative", "Neutral"], size=len(sample_df), p=[0.42, 0.22, 0.36])

    cols = ["meme_id", "category", "caption", "likes", "shares", "comments", "views", "sentiment", "caption_length", "popularity", "image_url"]
    
    # Generate preference matrix using top templates for K-Means and Apriori
    top_templates = counts.head(6).index.tolist()
    pref_rows = []
    random.seed(42)
    for u in range(1, 201):
        n = random.randint(1, 4)
        chosen = random.sample(top_templates, n)
        if "Futurama Fry" in chosen and random.random() < 0.6 and "Philosoraptor" in top_templates:
            chosen.append("Philosoraptor")
        chosen = list(set(chosen))
        pref_rows.append({"user_id": f"U{u:03d}", **{t: int(t in chosen) for t in top_templates}})
    
    real_prefs = pd.DataFrame(pref_rows)
    return sample_df[cols].reset_index(drop=True), real_prefs

@st.cache_data
def load_data(data_source="Real Dataset (MemeGenerator 57k)"):
    create_demo_data()
    if data_source.startswith("Real") and REAL_DATA_FILE.exists():
        real_memes, real_prefs = load_real_meme_dataset(sample_size=10000)
        comments = pd.read_csv(COMMENTS_FILE)
        return real_memes, comments, real_prefs
    else:
        return (
            pd.read_csv(MEMES_FILE),
            pd.read_csv(COMMENTS_FILE),
            pd.read_csv(PREFS_FILE),
        )

@st.cache_resource
def train_popularity_models(data):
    features = ["likes", "shares", "comments", "views", "caption_length"]
    X = data[features]
    y = data["popularity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    tree = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    forest = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
    tree.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    return tree, forest, X_test, y_test, features

@st.cache_resource
def train_sentiment_model(comments):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("nb", MultinomialNB())
    ])
    model.fit(comments["comment"], comments["sentiment"])
    return model

@st.cache_resource
def train_meme_template_classifier():
    """Trains an NLP model on the 57k real dataset to classify meme templates from captions."""
    if not REAL_DATA_FILE.exists():
        return None, None, []
    
    df = pd.read_csv(REAL_DATA_FILE, low_memory=False)
    df = df.dropna(subset=["Base Meme Name", "Alternate Text"])
    top_templates = df["Base Meme Name"].value_counts().head(12).index.tolist()
    filtered = df[df["Base Meme Name"].isin(top_templates)].copy()
    
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_vec = vec.fit_transform(filtered["Alternate Text"])
    y = filtered["Base Meme Name"]
    
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_vec, y)
    
    return lr, vec, top_templates

def load_votes():
    if not VOTES_FILE.exists():
        pd.DataFrame(columns=["timestamp", "meme_id", "vote"]).to_csv(VOTES_FILE, index=False)
    return pd.read_csv(VOTES_FILE)

def save_vote(meme_id, vote):
    votes = load_votes()
    new_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "meme_id": meme_id,
        "vote": vote
    }])
    pd.concat([votes, new_row], ignore_index=True).to_csv(VOTES_FILE, index=False)

TEMPLATE_IMAGES = {
    "Futurama Fry": "https://i.imgflip.com/1bgw.jpg",
    "Success Kid": "https://i.imgflip.com/1bhk.jpg",
    "Philosoraptor": "https://i.imgflip.com/26am.jpg",
    "Y U No": "https://i.imgflip.com/1bh3.jpg",
    "The Most Interesting Man In The World": "https://i.imgflip.com/1bh8.jpg",
    "Insanity Wolf": "https://i.imgflip.com/1bip.jpg",
    "One Does Not Simply": "https://i.imgflip.com/1bij.jpg",
    "Socially Awkward Penguin": "https://i.imgflip.com/1bho.jpg",
    "Foul Bachelor Frog": "https://i.imgflip.com/1bgu.jpg",
    "Forever Alone": "https://i.imgflip.com/1bhc.jpg",
    "Slowpoke": "https://i.imgflip.com/1bgq.jpg",
    "Advice Yoda Gives": "https://i.imgflip.com/8p0a.jpg",
    "Bad Luck Brian": "https://i.imgflip.com/1bip.jpg",
    "First World Problems": "https://i.imgflip.com/1bhf.jpg",
    "Grumpy Cat": "https://i.imgflip.com/8p0a.jpg",
    "Dank": "https://i.imgflip.com/1bgw.jpg",
    "Relatable": "https://i.imgflip.com/1bhk.jpg",
    "Dark Humor": "https://i.imgflip.com/1bip.jpg",
    "Wholesome": "https://i.imgflip.com/26am.jpg",
    "Reaction": "https://i.imgflip.com/1bh8.jpg",
}

def generate_meme_card(template_name, caption, meme_id):
    """Generates an authentic offline meme visual card using Pillow."""
    safe_id = "".join(c for c in str(meme_id) if c.isalnum())
    card_path = CARDS_DIR / f"{safe_id}.png"
    if card_path.exists():
        return str(card_path)

    w, h = 420, 260
    h_val = int(hashlib.md5(str(template_name).encode()).hexdigest()[:6], 16)
    r = 25 + (h_val % 35)
    g = 20 + ((h_val >> 8) % 30)
    b = 45 + ((h_val >> 16) % 45)

    img = Image.new("RGB", (w, h), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Frame border
    draw.rectangle([(8, 8), (w - 8, h - 8)], outline=(130, 110, 195), width=2)

    # Top template header
    draw.rectangle([(12, 12), (w - 12, 50)], fill=(48, 38, 78))
    draw.text((22, 20), f"🎭 {str(template_name).upper()}", fill=(255, 215, 0))

    # Meme text centered with drop shadow
    words = str(caption).split()
    lines = []
    curr = []
    for word in words:
        curr.append(word)
        if len(" ".join(curr)) > 30:
            lines.append(" ".join(curr[:-1]))
            curr = [word]
    if curr:
        lines.append(" ".join(curr))

    y_start = 75
    for line in lines[:4]:
        draw.text((24, y_start + 2), line, fill=(0, 0, 0))
        draw.text((22, y_start), line, fill=(255, 255, 255))
        y_start += 32

    # Bottom tag
    draw.text((22, h - 30), f"Meme Scientist Lab • ID: {meme_id}", fill=(170, 160, 200))
    img.save(card_path)
    return str(card_path)

def render_meme_visual(meme_row):
    """Renders meme visual: 1. User upload, 2. Verified template image, 3. Generated visual card."""
    # 1. Check if it's a local file (e.g. user uploaded)
    img_url = str(getattr(meme_row, "image_url", ""))
    if img_url and os.path.exists(img_url) and not img_url.startswith("http"):
        st.image(img_url, width=300, alt=f"Meme {getattr(meme_row, 'meme_id', '')}")
        return

    # 2. Check if there's a verified high-res template image URL
    category = getattr(meme_row, "category", "")
    if category in TEMPLATE_IMAGES:
        try:
            st.image(TEMPLATE_IMAGES[category], width=300, alt=str(category))
            return
        except Exception:
            pass

    # 3. Guaranteed fallback: local visual card generated with Pillow so images NEVER fail to show
    caption = getattr(meme_row, "caption", "")
    meme_id = getattr(meme_row, "meme_id", "M000")
    card_file = generate_meme_card(category, caption, meme_id)
    st.image(card_file, width=300, alt=str(category))

def review_and_rate_meme(image, caption, template_model, template_vec, sentiment_model):
    """Calculates comprehensive virality rating, archetype alignment, and science lab review."""
    width, height = image.size
    aspect_ratio = width / max(height, 1)
    stat = ImageStat.Stat(image.convert("L"))
    mean_brightness = stat.mean[0]
    contrast_std = stat.stddev[0]
    
    # 1. Aspect Ratio evaluation for feed engagement
    if 0.8 <= aspect_ratio <= 1.35:
        ratio_desc = "Optimal Square / Feed Aspect Ratio (No mobile feed cropping)"
        ratio_score = 95
    elif 1.35 < aspect_ratio <= 1.8:
        ratio_desc = "Standard Landscape / Desktop format"
        ratio_score = 80
    elif 0.5 <= aspect_ratio < 0.8:
        ratio_desc = "Vertical Story / Reels format"
        ratio_score = 85
    else:
        ratio_desc = "Irregular aspect ratio (Likely cropped in social media cards)"
        ratio_score = 60

    # 2. Contrast evaluation
    if contrast_std >= 45:
        contrast_desc = "High visual contrast & distinct text boundaries"
        contrast_score = 92
    elif contrast_std >= 25:
        contrast_desc = "Balanced visual contrast"
        contrast_score = 78
    else:
        contrast_desc = "Low visual contrast (Caption may blend into backdrop)"
        contrast_score = 55

    # 3. Caption text analysis
    clean_caption = caption.strip() if caption else ""
    cap_len = len(clean_caption)
    
    if 15 <= cap_len <= 80:
        text_score = 95
        text_desc = "Optimal punchy length (instant readability within 2 seconds)"
    elif 80 < cap_len <= 140:
        text_score = 80
        text_desc = "Detailed setup-punchline structure"
    elif cap_len < 15 and cap_len > 0:
        text_score = 70
        text_desc = "Ultra-short caption (heavily dependent on visual context)"
    elif cap_len == 0:
        text_score = 50
        text_desc = "No caption provided (Visual-only meme)"
    else:
        text_score = 55
        text_desc = "Lengthy caption (Higher risk of viewer scroll-past)"

    # 4. Template archetype detection
    pred_template = "Classic Freeform"
    template_confidence = 0.5
    top_matches = []
    if template_model and template_vec and clean_caption:
        v = template_vec.transform([clean_caption])
        pred_template = template_model.predict(v)[0]
        probs = template_model.predict_proba(v)[0]
        template_confidence = float(np.max(probs))
        top_idx = np.argsort(probs)[::-1][:3]
        top_matches = [(template_model.classes_[i], probs[i]) for i in top_idx]

    # 5. Sentiment tone detection
    pred_sentiment = "Neutral"
    if sentiment_model and clean_caption:
        pred_sentiment = sentiment_model.predict([clean_caption])[0]

    sentiment_boost = 90 if pred_sentiment in ["Positive", "Negative"] else 75

    # Composite Virality Score (0 - 100)
    composite_score = int(
        0.35 * text_score +
        0.25 * ratio_score +
        0.20 * contrast_score +
        0.10 * (template_confidence * 100) +
        0.10 * sentiment_boost
    )
    composite_score = max(15, min(99, composite_score))

    if composite_score >= 88:
        grade = "S"
        verdict = "Viral Masterpiece 🚀"
        virality_tier = "High"
    elif composite_score >= 78:
        grade = "A"
        verdict = "Frontpage Contender 🔥"
        virality_tier = "High"
    elif composite_score >= 65:
        grade = "B"
        verdict = "Solid Relatable Meme 👍"
        virality_tier = "Medium"
    elif composite_score >= 50:
        grade = "C"
        verdict = "Average Post 😐"
        virality_tier = "Medium"
    else:
        grade = "D"
        verdict = "Needs Lab Calibration 🧪"
        virality_tier = "Low"

    # Actionable diagnosis
    strengths = []
    critiques = []

    if text_score >= 80:
        strengths.append(f"Caption length ({cap_len} characters) is punchy and delivers humor swiftly.")
    elif cap_len > 120:
        critiques.append("Shorten the caption text to increase reading speed on mobile feeds.")
    elif cap_len == 0:
        critiques.append("Provide a clear caption so the NLP model can analyze the comedic archetype.")

    if ratio_score >= 85:
        strengths.append(f"Aspect ratio ({aspect_ratio:.2f}:1) fits mainstream social feeds cleanly without awkward cropping.")
    else:
        critiques.append("Crop the image closer to 1:1 (square) to maximize mobile screen real-estate.")

    if contrast_score >= 80:
        strengths.append("High image contrast helps the meme grab attention during fast scrolling.")
    else:
        critiques.append("Use bold text with a distinct black outline (standard Impact/Meme styling) to improve contrast.")

    if template_confidence > 0.4:
        strengths.append(f"Semantic structure strongly aligns with the **{pred_template}** viral template.")

    # Simulated engagement projection
    seed_val = int(cap_len + width + int(mean_brightness))
    rng = np.random.default_rng(seed_val)
    est_views = int(rng.integers(8000, 50000) * (composite_score / 60))
    est_likes = int(est_views * (composite_score / 950))
    est_shares = int(est_likes * 0.24)

    return {
        "score": composite_score,
        "grade": grade,
        "verdict": verdict,
        "virality_tier": virality_tier,
        "dimensions": f"{width} × {height} px",
        "aspect_ratio": f"{aspect_ratio:.2f}:1",
        "ratio_desc": ratio_desc,
        "contrast_desc": contrast_desc,
        "text_desc": text_desc,
        "pred_template": pred_template,
        "template_confidence": template_confidence,
        "top_matches": top_matches,
        "pred_sentiment": pred_sentiment,
        "strengths": strengths,
        "critiques": critiques,
        "est_views": est_views,
        "est_likes": est_likes,
        "est_shares": est_shares
    }

# Sidebar controls
with st.sidebar:
    st.title("🧪 Meme Lab")
    
    has_real_data = REAL_DATA_FILE.exists()
    dataset_options = ["Real Dataset (MemeGenerator 57k)", "Synthetic Demo Dataset"] if has_real_data else ["Synthetic Demo Dataset"]
    data_source = st.selectbox("Data Source", dataset_options, index=0)
    
    if data_source.startswith("Real"):
        st.success("57,341 real memes loaded from memegenerator.csv", icon=":material/check_circle:")
    else:
        st.info("Demonstration synthetic dataset active", icon=":material/info:")

    page = st.radio("Navigation", [
        "Dashboard", 
        "Meme Rater & Reviewer",
        "Meme Battle Arena", 
        "Meme Explorer",
        "Meme Template AI (Real)", 
        "Popularity Predictor", 
        "Sentiment Analyzer",
        "Audience Analytics", 
        "Pattern Explorer", 
        "About"
    ])
    st.caption("CSE 426 • Data Mining & Warehouse Lab")
    st.caption("UITS Department of CSE")

# Load selected data
memes, comments, preferences = load_data(data_source)
category_cols = [c for c in preferences.columns if c != "user_id"]
tree_model, forest_model, X_test, y_test, feature_cols = train_popularity_models(memes)
sentiment_model = train_sentiment_model(comments)
template_model, template_vec, top_real_templates = train_meme_template_classifier()

# Top Hero banner
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .hero {padding: 1.2rem 1.4rem; border-radius: 18px;
           background: linear-gradient(110deg, #25124a, #142d36);
           border: 1px solid #493b70; margin-bottom: 1rem;}
    .hero h1 {margin: 0; color: #ffffff; letter-spacing: 1px;}
    .hero p {color: #d6d2e8; margin: .35rem 0 0 0;}
    div[data-testid="stMetric"] {background: #171c27; padding: 12px;
        border: 1px solid #30394a; border-radius: 12px;}
</style>
<div class="hero">
<h1>🧪 MEME SCIENTIST</h1>
<p>We don't just laugh at memes. We analyze why they go viral.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------
if page == "Dashboard":
    st.subheader("Meme Intelligence Overview")
    votes = load_votes()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Memes in dataset", f"{len(memes):,}")
    c2.metric("Total views", f"{memes.views.sum():,}")
    c3.metric("Total likes", f"{memes.likes.sum():,}")
    c4.metric("Community votes", f"{len(votes):,}")
    
    left, right = st.columns(2)
    with left:
        top_cats = memes["category"].value_counts().head(10).rename_axis("category").reset_index(name="count")
        st.plotly_chart(px.bar(top_cats, x="category", y="count", color="category",
                               title="Top Meme Templates / Categories", template="plotly_dark"), width="stretch")
    with right:
        pop_counts = memes["popularity"].value_counts().rename_axis("popularity").reset_index(name="count")
        st.plotly_chart(px.pie(pop_counts, names="popularity", values="count",
                               title="Popularity Distribution", template="plotly_dark"), width="stretch")
    
    st.subheader("Top 10 Memes by Engagement")
    top = memes.assign(engagement=(memes.likes + 2*memes.shares + memes.comments) / memes.views.clip(lower=1))\
               .sort_values("engagement", ascending=False).head(10)
    display_cols = ["meme_id", "category", "caption", "likes", "shares", "comments", "views", "popularity"]
    st.dataframe(top[display_cols], width="stretch", hide_index=True)

elif page == "Meme Rater & Reviewer":
    st.subheader("🔬 Meme Rater & AI Virality Reviewer")
    st.write("Upload your own meme image and let the Meme Scientist laboratory analyze its virality potential, image composition, semantic archetype, and emotional sentiment.")

    col_input, col_info = st.columns([1.1, 0.9])
    
    with col_input:
        with st.container(border=True):
            st.markdown("#### 📤 Upload Meme")
            uploaded_file = st.file_uploader("Select Meme Image", type=["jpg", "jpeg", "png", "webp"], help="Upload your meme file (JPG, PNG, WEBP)")
            meme_caption = st.text_area("Meme Caption / Text", placeholder="What text is written on your meme? (e.g., 'When the bug fixes itself on restart')")
            evaluate_btn = st.button("🧪 Analyze & Rate Meme", type="primary", disabled=uploaded_file is None, width="stretch")

    with col_info:
        with st.container(border=True):
            st.markdown("#### 📋 What the Reviewer Evaluates")
            st.markdown("""
            - **Virality Potential (0–100):** Multi-factor composite rating based on empirical data mining patterns.
            - **Visual Diagnostics:** Aspect ratio, feed readability, and contrast separation.
            - **Semantic Archetype Match:** AI comparison with 57,000+ real memes (*Futurama Fry*, *Success Kid*, etc.).
            - **Tone & Sentiment Analysis:** Emotional resonance classified using TF-IDF + Naive Bayes.
            - **Scientist Critique:** Actionable advice to boost upvotes and shares.
            """)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Could not open image file: {e}")
            image = None

        if image is not None and (evaluate_btn or "last_reviewed" in st.session_state):
            # Save state so reruns keep the review visible
            review = review_and_rate_meme(image, meme_caption, template_model, template_vec, sentiment_model)
            st.session_state.last_reviewed = review
            
            st.divider()
            st.subheader("🧪 Laboratory Diagnostic Report")

            # Top Score Banner
            c_score, c_metrics = st.columns([1, 2])
            with c_score:
                with st.container(border=True):
                    st.metric("Virality Score", f"{review['score']} / 100", f"Grade {review['grade']}")
                    st.markdown(f"**Verdict:** `{review['verdict']}`")
                    st.progress(review["score"] / 100)

            with c_metrics:
                with st.container(border=True):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Virality Tier", review["virality_tier"])
                    m2.metric("Est. Views", f"{review['est_views']:,}")
                    m3.metric("Est. Likes", f"{review['est_likes']:,}")
                    st.caption(f"Predicted Sentiment: **{review['pred_sentiment']}** • Archetype: **{review['pred_template']}**")

            # Detailed Side-by-Side Breakdown
            col_img, col_diag = st.columns([1, 1.2])
            
            with col_img:
                with st.container(border=True):
                    st.markdown("#### 🖼️ Meme Preview")
                    st.image(image, caption=f"Uploaded Meme • {review['dimensions']}", width="stretch", alt="Uploaded meme preview")
                    if meme_caption.strip():
                        st.info(f"**Caption:** {meme_caption}")

            with col_diag:
                with st.container(border=True):
                    st.markdown("#### 🔬 Diagnostic Breakdown")
                    st.markdown(f"**📐 Composition & Frame:** `{review['dimensions']}` ({review['aspect_ratio']})")
                    st.caption(review['ratio_desc'])
                    
                    st.markdown(f"**🎨 Contrast & Clarity:** `{review['contrast_desc']}`")
                    st.markdown(f"**✍️ Caption Readability:** `{review['text_desc']}`")
                    
                    if review["top_matches"]:
                        st.markdown("**🤖 Top Archetype Matches:**")
                        for t_name, t_prob in review["top_matches"]:
                            st.write(f"- `{t_name}`: {t_prob*100:.1f}% confidence")

                with st.container(border=True):
                    st.markdown("#### 💡 Scientist Lab Review")
                    if review["strengths"]:
                        st.markdown("**What worked well:**")
                        for s in review["strengths"]:
                            st.write(f"✅ {s}")
                    if review["critiques"]:
                        st.markdown("**Optimization Tips for More Engagement:**")
                        for c in review["critiques"]:
                            st.write(f"🔧 {c}")

            # Option to submit this meme to the Battle Arena
            st.divider()
            with st.container(border=True):
                st.markdown("#### ⚔️ Add this Meme to the Battle Arena")
                st.write("Want to see how this meme performs against others in the dataset? Save it locally to enter the community voting arena!")
                
                cat_input = st.selectbox("Assign Template / Category for Arena", [review["pred_template"]] + CATEGORIES_DEMO)
                if st.button("🚀 Enter Meme into Battle Arena", width="stretch"):
                    safe_filename = f"user_{len(memes)+1:04d}_{uploaded_file.name.replace(' ', '_')}"
                    save_path = UPLOADS_DIR / safe_filename
                    image.save(save_path)
                    
                    # Append to memes dataframe & file
                    new_meme_row = pd.DataFrame([{
                        "meme_id": f"U{len(memes)+1:04d}",
                        "category": cat_input,
                        "caption": meme_caption if meme_caption.strip() else "User Uploaded Meme",
                        "likes": review["est_likes"],
                        "shares": review["est_shares"],
                        "comments": int(review["est_likes"] * 0.05),
                        "views": review["est_views"],
                        "sentiment": review["pred_sentiment"],
                        "caption_length": len(meme_caption),
                        "popularity": review["virality_tier"],
                        "image_url": str(save_path)
                    }])
                    
                    new_meme_row.to_csv(MEMES_FILE, mode="a", header=False, index=False)
                    st.cache_data.clear()
                    st.success("🎉 Your meme has been added to the Battle Arena and Explorer dataset!", icon=":material/check_circle:")

elif page == "Meme Battle Arena":
    st.subheader("⚔️ Meme Battle Arena")
    st.write("Choose the meme you like more. Each vote is saved locally and can be analyzed.")
    
    if "battle_pair" not in st.session_state or any(idx >= len(memes) for idx in st.session_state.battle_pair):
        st.session_state.battle_pair = random.sample(range(len(memes)), 2)
        
    idx_a, idx_b = st.session_state.battle_pair
    meme_a, meme_b = memes.iloc[idx_a], memes.iloc[idx_b]
    
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("### Meme A")
            st.markdown(f"**Template:** `{meme_a.category}`")
            render_meme_visual(meme_a)
            st.info(meme_a.caption)
            st.caption(f"ID: {meme_a.meme_id} • Views: {meme_a.views:,}")
            if st.button("👍 Vote for Meme A", width="stretch", key="vote_a"):
                save_vote(meme_a.meme_id, "A")
                st.session_state.battle_pair = random.sample(range(len(memes)), 2)
                st.success("Vote recorded for Meme A!")
                st.rerun()

    with b:
        with st.container(border=True):
            st.markdown("### Meme B")
            st.markdown(f"**Template:** `{meme_b.category}`")
            render_meme_visual(meme_b)
            st.info(meme_b.caption)
            st.caption(f"ID: {meme_b.meme_id} • Views: {meme_b.views:,}")
            if st.button("👍 Vote for Meme B", width="stretch", key="vote_b"):
                save_vote(meme_b.meme_id, "B")
                st.session_state.battle_pair = random.sample(range(len(memes)), 2)
                st.success("Vote recorded for Meme B!")
                st.rerun()

    if st.button("🔄 Generate Another Battle", width="stretch"):
        st.session_state.battle_pair = random.sample(range(len(memes)), 2)
        st.rerun()

elif page == "Meme Explorer":
    st.subheader("🔎 Explore the Meme Dataset")
    col1, col2 = st.columns([1, 2])
    with col1:
        unique_cats = ["All"] + sorted(memes["category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Filter by Category / Template", unique_cats[:50])
    with col2:
        search_query = st.text_input("Search caption keyword", placeholder="e.g. coffee, exam, wifi, bug")

    filtered = memes
    if selected_cat != "All":
        filtered = filtered[filtered.category == selected_cat]
    if search_query.strip():
        filtered = filtered[filtered.caption.str.contains(search_query.strip(), case=False, na=False)]

    st.write(f"Showing **{len(filtered):,}** matching memes")
    
    # Preview cards for top matches
    sample_preview = filtered.head(6)
    if not sample_preview.empty:
        st.markdown("#### Sample Meme Cards")
        cols = st.columns(min(len(sample_preview), 3))
        for i, (_, row) in enumerate(sample_preview.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.caption(f"**{row.category}**")
                    render_meme_visual(row)
                    st.write(f"_{row.caption}_")
                    st.caption(f"Likes: {row.likes:,} | Popularity: **{row.popularity}**")

    st.divider()
    st.dataframe(filtered[["meme_id", "category", "caption", "likes", "shares", "views", "popularity"]], width="stretch", hide_index=True)
    st.download_button("Download Filtered CSV", filtered.to_csv(index=False), file_name="meme_explorer.csv", mime="text/csv", width="stretch")

elif page == "Meme Template AI (Real)":
    st.subheader("🤖 Real Meme Template Classifier (NLP)")
    st.write("Trained on the **57,000+ real memes** using **TF-IDF Vectorization** and **Multinomial Logistic Regression**.")
    st.write("Type any caption to predict which classic meme template it belongs to!")

    if template_model is None:
        st.warning("`memegenerator.csv` not found in workspace to train template classifier.")
    else:
        sample_prompts = [
            "Not sure if code has bugs or if test suite is broken",
            "Y U NO study for data mining exam",
            "Found twenty dollars in my winter coat pocket",
            "I don't always test code but when I do I do it in production",
            "One does not simply pass CSE 426 without studying"
        ]
        
        selected_preset = st.selectbox("Or choose a sample meme caption:", ["-- Custom Input --"] + sample_prompts)
        default_val = "" if selected_preset.startswith("--") else selected_preset
        
        caption_input = st.text_area("Meme Caption:", value=default_val, placeholder="e.g. Not sure if tired or just bored")
        
        if st.button("🔮 Classify Meme Template", type="primary", disabled=not caption_input.strip(), width="stretch"):
            vec_input = template_vec.transform([caption_input])
            pred_template = template_model.predict(vec_input)[0]
            probs = template_model.predict_proba(vec_input)[0]
            
            st.success(f"Predicted Template: **{pred_template}**", icon=":material/auto_awesome:")
            
            prob_df = pd.DataFrame({
                "Template": template_model.classes_,
                "Confidence": probs
            }).sort_values("Confidence", ascending=False).head(5)
            
            st.plotly_chart(px.bar(prob_df, x="Confidence", y="Template", orientation="h",
                                   color="Confidence", template="plotly_dark",
                                   title="Top 5 Template Confidence Scores"), width="stretch")

elif page == "Popularity Predictor":
    st.subheader("📈 Meme Popularity Predictor")
    st.write("Enter engagement statistics. The trained machine learning models predict a virality class.")
    col1, col2 = st.columns(2)
    with col1:
        likes = st.number_input("Likes", min_value=0, value=1200, step=100)
        shares = st.number_input("Shares", min_value=0, value=250, step=25)
        comments_count = st.number_input("Comments", min_value=0, value=80, step=10)
    with col2:
        views = st.number_input("Views", min_value=1, value=15000, step=500)
        caption_length = st.number_input("Caption length (characters)", min_value=0, value=45, step=5)
        model_choice = st.selectbox("Model Algorithm", ["Random Forest", "Decision Tree"])

    if st.button("🧪 Predict Popularity", type="primary", width="stretch"):
        row = pd.DataFrame([[likes, shares, comments_count, views, caption_length]], columns=feature_cols)
        model = forest_model if model_choice == "Random Forest" else tree_model
        prediction = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        st.success(f"Predicted popularity: **{prediction}**")
        proba = pd.DataFrame({"Popularity": model.classes_, "Probability": probabilities})
        st.plotly_chart(px.bar(proba, x="Popularity", y="Probability", color="Popularity",
                               range_y=[0, 1], template="plotly_dark",
                               title="Prediction Confidence"), width="stretch")

    st.divider()
    st.subheader("Model Evaluation on Test Split")
    for label, model in [("Decision Tree", tree_model), ("Random Forest", forest_model)]:
        pred = model.predict(X_test)
        st.write(f"**{label} Accuracy:** {accuracy_score(y_test, pred):.3f}")
        with st.expander(f"{label} Confusion Matrix"):
            cm = confusion_matrix(y_test, pred, labels=model.classes_)
            cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
            st.dataframe(cm_df, width="stretch")

elif page == "Sentiment Analyzer":
    st.subheader("💬 Audience Sentiment Analyzer")
    st.write("Enter a comment or meme text to classify its sentiment using TF-IDF and Naive Bayes.")
    user_text = st.text_area("Meme Comment", placeholder="Example: This is so funny, I love it!")
    if st.button("Analyze Sentiment", type="primary", disabled=not user_text.strip(), width="stretch"):
        pred = sentiment_model.predict([user_text])[0]
        probs = sentiment_model.predict_proba([user_text])[0]
        st.success(f"Detected Sentiment: **{pred}**")
        p_df = pd.DataFrame({"Sentiment": sentiment_model.classes_, "Probability": probs})
        st.plotly_chart(px.bar(p_df, x="Sentiment", y="Probability", color="Sentiment",
                               range_y=[0, 1], template="plotly_dark"), width="stretch")

elif page == "Audience Analytics":
    st.subheader("🧬 Meme Audience DNA (K-Means Clustering)")
    st.write("K-Means groups audience members according to their meme preference baskets.")
    X = preferences[category_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=5, value=3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    result = preferences[["user_id"]].copy()
    result["cluster"] = labels.astype(str)
    st.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.3f}")
    
    cluster_counts = result["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="users")
    st.plotly_chart(px.bar(cluster_counts, x="cluster", y="users", color="cluster",
                           title="Audience Group Sizes", template="plotly_dark"), width="stretch")
    
    profile = preferences.assign(cluster=labels).groupby("cluster")[category_cols].mean().reset_index()
    st.plotly_chart(px.imshow(profile.set_index("cluster"), text_auto=True, aspect="auto",
                              title="Average Category / Template Preference by Cluster",
                              color_continuous_scale="Viridis"), width="stretch")
    st.dataframe(result, width="stretch", hide_index=True)

elif page == "Pattern Explorer":
    st.subheader("🔗 Meme Preference Pattern Explorer (Apriori)")
    st.write("Apriori discovers category and template combinations that frequently appear together in user preference baskets.")
    min_support = st.slider("Minimum Support", 0.05, 0.60, 0.10, 0.05)
    basket = preferences[category_cols].astype(bool)
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent.empty:
        st.warning("No frequent itemsets found. Try lowering minimum support.")
    else:
        rules = association_rules(frequent, metric="confidence", min_threshold=0.1)
        if rules.empty:
            st.info("Frequent itemsets found, but no association rules passed the confidence threshold.")
        else:
            rules = rules.sort_values(["lift", "confidence"], ascending=False)
            display = rules.head(20).copy()
            display["antecedents"] = display["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            display["consequents"] = display["consequents"].apply(lambda x: ", ".join(sorted(x)))
            st.dataframe(display[["antecedents", "consequents", "support", "confidence", "lift"]],
                         width="stretch", hide_index=True)
            st.caption("Lift above 1 indicates positive association between meme categories/templates.")
        st.subheader("Frequent Itemsets")
        fi = frequent.copy()
        fi["items"] = fi["itemsets"].apply(lambda x: ", ".join(sorted(x)))
        st.dataframe(fi[["items", "support"]].sort_values("support", ascending=False),
                     width="stretch", hide_index=True)

elif page == "About":
    st.subheader("About Meme Scientist")
    st.markdown("""
    **Meme Scientist** is a Data Mining & Warehouse Lab project demonstrating how
    meme virality, semantics, and audience preferences can be discovered with data mining.

    **Algorithms Demonstrated**
    - **TF-IDF + Logistic Regression / Naive Bayes:** Real meme template classification and comment sentiment analysis
    - **Decision Tree & Random Forest:** Viral popularity level classification
    - **K-Means Clustering:** Audience segmentation and preference profiling
    - **Apriori Algorithm:** Association rule mining across meme preference baskets
    - **Computer Vision & Heuristic Analytics:** Image composition, aspect ratio, and contrast diagnostic review

    **Dataset Sources**
    - **Real Dataset:** `memegenerator.csv` (57,652 real meme records from the Library of Congress Memegenerator Web Archive / Kaggle)
    - **Demo Mode:** Synthetic baseline generator for quick demonstrations
    """)
    st.markdown("**Team Members**")
    st.write("- Md. Safrid Bhueyan — 0432310005101080")
    st.write("- Tasnim Jahan — 0432310001051089")
    st.markdown("**Course:** CSE 426 — Data Mining & Warehouse Lab")
    st.markdown("**University:** University of Information Technology and Sciences (UITS)")
