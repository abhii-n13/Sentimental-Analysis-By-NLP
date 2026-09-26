import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocessing import clean_text


# Page config

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "emo_model.pkl"

EMOTION_STYLE = {
    "joy":      {"emoji": "😄", "color": "#FFC93C", "glow": "255, 201, 60"},
    "sadness":  {"emoji": "😢", "color": "#4E9DE6", "glow": "78, 157, 230"},
    "anger":    {"emoji": "😠", "color": "#FF5C5C", "glow": "255, 92, 92"},
    "fear":     {"emoji": "😨", "color": "#B78CFF", "glow": "183, 140, 255"},
    "love":     {"emoji": "❤️", "color": "#FF7AA8", "glow": "255, 122, 168"},
    "surprise": {"emoji": "😲", "color": "#3FE0C5", "glow": "63, 224, 197"},
}

EXAMPLES = [
    "I just got accepted into my dream university!",
    "I feel so alone since she left.",
    "How dare you cancel without telling me!",
    "My hands are shaking, I'm terrified.",
    "I never expected this gift from you.",
    "Being with my family meant everything to me.",
]


# Custom CSS — dark theme, glassmorphism, gradients, animations

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #1c1435 0%, #0d0a1a 45%, #08060f 100%);
        color: #f1eefc;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 1.8rem 1rem 0.6rem 1rem;
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #FF7AA8, #B78CFF 45%, #4E9DE6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        color: #a79fc7;
        font-size: 1.02rem;
        margin-top: 0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.4rem 1.4rem 0.8rem 1.4rem;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.2rem;
    }

    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #f1eefc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        font-size: 1.02rem !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid #B78CFF !important;
        box-shadow: 0 0 0 2px rgba(183, 140, 255, 0.25) !important;
    }

    div.stButton > button[kind="primary"] {
        width: 100%;
        background: linear-gradient(90deg, #FF7AA8, #B78CFF, #4E9DE6);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 0;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        transition: 0.35s ease;
        box-shadow: 0 4px 18px rgba(183, 140, 255, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(183, 140, 255, 0.5);
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        color: #cfc8ea;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.82rem;
        transition: 0.25s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #B78CFF;
        color: #ffffff;
        background: rgba(183, 140, 255, 0.15);
        transform: translateY(-1px);
    }

    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.9) translateY(8px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    .result-card {
        animation: popIn 0.45s ease-out;
        text-align: center;
        padding: 1.6rem 1rem;
        border-radius: 20px;
        margin-top: 0.6rem;
        margin-bottom: 1.4rem;
    }
    .result-emoji {
        font-size: 3.4rem;
        line-height: 1;
        filter: drop-shadow(0 0 14px rgba(255,255,255,0.25));
    }
    .result-label {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 1.9rem;
        text-transform: capitalize;
        margin: 0.3rem 0 0.1rem 0;
        letter-spacing: 0.5px;
    }
    .result-confidence {
        color: #c9c2e6;
        font-size: 0.95rem;
    }

    .section-label {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #d9d4f0;
        margin: 0.4rem 0 0.3rem 0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150f28, #0a0716);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)



# Load model (single Pipeline: TF-IDF + Logistic Regression)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


pipeline = load_model()


def predict(text: str):
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    proba_df = pd.DataFrame({
        "emotion": pipeline.classes_,
        "probability": proba
    }).sort_values("probability", ascending=True).reset_index(drop=True)
    return pred, proba_df, cleaned


def render_probability_chart(proba_df: pd.DataFrame):
    colors = [EMOTION_STYLE.get(e, {"color": "#8888aa"})["color"] for e in proba_df["emotion"]]
    fig = go.Figure(go.Bar(
        x=proba_df["probability"] * 100,
        y=proba_df["emotion"].str.capitalize(),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in proba_df["probability"]],
        textposition="outside",
        textfont=dict(color="#f1eefc", size=13),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfc8ea", family="Inter"),
        margin=dict(l=10, r=30, t=10, b=10),
        height=260,
        xaxis=dict(range=[0, 105], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})



# Sidebar

with st.sidebar:
    st.markdown("### 🧠 Model Info")
    st.markdown(
        """
        <div style="color:#cfc8ea; font-size:0.9rem; line-height:1.7;">
        <b>Pipeline:</b> TF-IDF + Logistic Regression<br>
        <b>File:</b> emo_model.pkl<br>
        <b>Classes:</b> 6 emotions<br>
        <b>Dataset:</b> ~16,000 labeled sentences
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🎨 Emotions Detected")
    for emo, style in EMOTION_STYLE.items():
        st.markdown(
            f"<span style='color:{style['color']}; font-weight:600;'>{style['emoji']} {emo.capitalize()}</span>",
            unsafe_allow_html=True,
        )


# Hero header

st.markdown(
    """
    <div class="hero">
        <h1>🎭 Emotion AI</h1>
        <p>Type anything — I'll read the emotion behind your words.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Input card

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown('<div class="section-label">✨ Try an example</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, key=f"ex_{i}", type="secondary"):
        st.session_state.text_input = example

st.markdown('<div class="section-label">💬 Your text</div>', unsafe_allow_html=True)
user_text = st.text_area(
    "Your text",
    value=st.session_state.text_input,
    height=110,
    placeholder="e.g. I can't believe I finally finished the marathon!",
    label_visibility="collapsed",
)

analyze_clicked = st.button("✨ Analyze Emotion", type="primary")

st.markdown('</div>', unsafe_allow_html=True)


# Result

if analyze_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        pred, proba_df, cleaned = predict(user_text)
        style = EMOTION_STYLE.get(pred, {"emoji": "🤔", "color": "#999999", "glow": "150,150,150"})
        confidence = proba_df.loc[proba_df["emotion"] == pred, "probability"].values[0]

        st.markdown(
            f"""
            <div class="result-card" style="
                background: radial-gradient(circle at top, rgba({style['glow']}, 0.18), rgba({style['glow']}, 0.03));
                border: 1px solid rgba({style['glow']}, 0.45);
                box-shadow: 0 0 40px rgba({style['glow']}, 0.25);">
                <div class="result-emoji">{style['emoji']}</div>
                <div class="result-label" style="color:{style['color']};">{pred}</div>
                <div class="result-confidence">Confidence: {confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">📊 Emotion breakdown</div>', unsafe_allow_html=True)
        render_probability_chart(proba_df)

        with st.expander("🔍 See cleaned text (what the model actually saw)"):
            st.code(cleaned or "(empty after cleaning)")

st.markdown(
    "<div style='text-align:center; color:#6f6890; font-size:0.8rem; margin-top:2rem;'>"
    "Built with Bag of words + TF-IDF + Multinomial Naive Bayes + Multinomial_NB + Logistic Regression  · scikit-learn · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)