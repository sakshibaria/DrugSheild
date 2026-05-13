import joblib
import os

# ------------------------------------------------
# LOAD MODELS (loaded once at import time)
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NLP_DIR = os.path.join(BASE_DIR, "NLP_Text_Model", "NLP Text work updated")

model = joblib.load(os.path.join(NLP_DIR, "nlp_model.pkl"))
vectorizer = joblib.load(os.path.join(NLP_DIR, "tfidf_vectorizer.pkl"))

# Drug-related keywords for fallback detection
DRUG_KEYWORDS = [
    "weed", "cocaine", "heroin", "drug", "drugs",
    "pills", "powder", "tablet", "supplier",
    "delivery", "deliver", "stuff", "meth",
    "crack", "fentanyl", "opium", "marijuana",
    "hash", "mdma", "ecstasy", "lsd",
]


def predict_text(text):
    """
    Predict whether a text message is Suspicious or Normal.
    Uses ML probability + keyword fallback.
    Returns: (label, probability_percentage)
    """
    if not text or not text.strip():
        return "Normal", 0.0

    text_lower = text.lower().strip()

    # Keyword-based flag
    keyword_flag = any(kw in text_lower for kw in DRUG_KEYWORDS)

    # ML prediction
    text_vector = vectorizer.transform([text])
    probability = model.predict_proba(text_vector)[0][1]

    if keyword_flag or probability > 0.65:
        label = "Suspicious"
        confidence = max(probability, 0.7) * 100
    else:
        label = "Normal"
        confidence = (1 - probability) * 100

    return label, confidence
