"""
Hybrid detector that combines rule-based logic and ML model
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from model import predict_unclear, explain_features

# Ensure required NLTK tokenizer data is available.
# Newer NLTK versions may require both 'punkt' and 'punkt_tab'.
for resource in ["punkt", "punkt_tab"]:
    try:
        # punkt is located under tokenizers/punkt, punkt_tab under tokenizers/punkt_tab
        subdir = "punkt_tab" if resource == "punkt_tab" else "punkt"
        nltk.data.find(f"tokenizers/{subdir}")
    except LookupError:
        nltk.download(resource)

VAGUE_WORDS = [
    'fast', 'quick', 'efficient', 'user-friendly', 'secure', 'many', 'large',
    'simple', 'easy', 'robust', 'scalable', 'flexible', 'reliable'
]

def detect_vague_terms(text):
    return [word for word in VAGUE_WORDS if word in text.lower()]

def highlight_vague_terms(text):
    """Return text with vague terms wrapped for basic highlighting (bold).

    This keeps formatting simple and safe to combine with more advanced
    Streamlit-side coloring if needed.
    """
    highlighted = text
    lower_text = text.lower()
    # Replace from longest to shortest to avoid partial overlaps
    for term in sorted(VAGUE_WORDS, key=len, reverse=True):
        if term in lower_text:
            highlighted = re.sub(
                term,
                f"**{term}**",
                highlighted,
                flags=re.IGNORECASE,
            )
    return highlighted

def highlight_vague_terms_colored(text):
    """Return text with vague terms colored using Streamlit markdown syntax.

    Example: "fast" -> ":red[fast]".
    """
    highlighted = text
    lower_text = text.lower()
    for term in sorted(VAGUE_WORDS, key=len, reverse=True):
        if term in lower_text:
            highlighted = re.sub(
                term,
                f":red[{term}]",
                highlighted,
                flags=re.IGNORECASE,
            )
    return highlighted

def detect_missing_constraints(text):
    measurable_patterns = re.findall(r'\b\d+\s*(seconds?|minutes?|ms|MB|GB|%|users?)\b', text, re.IGNORECASE)
    return len(measurable_patterns) == 0

def detect_complex_sentence(text, max_length=20):
    return len(word_tokenize(text)) > max_length

def analyze_requirement(text, max_length=20, ml_threshold=0.6):
    """Analyze a requirement and return status, reasons, tags, and severity.

    - tags: high-level issue categories (e.g., VAGUE_TERMS, NO_CONSTRAINTS).
    - severity: simple rating based on number and type of issues (1–3).
    """
    reasons = []
    tags = set()

    vague = detect_vague_terms(text)
    if vague:
        reasons.append(f"Vague terms detected: {', '.join(vague)}")
        tags.add("VAGUE_TERMS")

    if detect_complex_sentence(text, max_length=max_length):
        reasons.append("Sentence too long or complex.")
        tags.add("COMPLEX_SENTENCE")

    if detect_missing_constraints(text):
        reasons.append("No measurable constraints provided.")
        tags.add("NO_CONSTRAINTS")

    # ML layer
    prob_unclear = predict_unclear(text)
    if prob_unclear > ml_threshold and "Vague terms" not in ''.join(reasons):
        reasons.append("ML model suggests possible ambiguity.")
        tags.add("ML_AMBIGUITY")

    top_features = explain_features(text)
    if top_features:
        formatted = ", ".join([f"{w} ({round(wt, 2)})" for w, wt in top_features])
        reasons.append(f"Top influential words (ML): {formatted}")

    # Final classification
    if len(reasons) == 0:
        status = "Clear"
    elif len(reasons) == 1:
        status = "Partially Clear"
    else:
        status = "Unclear"

    # Simple severity rating: 1 (low) to 3 (high)
    if status == "Clear":
        severity = 1
    elif status == "Partially Clear":
        severity = 2
    else:
        severity = 3

    return {
        "status": status,
        "reasons": reasons or ["Sentence appears well-defined."],
        "tags": sorted(tags),
        "severity": severity,
    }