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


def suggest_rewrite(text, tags, iteration: int = 1) -> str:
    """Return a simple, template-based rewrite suggestion.

    The goal is to incrementally replace vague phrases with measurable
    placeholders and optionally add missing constraints. To keep the
    behavior agent-like, only a couple of vague terms are adjusted per
    iteration.
    """
    suggestion = text

    # Handle vague terms incrementally
    if "VAGUE_TERMS" in tags:
        # Find vague terms present in order of appearance
        lower = suggestion.lower()
        present_terms = [t for t in VAGUE_WORDS if t in lower]
        # Adjust at most two per iteration
        start_index = (iteration - 1) * 2
        to_fix = present_terms[start_index:start_index + 2]

        for term in to_fix:
            pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
            replacement = term
            if term in {"fast", "quick", "efficient"}:
                replacement = "respond within [X seconds]"
            elif term in {"many", "large"}:
                replacement = "[N] users"
            elif term == "user-friendly":
                replacement = "achieve a usability score of at least [X]/[Y]"
            elif term in {"scalable", "flexible"}:
                replacement = "support up to [N] users without performance degradation"
            elif term in {"robust", "reliable"}:
                replacement = "maintain an uptime of [X]% over [Y] days"
            elif term == "secure":
                replacement = "meet [X] security standard (e.g., OWASP Top 10)"

            suggestion = pattern.sub(replacement, suggestion)

    # If there are no explicit numeric constraints, append a generic one
    if "NO_CONSTRAINTS" in tags:
        suggestion = suggestion.rstrip(" .") + \
            ". The system shall respond within [X seconds] for up to [N] users."

    # For complex sentences, lightly suggest a split using a semicolon
    if "COMPLEX_SENTENCE" in tags and " and " in suggestion:
        suggestion = suggestion.replace(" and ", "; and ", 1)

    return suggestion


def build_agent_rationale(tags) -> list:
    """Map analysis tags to simple rationale bullet points for the agent UI."""
    rationale = []
    if "VAGUE_TERMS" in tags:
        rationale.append("Detected vague term(s) that should be made measurable.")
    if "NO_CONSTRAINTS" in tags:
        rationale.append("Detected missing measurable constraints (time, volume, or users).")
    if "COMPLEX_SENTENCE" in tags:
        rationale.append("Sentence looks long or complex; consider splitting into smaller requirements.")
    if "ML_AMBIGUITY" in tags:
        rationale.append("ML model also flags potential ambiguity in this requirement.")
    if not rationale:
        rationale.append("Requirement appears reasonably clear; minor wording tweaks only.")
    return rationale