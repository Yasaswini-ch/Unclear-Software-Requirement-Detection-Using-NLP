🧠 Unclear Software Requirement Detection Using NLP

A simple, explainable NLP-based tool that detects whether a software requirement statement is **Clear**, **Partially Clear**, or **Unclear**.

It is designed to be:

- **Lightweight** – minimal dependencies, easy to run on a laptop.
- **Transparent** – rule-based checks combined with a very small ML model.
- **Beginner-friendly** – ideal for internship projects and portfolios.

---

## 🎯 Project Goal

Help software engineers and students **detect unclear requirement statements early** by:

- Finding **vague or subjective terms**.
- Flagging **missing measurable constraints** (time, amount, number of users, etc.).
- Highlighting **long or complex sentences**.
- Using a **simple ML model** to add an extra ambiguity score.

The final output clearly tells you whether the requirement is Clear / Partially Clear / Unclear and *why*.

---

## ⚙️ Features

- **Rule-based detection** of:
  - Vague words like `fast`, `user-friendly`, `scalable`, `robust`, etc.
  - Missing measurable constraints such as `2 seconds`, `500 users`, `10 GB`, `%`, etc.
  - Sentences that are **too long/complex**.
- **Lightweight ML model**:
  - A small Logistic Regression classifier built with scikit-learn.
  - Trained on a tiny, synthetic dataset of requirement sentences.
  - Produces a probability that a sentence is *unclear*.
- **Explainable output**:
  - Every prediction comes with a list of **reasons**.
  - Shows **top influential words** from the ML model and their weights.
- **Streamlit web app**:
  - **Single-statement mode** for detailed analysis of one requirement.
  - **Batch mode** (one requirement per line) for screening many requirements together.
  - **CSV export** of batch results for reporting or further analysis.
  - **Sensitivity controls** (sidebar sliders) to adjust complexity threshold and ML probability threshold.
  - Highlighting of vague terms directly in the displayed requirement text.

---

## 🧰 Tech Stack

- **Language**: Python 3
- **User Interface**: Streamlit
- **Natural Language Processing**: NLTK (tokenization)
- **Machine Learning**: scikit-learn (CountVectorizer + LogisticRegression)
- **Visualization**: Matplotlib, Pandas

---

## �️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/unclear-requirement-detector-nlp.git
cd unclear-requirement-detector-nlp
```

### 2. Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK tokenizer data (`punkt`)

The code will **try to download** `punkt` automatically at runtime. If you prefer to install it manually (recommended for offline/CI use), run:

```bash
python -m nltk.downloader punkt
```

---

## 🚀 Running the Streamlit App

From the project folder:

```bash
streamlit run app.py
```

Then open your browser to the URL shown in the terminal (usually:

- `http://localhost:8501`)

You will see:

- A **title and description** of the tool.
- A **mode selector** (single statement or batch).
- A **sidebar** with sensitivity sliders.

### Single-statement mode

1. Select **Single statement**.
2. Type or paste a requirement into the text area.
3. Click **Analyze**.

You will get:

- A color-coded **Status** (`Clear`, `Partially Clear`, or `Unclear`).
- A simple **severity rating** (1–3).
- The requirement text with **vague terms highlighted in color**.
- **Issue categories** as small tags (e.g., `VAGUE_TERMS`, `NO_CONSTRAINTS`).
- A **list of reasons** explaining the decision.
- A **bar chart** of the most influential words from the ML model (when available).

### Batch mode

1. Select **Batch (one per line)**.
2. Enter multiple requirement statements, one per line.
3. Click **Analyze batch**.

You will get:

- A **summary dashboard** showing counts of Clear / Partially Clear / Unclear.
- A **summary table** with requirement, status, severity, tags, and reasons.
- A **detailed view** (expanders) for each requirement, with:
  - Color-coded status and severity.
  - Tags for detected issues.
  - Highlighted text and reasons.
- A **Download results as CSV** button.

---

## 🧩 How It Works (High Level)

### Step 1: Rule-Based Analysis (`detector.py`)

1. **Vague term detection**
   - Uses a list of vague words (e.g., `fast`, `quick`, `efficient`, `user-friendly`, `scalable`, `robust`).
   - If any appear in the requirement, they are listed as *vague terms*.

2. **Missing measurable constraints**
   - Uses a regular expression to search for patterns like `2 seconds`, `5 minutes`, `100 users`, `10 GB`, `20%`, etc.
   - If **no** such pattern is found, it assumes the statement might be missing measurable constraints.

3. **Complex sentence check**
   - Uses NLTK’s `word_tokenize` to count tokens.
   - If there are more than a certain number of tokens (e.g., > 20), the sentence is considered long/complex.

All triggered checks add **human-readable reasons** to a list.

### Step 2: ML Enhancement (`model.py`)

1. A tiny training dataset of requirement sentences is defined in code.
2. A **CountVectorizer** converts sentences into bag-of-words features.
3. A **LogisticRegression** model is trained to predict:
   - `1` → Unclear
   - `0` → Clear
4. For a new sentence:
   - The model outputs a **probability of being unclear**.
   - If this probability is high enough (for example, > 0.6), a reason like *"ML model suggests possible ambiguity."* is added.

### Step 3: Explainable Output & Visualization

1. The ML model’s coefficients are extracted and mapped to words.
2. For the given sentence, only words that actually appear in it are considered.
3. The words are sorted by how strongly they influence the **unclear** class (by absolute weight).
4. The top few words (e.g., 5) are returned as:
   - `word (weight)`
5. The Streamlit app parses this and shows:
   - A **list of influential words** in text.
   - A **bar chart** (word vs. weight) illustrating influence toward “Unclear”.
6. Vague terms in the requirement text are **visually highlighted** using Streamlit’s markdown coloring.

Finally, the tool decides the overall **status**:

- If there are **no reasons** → `Clear` (with a default positive message).
- If there is **1 reason** → `Partially Clear`.
- If there are **2 or more reasons** → `Unclear`.

---

## 📈 Example Interaction

**Input:**

> The system should load data quickly and handle many users.

**Possible output:**

- **Status:** Unclear
- **Reasons:**
  - Vague terms detected: quickly, many
  - No measurable constraints provided.
  - Top influential words (ML): quickly (0.83), users (0.45)

This tells you *exactly* what to improve (e.g., replace `quickly` with a specific time and replace `many users` with a number).

---

## 🧪 Sample Test Cases

These are example inputs and the kind of output you can expect in **single-statement** mode:

1. **Input:** `The system shall respond in under 2 seconds.`
   - **Expected status:** Clear
   - **Severity:** 1 / 3
   - **Tags:** *(none or minimal)*
   - **Why:** Has a clear, measurable constraint (`2 seconds`).

2. **Input:** `The UI should be user-friendly.`
   - **Expected status:** Unclear
   - **Severity:** 3 / 3
   - **Tags:** `VAGUE_TERMS`, `NO_CONSTRAINTS`
   - **Why:** Contains vague term `user-friendly` and no measurable definition.

3. **Input:** `The app must handle 500 users without errors.`
   - **Expected status:** Clear
   - **Severity:** 1 / 3
   - **Tags:** *(none or minimal)*
   - **Why:** Well-defined numeric constraint (`500 users`) and clear objective.

4. **Input:** `The system shall be fast and scalable.`
   - **Expected status:** Unclear
   - **Severity:** 3 / 3
   - **Tags:** `VAGUE_TERMS`
   - **Why:** Vague terms `fast` and `scalable` without numeric definitions.

You can find more examples in `sample_tests.txt`, and you can paste them into the batch mode to see the dashboard and CSV export.

---

## 🏗️ Folder Structure

```text
unclear-requirement-detector-nlp/
├── app.py              # Streamlit UI
├── detector.py         # Rule-based + ML analyzer
├── model.py            # Lightweight classifier (CountVectorizer + LogisticRegression)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── sample_tests.txt    # Example input/output tests
```

---

## 📚 Why This Project Matters

In real-world software projects, **unclear requirements** are a major source of:

- Miscommunication between stakeholders and developers.
- Incorrect implementation and feature rework.
- Delays, cost overruns, and frustration.

By automating an **early clarity check**, this project helps:

- Students learn about **NLP + ML** on a practical use case.
- Teams quickly prototype a **requirements quality checker**.
- Interview or internship candidates showcase a **small but realistic NLP project**.

> This project is not a replacement for human review, but a supporting tool to make requirement discussions more concrete.

---

## 🪪 License

This project is open-source and available under the **MIT License**.

