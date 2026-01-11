"""Streamlit interface for the Unclear Software Requirement Detector."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from detector import (
    analyze_requirement,
    highlight_vague_terms,
    highlight_vague_terms_colored,
)


st.title("🧾 Unclear Software Requirement Detector")
st.markdown("Analyze the clarity of software requirement statements using rule-based NLP + simple ML.")
st.caption("Type a requirement below or switch to batch mode to check multiple at once.")


# Sensitivity controls (sidebar)
st.sidebar.header("Sensitivity settings")
st.sidebar.caption("Tune how strict the detector is.")
max_length = st.sidebar.slider(
    "Max tokens before a sentence is complex",
    min_value=5,
    max_value=60,
    value=20,
)
ml_threshold = st.sidebar.slider(
    "ML probability threshold for 'Unclear'",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.05,
)

st.markdown("---")

col_mode, col_help = st.columns([2, 1])
with col_mode:
    mode = st.radio("Choose analysis mode", ["Single statement", "Batch (one per line)"], horizontal=True)
with col_help:
    st.info("Use batch mode to quickly screen many requirements at once.")


st.markdown("---")

if "history" not in st.session_state:
    st.session_state["history"] = []

if mode == "Single statement":

    st.subheader("Single requirement analysis")
    input_text = st.text_area("Enter a requirement statement", height=140)

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if not input_text.strip():
            st.warning("Please enter a statement.")
        else:
            result = analyze_requirement(
                input_text,
                max_length=max_length,
                ml_threshold=ml_threshold,
            )

            # Store in in-session history
            st.session_state["history"].append({
                "Requirement": input_text,
                "Status": result["status"],
                "Reasons": result["reasons"],
                "Tags": result.get("tags", []),
                "Severity": result.get("severity", 1),
            })

            col_status, col_sentence = st.columns([1, 2])
            with col_status:
                status = result["status"]
                color = (
                    "green" if status == "Clear" else
                    "orange" if status == "Partially Clear" else
                    "red"
                )
                st.markdown(f"**Status:** :{color}[{status}]")

                # Simple severity display
                severity = result.get("severity", 1)
                st.caption(f"Severity: {severity} / 3")
            with col_sentence:
                st.write("**Highlighted requirement**")
                st.markdown(highlight_vague_terms_colored(input_text))

            # Tags as badges
            tags = result.get("tags", [])
            if tags:
                st.write("### Issue categories")
                tag_str = " ".join([f"`{t}`" for t in tags])
                st.write(tag_str)

            st.write("### Reasons")
            for reason in result['reasons']:
                st.write(f"- {reason}")

            feat_reason = [
                f for f in result['reasons']
                if f.startswith('Top influential words (ML): ')
            ]
            if feat_reason:
                st.write("### 🔍 Feature importance (ML)")
                word_data = feat_reason[-1].replace('Top influential words (ML): ', '')
                items = [item.strip().split(' (') for item in word_data.split(',')]
                df = pd.DataFrame({
                    'Word': [w for w, _ in items],
                    'Weight': [float(v.replace(')', '')) for _, v in items]
                })
                fig, ax = plt.subplots()
                ax.barh(df['Word'], df['Weight'], color='lightcoral')
                ax.set_xlabel('Weight toward Unclear')
                ax.set_ylabel('Word')
                st.pyplot(fig)
else:
    st.subheader("Batch analysis")

    st.caption("Enter one requirement per line. Empty lines are ignored.")

    batch_text = st.text_area(
        "Requirements (one per line)",
        height=200,
    )

    analyze_batch_clicked = st.button("Analyze batch", type="primary")

    if analyze_batch_clicked:
        lines = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one non-empty line.")
        else:
            records = []
            for line in lines:
                result = analyze_requirement(
                    line,
                    max_length=max_length,
                    ml_threshold=ml_threshold,
                )
                records.append({
                    "Requirement": line,
                    "Highlighted": highlight_vague_terms_colored(line),
                    "Status": result["status"],
                    "Reasons": " | ".join(result["reasons"]),
                    "Tags": ", ".join(result.get("tags", [])),
                    "Severity": result.get("severity", 1),
                })

            df_results = pd.DataFrame(records)

            # Dashboard summary
            st.write("### Summary dashboard")
            counts = df_results["Status"].value_counts().to_dict()
            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("Clear", counts.get("Clear", 0))
            col_c2.metric("Partially Clear", counts.get("Partially Clear", 0))
            col_c3.metric("Unclear", counts.get("Unclear", 0))

            st.write("### Summary table")
            st.dataframe(
                df_results[["Requirement", "Status", "Severity", "Tags", "Reasons"]],
                use_container_width=True,
            )

            st.write("### Detailed view")
            for _, row in df_results.iterrows():
                with st.expander(f"{row['Status']} - {row['Requirement'][:60]}..."):
                    status = row["Status"]
                    color = (
                        "green" if status == "Clear" else
                        "orange" if status == "Partially Clear" else
                        "red"
                    )
                    st.markdown(f"**Status:** :{color}[{status}]")
                    st.caption(f"Severity: {row['Severity']} / 3")
                    if row["Tags"]:
                        st.write("Tags:", " ".join([f"`{t}`" for t in row["Tags"].split(", ")]))
                    st.markdown(row["Highlighted"])
                    st.write(f"Reasons: {row['Reasons']}")

            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="requirement_analysis_results.csv",
                mime="text/csv",
            )

# Simple history panel in the sidebar
if st.session_state["history"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent requirements")
    for item in reversed(st.session_state["history"][-5:]):
        st.sidebar.caption(f"{item['Status']}: {item['Requirement'][:40]}...")

st.markdown("---")