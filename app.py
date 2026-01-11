"""Streamlit interface for the Unclear Software Requirement Detector."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from detector import (
    analyze_requirement,
    highlight_vague_terms,
    highlight_vague_terms_colored,
    suggest_rewrite,
    build_agent_rationale,
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
    mode = st.radio(
        "Choose analysis mode",
        ["Single statement", "Batch (one per line)", "Agent Assist"],
        horizontal=True,
    )
with col_help:
    st.info("Use batch mode to quickly screen many requirements at once.")


st.markdown("---")

if "history" not in st.session_state:
    st.session_state["history"] = []

if "agent_iteration" not in st.session_state:
    st.session_state["agent_iteration"] = 1

if "agent_original_severity" not in st.session_state:
    st.session_state["agent_original_severity"] = None

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
elif mode == "Batch (one per line)":

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
elif mode == "Agent Assist":
    st.subheader("Agent Assist")
    st.caption("The agent analyzes your requirement, suggests a clearer rewrite, and re-evaluates it.")

    original_text = st.text_area("Original requirement", height=140)

    col_agent_buttons = st.columns(2)
    with col_agent_buttons[0]:
        run_agent = st.button("Analyze with Agent", type="primary")
    with col_agent_buttons[1]:
        reset_agent = st.button("Reset agent state")

    if reset_agent:
        st.session_state["agent_iteration"] = 1
        st.session_state["agent_original_severity"] = None

    if run_agent and not original_text.strip():
        st.warning("Please enter a requirement for the agent to analyze.")

    if run_agent and original_text.strip():
        # First agent iteration over the original text
        base_result = analyze_requirement(
            original_text,
            max_length=max_length,
            ml_threshold=ml_threshold,
        )

        st.session_state["agent_original_severity"] = base_result.get("severity", 1)
        st.session_state["agent_iteration"] = 1

        # Generate initial suggestion
        suggestion_text = suggest_rewrite(
            original_text,
            base_result.get("tags", []),
            iteration=st.session_state["agent_iteration"],
        )

        st.session_state["agent_suggestion"] = suggestion_text
        st.session_state["agent_base_result"] = base_result

    if st.session_state.get("agent_base_result") and original_text.strip():
        base_result = st.session_state["agent_base_result"]
        suggestion_text = st.session_state.get("agent_suggestion", original_text)

        # Agent loop indicator
        current_iter = st.session_state["agent_iteration"]
        orig_sev = st.session_state["agent_original_severity"] or base_result.get("severity", 1)
        st.markdown(f"**Agent iteration:** {current_iter} → {current_iter + 1}")
        st.caption(f"Goal: Reduce severity from {orig_sev} → ≤ 1")

        col_orig, col_suggested = st.columns(2)

        with col_orig:
            st.markdown("**Original**")
            status = base_result["status"]
            color = (
                "green" if status == "Clear" else
                "orange" if status == "Partially Clear" else
                "red"
            )
            st.markdown(f"Status: :{color}[{status}]")
            st.caption(f"Severity: {base_result.get('severity', 1)} / 3")
            tags = base_result.get("tags", [])
            if tags:
                st.write("Tags:", " ".join([f"`{t}`" for t in tags]))
            st.markdown(highlight_vague_terms_colored(original_text))

        with col_suggested:
            st.markdown("**Suggested by agent**")
            suggestion_text = st.text_area(
                "Suggested clearer version (you can edit this)",
                value=suggestion_text,
                height=140,
            )
            st.session_state["agent_suggestion"] = suggestion_text
            st.caption(
                "Note: Suggested values are placeholders generated by the agent and should be confirmed by stakeholders."
            )

        # Agent rationale
        st.write("### Agent rationale")
        rationale_bullets = build_agent_rationale(base_result.get("tags", []))
        for r in rationale_bullets:
            st.write(f"- {r}")

        # Re-analyze suggestion
        if st.button("Re-analyze suggested version"):

            st.session_state["agent_iteration"] += 1
            iter_idx = st.session_state["agent_iteration"]

            # Optionally refine suggestion again before analyzing
            refined_suggestion = suggest_rewrite(
                suggestion_text,
                base_result.get("tags", []),
                iteration=iter_idx,
            )
            st.session_state["agent_suggestion"] = refined_suggestion

            suggested_result = analyze_requirement(
                refined_suggestion,
                max_length=max_length,
                ml_threshold=ml_threshold,
            )

            # Comparison panel
            st.write("### Version comparison")
            col_o, col_n = st.columns(2)
            with col_o:
                st.markdown("**Original**")
                st.caption(f"Status: {base_result['status']} | Severity: {base_result.get('severity', 1)} / 3")
            with col_n:
                st.markdown("**Suggestion (current iteration)**")
                st.caption(f"Status: {suggested_result['status']} | Severity: {suggested_result.get('severity', 1)} / 3")

            # Update base_result so next loop can continue from improved state if desired
            st.session_state["agent_base_result"] = suggested_result

# Simple history panel in the sidebar
if st.session_state["history"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent requirements")
    for item in reversed(st.session_state["history"][-5:]):
        st.sidebar.caption(f"{item['Status']}: {item['Requirement'][:40]}...")

st.markdown("---")