"""
Streamlit UI for Legal BERT NLP Application
Interactive document processing interface
"""
import streamlit as st
import torch
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from inference.processor import LegalDocumentProcessor, DocumentSummarizer, batch_process_documents
from preprocessing.text_cleaner import TextCleaner


# Page configuration
st.set_page_config(
    page_title="Legal BERT NLP",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Tokens ── */
  :root {
    --navy:    #0c2461;
    --blue:    #1e3a8a;
    --accent:  #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --error:   #ef4444;
    --muted:   #64748b;
    --border:  #e2e8f0;
    --surface: #f8fafc;
  }

  /* ── Global type ── */
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }

  /* ── Top header bar ── */
  .app-header {
    background: linear-gradient(135deg, #0c2461 0%, #1e3a8a 100%);
    color: #fff;
    padding: 2rem 2.5rem 1.75rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .app-header h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -.5px;
    margin: 0;
    line-height: 1.2;
  }
  .app-header p {
    font-size: .95rem;
    color: rgba(255,255,255,.75);
    margin: .35rem 0 0;
  }

  /* ── Section titles ── */
  .sec-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--navy);
    border-left: 4px solid var(--accent);
    padding-left: .75rem;
    margin: .5rem 0 1.25rem;
    line-height: 1.25;
  }

  /* ── Info callout ── */
  .callout {
    background: #eff6ff;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: .85rem 1.1rem;
    font-size: .88rem;
    color: #1e40af;
    margin-bottom: 1.25rem;
    line-height: 1.55;
  }

  /* ── Result card ── */
  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin: .75rem 0;
  }
  .result-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .3rem;
  }
  .result-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--navy);
    line-height: 1.1;
  }
  .result-sub {
    font-size: .82rem;
    color: var(--success);
    font-weight: 600;
    margin-top: .2rem;
  }

  /* ── Entity chip ── */
  .entity-chip {
    display: inline-flex;
    align-items: center;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-size: .8rem;
    font-weight: 600;
    padding: .25rem .7rem;
    border-radius: 100px;
    margin: .2rem;
  }

  /* ── Entity group box ── */
  .entity-group {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: .75rem;
  }
  .entity-group-title {
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--navy);
    margin-bottom: .6rem;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--muted);
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 12px;
    margin: 1.5rem 0;
  }
  .empty-state .es-icon { font-size: 2.5rem; margin-bottom: .75rem; }
  .empty-state h3 { font-size: 1rem; font-weight: 600; color: #374151; margin-bottom: .4rem; }
  .empty-state p  { font-size: .85rem; line-height: 1.55; max-width: 380px; margin: 0 auto; }

  /* ── Status pill ── */
  .pill-success {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    background: #d1fae5;
    color: #065f46;
    font-size: .8rem;
    font-weight: 600;
    padding: .3rem .85rem;
    border-radius: 100px;
    margin-bottom: 1rem;
  }

  /* ── Sidebar tweaks ── */
  [data-testid="stSidebar"] { background: #f8fafc !important; }
  .sidebar-brand {
    background: linear-gradient(135deg, #0c2461, #1e3a8a);
    color: #fff;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 1.25rem;
    font-weight: 700;
    font-size: .95rem;
    text-align: center;
  }
  .sidebar-brand small {
    display: block;
    font-size: .72rem;
    font-weight: 400;
    color: rgba(255,255,255,.7);
    margin-top: .2rem;
  }

  /* ── Hide Streamlit default footer/menu ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sample texts ───────────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Contract": (
        "This Service Agreement (the \"Agreement\") is entered into as of January 1, 2025, "
        "between ABC Corporation, a Delaware corporation (\"Company\"), and John Smith "
        "(\"Consultant\"). The Company hereby engages Consultant to provide software development "
        "services. Consultant shall be paid $150 per hour for services rendered. Either party may "
        "terminate this Agreement upon thirty (30) days written notice. This Agreement shall be "
        "governed by the laws of the State of California."
    ),
    "Court Case": (
        "IN THE UNITED STATES DISTRICT COURT FOR THE SOUTHERN DISTRICT OF NEW YORK. "
        "Plaintiff Jane Doe v. Defendant XYZ Corp., Case No. 2025-CV-01234. "
        "Judge Hon. Robert Williams presiding. Plaintiff alleges breach of contract and "
        "seeks compensatory damages of $500,000. Defendant moves to dismiss under Rule 12(b)(6) "
        "for failure to state a claim upon which relief can be granted."
    ),
    "Statute": (
        "Section 42. Prohibition of Unfair Trade Practices. (a) No person shall engage in "
        "unfair methods of competition or unfair or deceptive acts or practices in commerce. "
        "(b) Any violation of subsection (a) shall be subject to a civil penalty not to exceed "
        "$10,000 per violation. (c) The Attorney General is authorized to bring civil actions "
        "to enforce the provisions of this section."
    ),
    "Appeal": (
        "NOTICE OF APPEAL. Appellant Smith Corp. hereby appeals to the United States Court of "
        "Appeals for the Ninth Circuit from the final judgment entered on December 15, 2024 by "
        "the Honorable District Court Judge Adams in Case No. 2024-CV-5678. The basis for this "
        "appeal is that the lower court erred in granting summary judgment on the antitrust claims "
        "and misapplied the relevant legal standard under Sherman Act Section 1."
    ),
}


@st.cache_resource
def load_models():
    """Load models (cached for performance)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LegalDocumentProcessor(device=device)
    return processor


def main():
    # ── App header ──
    st.markdown(
        "<div class='app-header'>"
        "<h1>⚖️ Legal BERT NLP</h1>"
        "<p>AI-powered analysis for legal documents — Classification · NER · Similarity · Summarization</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    st.sidebar.markdown(
        "<div class='sidebar-brand'>⚖️ Legal BERT NLP<small>Advanced NLP for legal documents</small></div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("#### 🗂️ Analysis Task")
    task = st.sidebar.radio(
        "Select task:",
        ["📄 Document Classification", "🔍 Named Entity Recognition",
         "🔗 Similarity Analysis", "📊 Document Summarization",
         "⚡ Batch Processing"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("#### ⚙️ Display Options")
    show_metrics = st.sidebar.checkbox("Show detailed metrics", value=True,
                                       help="Display word count, reading time, and other stats")
    show_probabilities = st.sidebar.checkbox("Show confidence scores", value=True,
                                             help="Display per-class confidence breakdown")

    st.sidebar.divider()
    st.sidebar.markdown("#### 📋 System Info")
    st.sidebar.markdown(
        f"- **Base model:** `{config.DEFAULT_MODEL}`\n"
        f"- **Framework:** PyTorch + Transformers\n"
        f"- **Hardware:** {'🚀 GPU (CUDA)' if torch.cuda.is_available() else '💻 CPU'}\n"
        f"- **Doc types:** {len(config.CLASSIFICATION_LABELS)}"
    )

    # ── Load models ──
    try:
        with st.spinner("Loading AI models…"):
            processor = load_models()
        st.sidebar.success("✅ Models ready")
    except Exception as e:
        st.sidebar.error(f"Model load error: {str(e)[:60]}")
        st.error(
            "**Unable to load models.** "
            "Run `pip install -r requirements.txt` and restart."
        )
        return

    # ── Task routing ──
    if task == "📄 Document Classification":
        document_classification(processor, show_metrics, show_probabilities)
    elif task == "🔍 Named Entity Recognition":
        named_entity_recognition(processor, show_metrics)
    elif task == "🔗 Similarity Analysis":
        similarity_analysis(processor)
    elif task == "📊 Document Summarization":
        document_summarization(processor, show_metrics)
    elif task == "⚡ Batch Processing":
        batch_processing(processor)


def document_classification(processor, show_metrics, show_probabilities):
    """Document classification interface - identify document type"""
    st.markdown("<div class='sec-title'>📄 Document Classification</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Analyzes your legal document and classifies it into one of "
        f"{len(config.CLASSIFICATION_LABELS)} types (contract, case, appeal, statute) with confidence scoring."
        "</div>",
        unsafe_allow_html=True,
    )

    col_input, col_types = st.columns([3, 1])

    with col_input:
        # Sample text quick-fill
        sample_choice = st.selectbox(
            "💡 Load a sample document (optional):",
            ["— choose a sample —"] + list(SAMPLE_TEXTS.keys()),
            key="clf_sample",
        )
        prefill = SAMPLE_TEXTS.get(sample_choice, "")
        text_input = st.text_area(
            "Paste your legal document:",
            value=prefill,
            height=240,
            placeholder="Paste or type your legal document here…",
            label_visibility="collapsed",
            key="clf_text",
        )

    with col_types:
        st.markdown("**Supported types**")
        for label in config.CLASSIFICATION_LABELS:
            st.markdown(f"• `{label}`")

    # Empty state
    if not text_input.strip():
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>📄</div>"
            "<h3>No document provided yet</h3>"
            "<p>Paste a legal document above or choose a sample to try the classifier.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("🔍 Classify Document", key="classify", use_container_width=True):
        try:
            with st.spinner("Analyzing document…"):
                result = processor.classify_document(text_input, return_proba=show_probabilities)

            st.markdown("<div class='pill-success'>✅ Classification complete</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                confidence_pct = int(result["confidence"] * 100)
                st.markdown(
                    f"<div class='result-card'>"
                    f"<div class='result-label'>Document Type</div>"
                    f"<div class='result-value'>{result['label'].upper()}</div>"
                    f"<div class='result-sub'>↑ {result['confidence']:.1%} confidence</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence_pct}%",
                    delta="High" if confidence_pct >= 80 else ("Fair" if confidence_pct >= 60 else "Low"),
                )
            with col3:
                reliability = "Reliable" if confidence_pct >= 80 else ("Moderate" if confidence_pct >= 60 else "Review")
                st.metric("Reliability", reliability)

            if show_probabilities and "all_scores" in result:
                st.markdown("**Classification scores by type**")
                scores_df = __import__("pandas").DataFrame(
                    [(k, v) for k, v in result["all_scores"].items()],
                    columns=["Document Type", "Confidence"],
                ).sort_values("Confidence", ascending=False)
                scores_df["Confidence"] = scores_df["Confidence"].apply(lambda x: f"{x:.1%}")
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

            if show_metrics:
                st.markdown("**Document statistics**")
                words = text_input.split()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Words", len(words))
                c2.metric("Characters", len(text_input))
                c3.metric("Reading time", f"~{max(1, len(words) // 200)} min")
                c4.metric("Avg word length", f"{sum(len(w) for w in words) / len(words):.1f}")

        except Exception as e:
            st.error(f"**Classification failed.** {str(e)[:120]}")


def named_entity_recognition(processor, show_metrics):
    """Named Entity Recognition interface - extract legal entities"""
    st.markdown("<div class='sec-title'>🔍 Named Entity Recognition</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Identifies and extracts key entities from your legal text — "
        "people, judges, organizations, dates, and legal clauses — with per-entity confidence."
        "</div>",
        unsafe_allow_html=True,
    )

    sample_choice = st.selectbox(
        "💡 Load a sample document (optional):",
        ["— choose a sample —"] + list(SAMPLE_TEXTS.keys()),
        key="ner_sample",
    )
    prefill = SAMPLE_TEXTS.get(sample_choice, "")
    text_input = st.text_area(
        "Paste your legal document:",
        value=prefill,
        height=260,
        placeholder="Paste legal text here to extract named entities…",
        label_visibility="collapsed",
        key="ner_text",
    )

    if not text_input.strip():
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>🔍</div>"
            "<h3>No document provided yet</h3>"
            "<p>Paste legal text above or select a sample. "
            "Example: <em>'John Smith, as represented by Attorney Jane Doe, entered into an agreement with Corp Inc.'</em></p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("🔎 Extract Entities", key="ner", use_container_width=True):
        try:
            with st.spinner("Analyzing entities…"):
                entities = processor.extract_entities(text_input, group_by_type=True)

            st.markdown("<div class='pill-success'>✅ Entity extraction complete</div>", unsafe_allow_html=True)

            if entities:
                total_entities = sum(len(items) for items in entities.values())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Entities", total_entities)
                c2.metric("Entity Types", len(entities))
                c3.metric("Model", "Legal BERT")

                st.markdown("**Extracted entities by type**")
                for entity_type, items in entities.items():
                    if not items:
                        continue
                    with st.expander(f"**{entity_type}** — {len(items)} found", expanded=(len(items) <= 5)):
                        chips_html = "".join(
                            f"<span class='entity-chip'>{item}</span>"
                            for item in items[:15]
                        )
                        if len(items) > 15:
                            chips_html += f"<span class='entity-chip' style='background:#f1f5f9;color:#64748b;'>+{len(items)-15} more</span>"
                        st.markdown(chips_html, unsafe_allow_html=True)

                if show_metrics:
                    st.markdown("**Entity distribution**")
                    import pandas as pd
                    summary_df = __import__("pandas").DataFrame(
                        {"Entity Type": list(entities.keys()), "Count": [len(v) for v in entities.values()]}
                    ).sort_values("Count", ascending=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.bar_chart(summary_df.set_index("Entity Type"))
                    with c2:
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "**No entities found.** The document may not contain standard legal entities, "
                    "or they may be embedded in complex structures. Try documents with clear names, dates, or organizations."
                )

        except Exception as e:
            st.error(f"**Entity extraction failed.** {str(e)[:120]}")


def similarity_analysis(processor):
    """Similarity analysis interface - compare legal documents"""
    st.markdown("<div class='sec-title'>🔗 Similarity Analysis</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Compares legal documents or searches a corpus for similar matches. "
        "Useful for duplicate clause detection, related-case research, or parallel contract review."
        "</div>",
        unsafe_allow_html=True,
    )

    analysis_type = st.radio(
        "Mode:",
        ["Compare Two Documents", "Find Similar in Corpus"],
        horizontal=True,
    )

    if analysis_type == "Compare Two Documents":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Document A**")
            text1 = st.text_area(
                "Document A:",
                height=220,
                placeholder="Paste first legal document…",
                label_visibility="collapsed",
                key="sim_doc1",
            )
        with col2:
            st.markdown("**Document B**")
            text2 = st.text_area(
                "Document B:",
                height=220,
                placeholder="Paste second legal document…",
                label_visibility="collapsed",
                key="sim_doc2",
            )

        if not text1.strip() and not text2.strip():
            st.markdown(
                "<div class='empty-state'>"
                "<div class='es-icon'>🔗</div>"
                "<h3>Enter two documents to compare</h3>"
                "<p>Paste a document in each field above, then click Compare.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if st.button("📊 Compare Documents", key="compare", use_container_width=True):
            if not text1.strip() or not text2.strip():
                st.error("**Both documents required.** Please fill in both fields.")
                return
            try:
                with st.spinner("Calculating similarity…"):
                    similarity = processor.calculate_similarity(text1, text2)

                st.markdown("<div class='pill-success'>✅ Comparison complete</div>", unsafe_allow_html=True)

                pct = similarity * 100
                if similarity > 0.8:
                    match_label, interpretation = "🟢 Very Similar", "Highly similar — likely duplicates or near-identical documents."
                elif similarity > 0.6:
                    match_label, interpretation = "🟡 Moderately Similar", "Significant shared content or structure."
                elif similarity > 0.4:
                    match_label, interpretation = "🟠 Somewhat Similar", "Some overlap but substantially different."
                else:
                    match_label, interpretation = "🔴 Dissimilar", "Largely different in content and structure."

                c1, c2, c3 = st.columns(3)
                c1.metric("Similarity Score", f"{similarity:.3f}", help="0 = completely different · 1 = identical")
                c2.metric("Percentage Match", f"{pct:.1f}%")
                c3.metric("Classification", match_label)
                st.info(f"**Interpretation:** {interpretation}")

            except Exception as e:
                st.error(f"**Similarity calculation failed.** {str(e)[:120]}")

    else:  # Find Similar in Corpus
        st.markdown("**Search query document**")
        query = st.text_area(
            "Query:",
            height=130,
            placeholder="Paste the document you want to find matches for…",
            label_visibility="collapsed",
            key="sim_query",
        )
        st.markdown("**Corpus** — one document per paragraph (blank line between docs)")
        corpus_text = st.text_area(
            "Corpus:",
            height=180,
            placeholder="Document 1\n\nDocument 2\n\nDocument 3…",
            label_visibility="collapsed",
            key="sim_corpus",
        )

        if not query.strip() and not corpus_text.strip():
            st.markdown(
                "<div class='empty-state'>"
                "<div class='es-icon'>🔍</div>"
                "<h3>Enter a query and corpus</h3>"
                "<p>Paste your search document and the corpus to search within, then click Find Similar.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if st.button("🔎 Find Similar Documents", key="find_similar", use_container_width=True):
            if not query.strip():
                st.error("**Query document required.**")
                return
            if not corpus_text.strip():
                st.error("**Corpus required.**")
                return

            corpus_docs = [d.strip() for d in corpus_text.split("\n\n") if d.strip()]
            if len(corpus_docs) == 1:
                corpus_docs = [d.strip() for d in corpus_text.split("\n") if d.strip()]

            try:
                with st.spinner(f"Searching {len(corpus_docs)} documents…"):
                    results = processor.find_similar_documents(query, corpus_docs, top_k=min(5, len(corpus_docs)))

                st.markdown("<div class='pill-success'>✅ Search complete — ranked results below</div>", unsafe_allow_html=True)

                if results:
                    for i, (doc, score) in enumerate(results, 1):
                        quality = "Excellent" if score > 0.8 else ("Good" if score > 0.6 else "Fair")
                        with st.expander(f"**#{i}** — {score:.1%} match · {quality}", expanded=(i == 1)):
                            st.write(doc)
                            ca, cb = st.columns(2)
                            ca.metric("Similarity", f"{score:.3f}")
                            cb.metric("Match Quality", quality)
                else:
                    st.info("**No matches found.** Try adjusting your query or adding more documents to the corpus.")

            except Exception as e:
                st.error(f"**Search failed.** {str(e)[:120]}")


def document_summarization(processor, show_metrics):
    """Document summarization interface - extract key information"""
    st.markdown("<div class='sec-title'>📊 Document Summarization</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Generates a structured summary of your legal document — "
        "document type, key entities, important clauses, and metadata. Works best with 200+ word documents."
        "</div>",
        unsafe_allow_html=True,
    )

    sample_choice = st.selectbox(
        "💡 Load a sample document (optional):",
        ["— choose a sample —"] + list(SAMPLE_TEXTS.keys()),
        key="sum_sample",
    )
    prefill = SAMPLE_TEXTS.get(sample_choice, "")
    text_input = st.text_area(
        "Paste your legal document:",
        value=prefill,
        height=260,
        placeholder="Paste legal document text here…",
        label_visibility="collapsed",
        key="sum_text",
    )

    if not text_input.strip():
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>📊</div>"
            "<h3>No document provided yet</h3>"
            "<p>Paste a legal document above or select a sample to generate an intelligent summary.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("📋 Generate Summary", key="summarize", use_container_width=True):
        try:
            with st.spinner("Generating intelligent summary…"):
                summarizer = DocumentSummarizer(processor)
                summary = summarizer.get_document_summary(text_input)

            st.markdown("<div class='pill-success'>✅ Summary generated</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Document Type", summary["type"].upper())
            c2.metric("Confidence", f"{summary['confidence']:.0%}")
            c3.metric("Word Count", summary["word_count"])
            read_time = summary.get("estimated_reading_time_minutes", max(1, summary["word_count"] // 200))
            c4.metric("Reading Time", f"~{read_time} min")

            if summary.get("key_entities"):
                st.markdown("**Key entities identified**")
                for entity_type, ents in summary["key_entities"].items():
                    if ents:
                        chips = "".join(f"<span class='entity-chip'>{e}</span>" for e in ents[:5])
                        extra = f" <span class='entity-chip' style='background:#f1f5f9;color:#64748b;'>+{len(ents)-5} more</span>" if len(ents) > 5 else ""
                        st.markdown(
                            f"<div class='entity-group'>"
                            f"<div class='entity-group-title'>{entity_type}</div>"
                            f"{chips}{extra}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            if summary.get("key_clauses"):
                st.markdown("**Key clauses & provisions**")
                for i, clause in enumerate(summary["key_clauses"][:5], 1):
                    with st.expander(f"Clause {i}", expanded=(i == 1)):
                        st.write(clause[:300] + ("…" if len(clause) > 300 else ""))
                if len(summary.get("key_clauses", [])) > 5:
                    st.caption(f"+{len(summary['key_clauses']) - 5} additional clauses identified. Showing top 5.")

            if show_metrics:
                st.markdown("**Analysis details**")
                ca, cb = st.columns(2)
                ca.metric("Unique Entity Types", len(summary.get("key_entities", {})))
                cb.metric("Key Clauses Found", len(summary.get("key_clauses", [])))

        except Exception as e:
            st.error(f"**Summarization failed.** {str(e)[:120]}")


def batch_processing(processor):
    """Batch processing interface - process multiple documents efficiently"""
    st.markdown("<div class='sec-title'>⚡ Batch Processing</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Process large numbers of legal documents at once. "
        "Upload a plain-text file (one document per line) and run classification, NER, or summarization across all of them."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a .txt file — one document per line:",
        type=["txt"],
        help="Plain-text file with legal documents, one per line",
    )

    if not uploaded_file:
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>⚡</div>"
            "<h3>No file uploaded yet</h3>"
            "<p>Upload a plain-text file with one legal document per line to start batch processing.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        content = uploaded_file.read().decode("utf-8")
        documents = [line.strip() for line in content.split("\n") if line.strip()]

        if not documents:
            st.error("**File is empty.** Please upload a file with at least one document.")
            return

        st.success(f"✅ Loaded **{len(documents)}** documents")

        st.markdown("**Processing configuration**")
        task = st.selectbox(
            "Select task:",
            ["📄 Classify All Documents", "🔍 Extract All Entities", "📊 Summarize All Documents"],
        )
        batch_size = st.slider("Batch size (documents per batch)", 1, 50, 10)

        if st.button("⚡ Start Batch Processing", key="batch", use_container_width=True):
            task_map = {
                "📄 Classify All Documents": "classify",
                "🔍 Extract All Entities": "extract_entities",
                "📊 Summarize All Documents": "summarize",
            }

            try:
                with st.spinner(f"Processing {len(documents)} documents…"):
                    results = batch_process_documents(
                        documents, processor, task=task_map[task], batch_size=batch_size
                    )

                st.markdown(
                    f"<div class='pill-success'>✅ Processed {len(results)} documents successfully</div>",
                    unsafe_allow_html=True,
                )

                import pandas as pd
                import json

                if task_map[task] == "classify":
                    st.markdown("**Classification results**")
                    df = pd.DataFrame([
                        {
                            "Document": d[:60] + "…" if len(d) > 60 else d,
                            "Type": r.get("label", "Unknown").upper(),
                            "Confidence": f"{r.get('confidence', 0):.1%}",
                        }
                        for d, r in zip(documents[: len(results)], results)
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    c1, c2, c3 = st.columns(3)
                    labels = [r.get("label") for r in results]
                    most_common = max(set(labels), key=labels.count)
                    avg_conf = sum(r.get("confidence", 0) for r in results) / len(results)
                    c1.metric("Most Common Type", most_common)
                    c2.metric("Avg Confidence", f"{avg_conf:.1%}")
                    c3.metric("Documents Processed", len(results))

                st.markdown("**Export results**")
                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_data = pd.DataFrame(results).to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_data,
                        file_name=f"batch_{task_map[task]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_json:
                    json_data = json.dumps(results, indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        data=json_data,
                        file_name=f"batch_{task_map[task]}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"**Batch processing failed.** {str(e)[:120]}")

    except Exception as e:
        st.error(f"**File reading error.** Ensure it is a valid UTF-8 text file. {str(e)[:120]}")


if __name__ == "__main__":
    try:
        import pandas as pd
        main()
    except Exception as e:
        error_msg = str(e)
        st.error(
            f"**❌ Application Error.** An unexpected error occurred. "
            f"Details: {error_msg[:100]}"
        )
        st.info(
            "**How to fix this:**\n"
            "1. Make sure all dependencies are installed: `pip install -r requirements.txt`\n"
            "2. Check that PyTorch is installed: `python -c \"import torch; print(torch.__version__)\"`\n"
            "3. Verify transformers library: `python -c \"import transformers; print(transformers.__version__)\"`\n"
            "4. If using CPU, the app may run slower - consider using a GPU for better performance"
        )
        
        # Log the full traceback for debugging
        import traceback
        import logging
        logging.error(f"Full error trace:\n{traceback.format_exc()}")
