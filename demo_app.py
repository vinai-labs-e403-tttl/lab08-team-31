"""
demo_app.py - Streamlit demo UI for RAG pipeline
================================================
Demo app phục vụ 3 nhu cầu:
1. Hỏi đáp trực tiếp với pipeline RAG
2. Chạy evaluation trên bộ test questions
3. Xem scorecard/corpus nhanh để demo và đánh giá
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from eval import (
    BASELINE_CONFIG,
    VARIANT_CONFIG,
    generate_scorecard_summary,
    run_scorecard,
)
from index import CHROMA_DB_DIR, DOCS_DIR, chunk_document, preprocess_document
from rag_answer import rag_answer


TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_test_questions() -> List[Dict[str, Any]]:
    if not TEST_QUESTIONS_PATH.exists():
        return []
    return json.loads(TEST_QUESTIONS_PATH.read_text(encoding="utf-8"))


def load_existing_comparison_rows() -> List[Dict[str, str]]:
    csv_path = RESULTS_DIR / "ab_comparison.csv"
    if not csv_path.exists():
        return []

    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_metric(rows: List[Dict[str, Any]], metric: str) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(metric)
        if value in (None, "", "None"):
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def get_corpus_stats() -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(path))
        chunks = chunk_document(doc)
        meta = doc["metadata"]
        stats.append(
            {
                "file": path.name,
                "source": meta.get("source", ""),
                "department": meta.get("department", ""),
                "effective_date": meta.get("effective_date", ""),
                "chunks": len(chunks),
            }
        )
    return stats


def metric_label(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}/5"


def render_chunk_card(index: int, chunk: Dict[str, Any]) -> None:
    meta = chunk.get("metadata", {})
    title = f"Chunk {index}"
    caption = (
        f"Source: {meta.get('source', 'unknown')} | "
        f"Section: {meta.get('section', '') or 'N/A'} | "
        f"Score: {chunk.get('score', 0):.3f}"
    )
    with st.expander(title, expanded=index == 1):
        st.caption(caption)
        st.write(chunk.get("text", ""))


def render_existing_scorecard_tab() -> None:
    st.subheader("Kết quả đã có trong thư mục results/")
    rows = load_existing_comparison_rows()
    if not rows:
        st.info("Chưa tìm thấy `results/ab_comparison.csv`.")
        return

    baseline_rows = [r for r in rows if r.get("config_label") == "baseline_dense"]
    variant_rows = [r for r in rows if r.get("config_label") == "variant_hybrid_rerank"]

    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    cols = st.columns(4)
    for idx, metric in enumerate(metrics):
        b_avg = summarize_metric(baseline_rows, metric)
        v_avg = summarize_metric(variant_rows, metric)
        delta = None if b_avg is None or v_avg is None else v_avg - b_avg
        cols[idx].metric(
            metric.replace("_", " ").title(),
            metric_label(v_avg),
            None if delta is None else f"{delta:+.2f} vs baseline",
        )

    st.dataframe(rows, use_container_width=True)


def render_corpus_tab() -> None:
    st.subheader("Corpus đã index")
    stats = get_corpus_stats()
    total_chunks = sum(item["chunks"] for item in stats)

    info_cols = st.columns(3)
    info_cols[0].metric("Số tài liệu", len(stats))
    info_cols[1].metric("Tổng chunks", total_chunks)
    info_cols[2].metric("Chroma DB", "Ready" if CHROMA_DB_DIR.exists() else "Missing")

    st.dataframe(stats, use_container_width=True)


def render_live_qa_tab() -> None:
    st.subheader("Hỏi đáp trực tiếp với pipeline RAG")
    st.write(
        "Chọn cấu hình retrieval, nhập câu hỏi và xem ngay câu trả lời, source, "
        "cùng các chunk được dùng để generate."
    )

    sample_questions = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "Approval Matrix để cấp quyền hệ thống là tài liệu nào?",
        "Nếu cần hoàn tiền khẩn cấp cho khách hàng VIP, quy trình có khác không?",
    ]

    selected_sample = st.selectbox("Câu hỏi mẫu", [""] + sample_questions)
    query = st.text_area(
        "Câu hỏi",
        value=selected_sample,
        height=110,
        placeholder="Nhập câu hỏi để thử hệ thống...",
    )

    left, mid, right = st.columns(3)
    retrieval_mode = left.selectbox(
        "Retrieval mode",
        options=["dense", "hybrid", "sparse"],
        index=0,
    )
    top_k_search = mid.slider("Top-k search", min_value=3, max_value=15, value=10)
    top_k_select = right.slider("Top-k select", min_value=1, max_value=5, value=3)
    use_rerank = st.checkbox(
        "Bật rerank/select theo score hiện có",
        value=(retrieval_mode == "hybrid"),
    )

    if st.button("Chạy truy vấn", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Hãy nhập câu hỏi trước khi chạy.")
            return

        with st.spinner("Đang retrieve và generate câu trả lời..."):
            try:
                result = rag_answer(
                    query=query.strip(),
                    retrieval_mode=retrieval_mode,
                    top_k_search=top_k_search,
                    top_k_select=top_k_select,
                    use_rerank=use_rerank,
                    verbose=False,
                )
            except Exception as exc:
                st.error(f"Không chạy được pipeline: {exc}")
                return

        st.success("Đã tạo xong câu trả lời.")
        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Sources")
        for source in result.get("sources", []):
            st.code(source)

        st.markdown("### Retrieved Chunks")
        for idx, chunk in enumerate(result.get("chunks_used", []), start=1):
            render_chunk_card(idx, chunk)


def render_eval_tab() -> None:
    st.subheader("Chạy evaluation trên bộ test")
    questions = load_test_questions()
    if not questions:
        st.warning("Không tìm thấy `data/test_questions.json`.")
        return

    st.write(
        "Tab này dùng để chạy nhanh scorecard từ UI. "
        "Bạn có thể chọn baseline, variant hoặc tự cấu hình một run mới."
    )

    preset = st.radio(
        "Preset config",
        options=["Baseline", "Variant", "Custom"],
        horizontal=True,
    )

    if preset == "Baseline":
        config = dict(BASELINE_CONFIG)
    elif preset == "Variant":
        config = dict(VARIANT_CONFIG)
    else:
        c1, c2, c3 = st.columns(3)
        config = {
            "retrieval_mode": c1.selectbox("Retrieval mode", ["dense", "hybrid", "sparse"], index=0),
            "top_k_search": c2.slider("Top-k search", 3, 15, 10),
            "top_k_select": c3.slider("Top-k select", 1, 5, 3),
            "use_rerank": st.checkbox("Use rerank", value=False),
            "label": st.text_input("Config label", value="custom_demo_run"),
        }

    if preset != "Custom":
        st.code(json.dumps(config, ensure_ascii=False, indent=2), language="json")

    selected_ids = st.multiselect(
        "Chọn câu hỏi để chạy",
        options=[q["id"] for q in questions],
        default=[q["id"] for q in questions],
        format_func=lambda qid: next(
            (f"{qid} - {q['question']}" for q in questions if q["id"] == qid),
            qid,
        ),
    )

    if st.button("Chạy scorecard", use_container_width=True):
        chosen = [q for q in questions if q["id"] in selected_ids]
        if not chosen:
            st.warning("Hãy chọn ít nhất một câu hỏi.")
            return

        with st.spinner("Đang chạy scorecard..."):
            try:
                rows = run_scorecard(config=config, test_questions=chosen, verbose=False)
            except Exception as exc:
                st.error(f"Evaluation thất bại: {exc}")
                return

        st.success(f"Đã chạy xong {len(rows)} câu hỏi.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Faithfulness", metric_label(summarize_metric(rows, "faithfulness")))
        m2.metric("Relevance", metric_label(summarize_metric(rows, "relevance")))
        m3.metric("Context Recall", metric_label(summarize_metric(rows, "context_recall")))
        m4.metric("Completeness", metric_label(summarize_metric(rows, "completeness")))

        st.dataframe(rows, use_container_width=True)

        summary_md = generate_scorecard_summary(rows, config.get("label", "ui_run"))
        st.markdown("### Scorecard Summary")
        st.markdown(summary_md)


def main() -> None:
    st.set_page_config(
        page_title="RAG Pipeline Demo",
        page_icon="📚",
        layout="wide",
    )

    st.title("RAG Pipeline Demo")
    st.caption("Demo UI cho truy vấn, evaluation và xem scorecard của lab Day 08.")

    intro_left, intro_right = st.columns([2, 1])
    with intro_left:
        st.write(
            "Ứng dụng này bọc trực tiếp các hàm trong `index.py`, `rag_answer.py` và `eval.py` "
            "để phục vụ demo trước lớp hoặc kiểm tra nhanh chất lượng pipeline."
        )
    with intro_right:
        st.info(
            "Yêu cầu: đã có `chroma_db/` và API key hợp lệ trong `.env` "
            "nếu muốn chạy phần hỏi đáp hoặc evaluation."
        )

    tab_qa, tab_eval, tab_results, tab_corpus = st.tabs(
        ["Live QA", "Evaluation", "Existing Results", "Corpus"]
    )

    with tab_qa:
        render_live_qa_tab()
    with tab_eval:
        render_eval_tab()
    with tab_results:
        render_existing_scorecard_tab()
    with tab_corpus:
        render_corpus_tab()


if __name__ == "__main__":
    main()
