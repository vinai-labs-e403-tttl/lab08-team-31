"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2: Dense retrieval + grounded answer với citation
Sprint 3: Hybrid retrieval (Dense + BM25) với Reciprocal Rank Fusion
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store (search rộng)
TOP_K_SELECT = 3     # Số chunk đưa vào prompt (top-3)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Ngưỡng score tối thiểu để trả lời (dưới ngưỡng → abstain)
MIN_SCORE_THRESHOLD = 0.25


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

_chroma_collection = None

def _get_collection():
    """Cache ChromaDB collection để không khởi tạo lại nhiều lần."""
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        from index import CHROMA_DB_DIR
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _chroma_collection = client.get_collection("rag_lab")
    return _chroma_collection


def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Returns list chunks, mỗi chunk có:
      - "text": nội dung
      - "metadata": source, section, department, effective_date, access
      - "score": cosine similarity (1 - distance)
    """
    from index import get_embedding

    collection = _get_collection()
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: score = 1 - distance
        score = max(0.0, 1.0 - dist)
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": score,
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

_bm25_index = None
_bm25_chunks = None

def _build_bm25_index():
    """Build BM25 index từ toàn bộ chunks trong ChromaDB (cache sau lần đầu)."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    from rank_bm25 import BM25Okapi

    collection = _get_collection()
    results = collection.get(include=["documents", "metadatas"])

    _bm25_chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        _bm25_chunks.append({"text": doc, "metadata": meta, "score": 0.0})

    # Tokenize: lowercase + split (đủ dùng cho tiếng Việt không dấu + tiếng Anh)
    tokenized = [chunk["text"].lower().split() for chunk in _bm25_chunks]
    _bm25_index = BM25Okapi(tokenized)

    return _bm25_index, _bm25_chunks


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên viết tắt (P1, SLA, ERR-403...)
    """
    bm25, all_chunks = _build_bm25_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Normalize scores về [0, 1]
    max_score = max(scores) if max(scores) > 0 else 1.0
    normalized = [s / max_score for s in scores]

    # Lấy top_k theo score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        chunk = all_chunks[idx].copy()
        chunk["score"] = normalized[idx]
        results.append(chunk)

    return results


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# Sprint 3 Variant — lý do chọn: corpus lẫn lộn câu tự nhiên (tiếng Việt)
# và keyword kỹ thuật (P1, SLA, ERR-403, Level 3...) nên hybrid phù hợp nhất
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF_score(doc) = dense_weight * 1/(60 + dense_rank)
                   + sparse_weight * 1/(60 + sparse_rank)

    Lý do chọn Hybrid (Sprint 3):
    - Corpus có cả câu tiếng Việt tự nhiên VÀ keyword kỹ thuật (P1, SLA, Level 3)
    - Dense tốt cho paraphrase: "bao lâu" → "resolution time"
    - Sparse tốt cho exact match: "ERR-403", "Flash Sale", "store credit"
    - RRF không cần tune score scale giữa hai hệ thống
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # Tạo dict text → chunk để merge (dùng text đầu 100 ký tự làm key)
    def chunk_key(chunk):
        return chunk["text"][:100]

    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    # Cộng điểm RRF từ dense
    for rank, chunk in enumerate(dense_results):
        key = chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + dense_weight * (1 / (60 + rank + 1))
        chunk_map[key] = chunk

    # Cộng điểm RRF từ sparse
    for rank, chunk in enumerate(sparse_results):
        key = chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + sparse_weight * (1 / (60 + rank + 1))
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort theo RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_map[key].copy()
        chunk["score"] = rrf_scores[key] * 100  # Scale để dễ đọc
        results.append(chunk)

    return results


# =============================================================================
# RERANK (Sprint 3 — không chọn variant này nhưng vẫn có fallback)
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """Rerank đơn giản: trả về top_k theo score hiện tại."""
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_candidates[:top_k]

# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde": Sinh câu trả lời giả (hypothetical document) để embed thay query

    TODO Sprint 3 (nếu chọn query transformation):
    Gọi LLM với prompt phù hợp với từng strategy.

    Ví dụ expansion prompt:
        "Given the query: '{query}'
         Generate 2-3 alternative phrasings or related terms in Vietnamese.
         Output as JSON array of strings."

    Ví dụ decomposition:
        "Break down this complex query into 2-3 simpler sub-queries: '{query}'
         Output as JSON array."

    Khi nào dùng:
    - Expansion: query dùng alias/tên cũ (ví dụ: "Approval Matrix" → "Access Control SOP")
    - Decomposition: query hỏi nhiều thứ một lúc
    - HyDE: query mơ hồ, search theo nghĩa không hiệu quả
    """
    # TODO Sprint 3: Implement query transformation
    # Tạm thời trả về query gốc
    import json
    
    if strategy == "expansion":
        prompt = f"""Given the query: '{query}'
Generate 2-3 alternative phrasings or related terms in Vietnamese to help with semantic search retrieval.
Output ONLY a JSON array of strings, strictly with no markdown formatting or other text."""
    elif strategy == "decomposition":
        prompt = f"""Break down this complex query into 2 simpler sub-queries in Vietnamese: '{query}'
Output ONLY a JSON array of strings, strictly with no markdown formatting or other text."""
    elif strategy == "hyde":
        prompt = f"""Write a hypothetical short paragraph (2-3 sentences) that directly answers this query in Vietnamese: '{query}'
This snippet will be used for Dense similarity search, so simulate the language and keywords typical of formal internal company policy documents.
Output ONLY a JSON array containing exactly one single string (the paragraph), with no markdown."""
    else:
        return [query]

    try:
        # Gọi hàm LLM đã config ở dưới
        response = call_llm(prompt).strip()
        
        # Tiền xử lý xóa markdown code block nếu LLM lỡ sinh ra
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
            
        transformed_list = json.loads(response.strip())
        
        if isinstance(transformed_list, list) and len(transformed_list) > 0:
            if strategy in ["expansion", "decomposition"]:
                # Giữ cả câu hỏi gốc và kết hợp với mảng mở rộng
                return [query] + transformed_list
            return transformed_list  # Cho HyDE, mảng vốn chỉ có 1 tài liệu giả lập
            
    except NotImplementedError:
        print("[transform_query] Cảnh báo: Hàm call_llm() chưa được cài đặt. Fallback về query gốc.")
    except Exception as e:
        print(f"[transform_query] Lỗi parsing hay LLM call: {e}. Fallback về query gốc.")
        
    return [query]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # Trích xuất thêm các metadata quan trọng đã làm ở Sprint 1
        department = meta.get("department", "")
        effective_date = meta.get("effective_date", "")

        header = f"[{i}] Root: {source}"
        
        # Gắn chuỗi thông tin phụ trợ bổ sung
        extras = []
        if section: extras.append(f"Section: {section}")
        if department: extras.append(f"Dept: {department}")
        if effective_date: extras.append(f"Date: {effective_date}")
        if score > 0: extras.append(f"Score: {score:.2f}")

        if extras:
            header += " | " + " - ".join(extras)

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    TODO Sprint 2:
    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""You are a professional and helpful AI Assistant for the Internal CS & IT Helpdesk.
Your task is to answer the user's question based strictly on the retrieved context below.

CRITICAL RULES:
1. Evidence-only: Answer ONLY using the provided context. Do NOT use outside knowledge.
2. Abstain: If the context does not contain sufficient information to answer the question, firmly say "Xin lỗi, tôi không có đủ dữ liệu hiện tại để trả lời câu hỏi này." and stop. Do not make up information.
3. Citation: You MUST cite your sources using the bracket format (e.g., [1], [2]) at the end of the relevant facts or sentences.
4. Format: Present your answer clearly. Use bullet points or numbered lists if explaining a multi-step process or multiple conditions.
5. Tone & Language: Your tone should be polite, professional, and helpful. Always respond in Vietnamese (unless the user explicitly inputs a different language).

User Question: {query}

Retrieved Context:
{context_block}

Final Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    TODO Sprint 2:
    Chọn một trong hai:

    Option A — OpenAI (cần OPENAI_API_KEY):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,     # temperature=0 để output ổn định, dễ đánh giá
            max_tokens=512,
        )
        return response.choices[0].message.content

    Option B — Google Gemini (cần GOOGLE_API_KEY):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    Lưu ý: Dùng temperature=0 hoặc thấp để output ổn định cho evaluation.
    """
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GOOGLE_API_KEY")

    if api_key_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key_openai)
            response = client.chat.completions.create(
                model=LLM_MODEL,  # Ví dụ: gpt-4o-mini hoặc gpt-4o
                messages=[{"role": "user", "content": prompt}],
                temperature=0,    # 0 để RAG trả lời nghiêm túc, không chế chữ
                max_tokens=600,
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("Vui lòng cài đặt OpenAI bằng lệnh: pip install openai")
            
    elif api_key_gemini:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key_gemini)
            
            # Cấu hình tự động chuyển model gpt sang gemini nếu đang dùng biến môi trường mặc định
            model_name = "gemini-1.5-flash" if "gpt" in LLM_MODEL else LLM_MODEL
            model = genai.GenerativeModel(model_name)
            
            # Khởi tạo thông số chuẩn xác GenerationConfig để set temperature cho gemini
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return response.text
        except ImportError:
            raise ImportError("Vui lòng cài đặt Gemini bằng lệnh: pip install google-generativeai")
            
    else:
        raise ValueError("[LỖI] Không tìm thấy OPENAI_API_KEY hoặc GOOGLE_API_KEY trong file .env. Vui lòng thiết lập để kết nối với bộ não LLM!")


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    TODO Sprint 2 — Implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    TODO Sprint 3 — Thử các variant:
    - Variant A: đổi retrieval_mode="hybrid"
    - Variant B: bật use_rerank=True
    - Variant C: thêm query transformation trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid", "sparse"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError:
            print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    # Uncomment sau khi Sprint 3 hoàn thành:
    print("\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nViệc cần làm Sprint 2:")
    print("  1. Implement retrieve_dense() — query ChromaDB")
    print("  2. Implement call_llm() — gọi OpenAI hoặc Gemini")
    print("  3. Chạy rag_answer() với 3+ test queries")
    print("  4. Verify: output có citation không? Câu không có docs → abstain không?")

    print("\nViệc cần làm Sprint 3:")
    print("  1. Chọn 1 trong 3 variants: hybrid, rerank, hoặc query transformation")
    print("  2. Implement variant đó")
    print("  3. Chạy compare_retrieval_strategies() để thấy sự khác biệt")
    print("  4. Ghi lý do chọn biến đó vào docs/tuning-log.md")
