# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Đặng  
**Vai trò trong nhóm:** Tech Lead  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi đảm nhiệm vai trò Tech Lead, chủ yếu dẫn dắt Sprint 1 và Sprint 2, đồng thời hỗ trợ kết nối code xuyên suốt 4 sprint.

Ở Sprint 1, tôi implement toàn bộ `index.py`: viết hàm `preprocess_document()` để parse metadata từ header file (Source, Department, Effective Date, Access), viết `chunk_document()` để split theo section heading `=== ... ===` trước rồi dùng `_split_by_size()` để cắt tiếp theo paragraph với overlap 80 tokens, và implement `get_embedding()` sử dụng OpenAI `text-embedding-3-small` để embed toàn bộ 5 tài liệu vào ChromaDB.

Ở Sprint 2, tôi implement `retrieve_dense()`, `build_grounded_prompt()`, `call_llm()` (hỗ trợ cả OpenAI và Gemini), và hàm `rag_answer()` tổng hợp pipeline RAG end-to-end. Công việc của tôi là nền tảng để Retrieval Owner (Sprint 3) và Eval Owner (Sprint 4) cắm vào variant hybrid và chạy scorecard.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

**Chunking theo cấu trúc tự nhiên vs. cắt theo token cứng:** Trước lab, tôi nghĩ chunking chỉ là chia đều theo số token. Sau khi implement và debug `_split_by_size()`, tôi hiểu rằng cắt giữa một điều khoản chính sách có thể làm mất hoàn toàn ngữ nghĩa của điều khoản đó — ví dụ, cắt sau câu "Điều kiện ngoại lệ không hoàn tiền bao gồm:" mà không giữ lại danh sách phía sau sẽ khiến retriever không bao giờ trả lời đúng câu hỏi về sản phẩm kỹ thuật số. Overlap 80 tokens giúp giảm thiểu nhưng không giải quyết triệt để nếu section quá dài.

**Grounded prompt và abstain behavior:** Tôi hiểu rõ hơn tầm quan trọng của rule "Answer only from context" và "Abstain nếu thiếu dữ liệu". Trường hợp q04 (sản phẩm kỹ thuật số) cho thấy ngay cả khi Context Recall = 5 (tức retriever lấy đúng source), model vẫn có thể abstain nếu prompt không đủ rõ hướng dẫn đọc điều khoản ngoại lệ.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều ngạc nhiên nhất là **Context Recall = 5.00/5 ngay từ baseline dense**, nghĩa là retriever đã lấy đúng source cho gần như toàn bộ câu hỏi. Tôi kỳ vọng đây là điểm yếu lớn nhất cần tập trung fix, nhưng thực tế cho thấy bottleneck lại nằm ở **generation**, không phải retrieval. Ba câu q04, q09, q10 đều có recall tốt nhưng Faithfulness/Completeness = 1, vì model hoặc abstain không đúng lúc (q04, q10), hoặc abstain đúng nhưng thiếu gợi ý có ích (q09).

Lỗi mất nhiều thời gian debug nhất là lúc `call_llm()` trả về response có markdown code block (` ```json ... ``` `) thay vì JSON thuần, khiến `json.loads()` fail. Phải thêm bước strip markdown trước khi parse. Giả thuyết ban đầu của tôi là LLM sẽ luôn tuân theo instruction "Output ONLY a JSON object", nhưng thực tế `gpt-4o-mini` vẫn thỉnh thoảng wrap JSON trong code block dù đã nhắc rõ ràng.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q04` — *"Sản phẩm kỹ thuật số có được hoàn tiền không?"*

**Phân tích:**

Đây là câu hỏi thú vị nhất vì nó thể hiện rõ sự tách biệt giữa retrieval tốt và generation sai.

**Baseline:** Faithfulness = 1, Relevance = 1, Context Recall = 5, Completeness = 1. Retriever lấy đúng chunk từ `policy_refund_v4.txt` (recall hoàn hảo), nhưng model abstain ("Xin lỗi, tôi không có đủ dữ liệu") thay vì đọc điều khoản ngoại lệ trong chunk đó. Lỗi nằm ở **generation**: prompt grounding quá chặt khiến model từ chối trả lời khi thông tin nằm trong một câu phụ của điều khoản chứ không phải câu chính.

**Lỗi nằm ở đâu:** Generation. Chunking có thể đã cắt điều khoản ngoại lệ ra một chunk riêng (section "Điều khoản loại trừ"), dẫn đến chunk được retrieve chứa nội dung policy chung nhưng thiếu câu "license key, subscription là ngoại lệ không được hoàn tiền" rõ ràng.

**Variant hybrid:** Không cải thiện — Faithfulness/Relevance/Completeness vẫn = 1. Đúng như dự đoán: vấn đề không phải là retrieval recall mà là reasoning từ ngoại lệ trong context.

**Cải tiến cần thiết:** Thêm instruction trong prompt: *"Pay special attention to exception clauses and negative conditions (e.g., 'Không áp dụng với...', 'Ngoại lệ bao gồm...'). These also constitute valid answers."*

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Tôi sẽ thử hai cải tiến cụ thể:

1. **Cải thiện generation prompt cho negative reasoning:** Eval cho thấy q04 và q10 thất bại dù recall = 5. Tôi sẽ thêm instruction trong `build_grounded_prompt()` để model nhận biết và trả lời dựa trên điều khoản ngoại lệ/phủ định thay vì mặc định abstain.

2. **Query expansion cho alias:** q07 ("Approval Matrix") cho thấy hybrid chưa đủ — BM25 không biết "Approval Matrix" là tên cũ của "Access Control SOP". Tôi sẽ implement `transform_query()` với strategy `expansion` để inject alias vào query trước khi retrieve, vì đây là lỗi có thể fix ở tầng retrieval.

---

*Lưu file này với tên: `reports/individual/dang.md`*
