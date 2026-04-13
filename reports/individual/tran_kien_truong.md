# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Kiên Trường - 2A202600496
**Vai trò trong nhóm:** Eval Owner
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi đảm nhận vai trò **Eval Owner** và tập trung hoàn thành Sprint 4 — Evaluation & Scorecard.

Cụ thể, tôi đã implement các hàm chấm điểm tự động bằng phương pháp **LLM-as-Judge** trong `eval.py`:
- **`score_faithfulness`**: Đánh giá câu trả lời có bám sát vào retrieved context hay không
- **`score_answer_relevance`**: Đánh giá câu trả lời có đúng trọng tâm câu hỏi người dùng không
- **`score_completeness`**: So sánh câu trả lời thực tế với expected_answer để xác định các điểm còn thiếu

Công việc của tôi kết nối trực tiếp với phần Sprint 2 (Retrieval Owner đã implement `retrieve_dense`) và Sprint 3 (Retrieval Owner đã implement `retrieve_hybrid`). Khi hai sprint đó hoàn thành, tôi chạy `run_scorecard()` để đánh giá và so sánh baseline vs variant qua `compare_ab()`.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

### Chunking và Groundness

Trước lab, tôi nghĩ chunking chỉ là cắt văn bản theo kích thước cố định. Thực tế, chunking tốt phải bảo toàn ranh giới tự nhiên (heading, paragraph) để mỗi chunk giữ được ngữ cảnh hoàn chỉnh. Nếu chunk cắt giữa một điều khoản, retrieval sẽ lấy được chunk đó nhưng generation vẫn không trả lời đúng vì thiếu thông tin quan trọng.

### LLM-as-Judge cho RAG Evaluation

Tôi hiểu rằng đánh giá RAG không chỉ là accuracy. Ba metric riêng biệt cần đo: **Faithfulness** (trả lời có bám context không), **Relevance** (trả lời có đúng câu hỏi không), và **Completeness** (trả lời có đầy đủ không). LLM-as-Judge giúp tự động hóa việc chấm điểm thay vì phải thủ công đọc từng câu trả lời.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

### Khó khăn: Xử lý JSON từ LLM

Khi implement LLM-as-Judge, LLM đôi khi trả về JSON kèm markdown code block (ví dụ: ```json ... ```). Tôi phải thêm bước tiền xử lý để strip các formatting này trước khi parse JSON. Đây là lỗi tưởng nhỏ nhưng nếu không xử lý sẽ khiến toàn bộ scoring thất bại.

### Ngạc nhiên: Hybrid không cải thiện nhiều như kỳ vọng

Tôi kỳ vọng variant hybrid (BM25 + dense) sẽ cải thiện rõ rệt, đặc biệt với các câu hỏi chứa alias như "Approval Matrix" (q07). Kết quả thực tế cho thấy hybrid chỉ tăng Answer Relevance lên 3.80 (so với baseline 3.70) nhưng Faithfulness lại giảm từ 3.70 xuống 3.50. Điều này dạy tôi rằng: **recall cao hơn không đồng nghĩa với faithfulness cao hơn** — nếu model nhận thêm nhiều context "na ná", nó có xu hướng diễn giải thêm thay vì answer đúng trọng tâm.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** q07 — "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

**Phân tích:**

Baseline (dense) cho q07 đạt điểm recall cao vì chunk chứa thông tin về "Access Control SOP" đã được retrieve. Tuy nhiên, Faithfulness chỉ đạt 2/5 vì model trả lời dài hơn cần thiết, đề cập thêm chi tiết về quyền Level 1, Level 2 mà không có trong chunk được chọn. Expected answer chỉ cần một câu ngắn: *"Tài liệu 'Approval Matrix for System Access' có tên mới là 'Access Control SOP'"*.

Variant hybrid (dense + BM25) cải thiện Answer Relevance lên 5/5 vì BM25 giỏi match exact term "Approval Matrix", nhưng Faithfulness lại giảm xuống 2/5 do model tiếp tục diễn giải quá mức. Lỗi ở đây nằm ở **generation/prompt** hơn là retrieval — context đã đúng nhưng prompt không hạn chế model trả lời ngắn gọn.

**Kết luận:** Lỗi chủ yếu thuộc về generation. Hybrid giúp retrieve đúng tài liệu hơn (query alias match được), nhưng đổi lại làm tăng nhiễu từ các chunk khác dẫn đến model suy diễn thêm. Biến cần thay đổi tiếp theo là **rerank mạnh hơn** để chọn chunk trọng tâm nhất, hoặc **prompt engineering** để yêu cầu trả lời ngắn gọn hơn.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ thử:

1. **Implement cross-encoder rerank thật sự** cho top-10 candidates thay vì chỉ sort theo score hiện có. Kết quả eval cho thấy q06 bị kéo context nhầm sang Access Control SOP thay vì SLA, nghĩa là cần rerank để boost chunk có đúng keyword "P1 escalation" lên trên.

2. **Tinh chỉnh abstain prompt** cho những câu hỏi dạng "không có chính sách riêng" (q04, q10). Hiện tại model abstain quá sớm thay vì suy luận từ evidence gián tiếp rằng "không có quy trình VIP → dùng quy trình tiêu chuẩn".

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
