# Tuning Log - RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13  
**Config:**
```python
retrieval_mode = "dense"
chunk_size = 400
overlap = 80
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 3.70 /5 |
| Answer Relevance | 3.70 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 3.90 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
> `q04` (Refund - sản phẩm kỹ thuật số) có Faithfulness/Relevance/Completeness = 1/1/1 dù recall = 5, cho thấy retriever lấy đúng source nhưng generation vẫn abstain thay vì đọc được mục ngoại lệ không hoàn tiền.  
> `q10` (Refund - VIP urgent refund) cũng có 1/1/1 dù recall = 5, vì model không suy ra được rằng tài liệu không có quy trình riêng cho VIP và quy trình chuẩn vẫn là 3-5 ngày làm việc.  
> `q09` (ERR-403-AUTH) là case không có dữ liệu trong docs; hệ thống abstain đúng tinh thần grounding nhưng bị score thấp do expected answer muốn có thêm gợi ý "có thể liên quan authentication" và "liên hệ IT Helpdesk".

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít -> thiếu evidence
- [x] Generation: Prompt không đủ grounding cho câu hỏi cần suy luận phủ định
- [ ] Generation: Context quá dài -> lost in the middle

**Đánh giá baseline:**
> Baseline dense khá ổn cho phần lớn câu hỏi factoid trực tiếp: 7/10 câu cho kết quả tốt và context recall đạt tối đa. Điểm yếu lớn nhất không nằm ở việc tìm source, mà ở việc biến evidence thành câu trả lời đúng khi câu hỏi đòi đọc ngoại lệ, tên cũ/tên mới, hoặc suy luận "không có chính sách riêng".

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13  
**Biến thay đổi:** Đổi retrieval từ `dense` sang `hybrid` (dense + BM25 + RRF), đồng thời giữ `top_k_search = 10`, `top_k_select = 3`. `use_rerank = True` trong code hiện tại nhưng thực chất chỉ sort theo score hiện có, không phải cross-encoder rerank thực sự.  
**Lý do chọn biến này:**
> Baseline có dấu hiệu yếu ở các query chứa alias hoặc exact keyword, đặc biệt `q07` với cụm "Approval Matrix" là tên cũ của `Access Control SOP`. Corpus cũng trộn giữa policy ngôn ngữ tự nhiên và các keyword chuyên ngành như `P1`, `SLA`, `Level 3`, `license key`, nên hybrid retrieval là biến hợp lý nhất để thử tăng recall cho alias và mã/term đặc thù.

**Config thay đổi:**
```python
retrieval_mode = "hybrid"
chunk_size = 400
overlap = 80
top_k_search = 10
top_k_select = 3
use_rerank = True
llm_model = "gpt-4o-mini"
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 3.70/5 | 3.50/5 | -0.20 |
| Answer Relevance | 3.70/5 | 3.80/5 | +0.10 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.90/5 | 3.40/5 | -0.50 |

**Nhận xét:**
> Variant 1 không cải thiện rõ rệt so với baseline. `q07` đáng ra là câu hybrid có thể giúp nhiều nhất vì liên quan alias "Approval Matrix", nhưng thực tế relevance tăng lên 5 còn faithfulness giảm xuống 2 do model diễn giải thêm ngoài context thay vì trả lời ngắn gọn rằng đây là tên cũ của `Access Control SOP`.  
> `q06` cũng kém hơn baseline ở completeness: hybrid kéo context thiên về quy trình escalation quyền truy cập khẩn cấp trong `Access Control SOP`, trong khi expected answer cần đúng câu "P1 tự động escalate lên Senior Engineer sau 10 phút" từ tài liệu SLA.  
> Các câu `q04`, `q09`, `q10` gần như không đổi, nghĩa là hybrid retrieval không giải quyết được nhóm lỗi thuộc về reasoning/generation.

**Kết luận:**
> Variant 1 không tốt hơn baseline trên bộ test hiện tại. Bằng chứng là Faithfulness giảm từ 3.70 xuống 3.50 và Completeness giảm mạnh hơn, từ 3.90 xuống 3.40, trong khi Context Recall giữ nguyên 5.00/5. Điều này cho thấy hệ thống không thiếu recall tổng quát; vấn đề lớn hơn là chọn chunk đúng trọng tâm và hạn chế model suy diễn từ các chunk "na ná" liên quan.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** Rerank thật sự theo query-chunk relevance hoặc query transformation cho alias  
**Config:**
```python
# Đề xuất:
# retrieval_mode = "dense" hoặc "hybrid"
# use_rerank = True  # nhưng dùng cross-encoder / LLM rerank thật
# hoặc thêm query expansion cho alias như "Approval Matrix" -> "Access Control SOP"
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | 3.70 | 3.50 | N/A | Baseline |
| Answer Relevance | 3.70 | 3.80 | N/A | Variant 1 |
| Context Recall | 5.00 | 5.00 | N/A | Tie |
| Completeness | 3.90 | 3.40 | N/A | Baseline |

**Đề xuất nếu làm tiếp:**
> Nếu có thêm thời gian, biến đáng thử nhất không phải hybrid nữa mà là rerank mạnh hơn hoặc query expansion. Hai hướng này nhắm đúng lỗi hiện tại: chọn nhầm evidence trọng tâm (`q06`) và xử lý alias/tên cũ (`q07`) mà không kéo thêm nhiều nhiễu.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Lỗi phổ biến nhất là retrieve được đúng source nhưng model vẫn trả lời chưa đúng trọng tâm hoặc abstain quá sớm, nhất là với câu hỏi có ngoại lệ và câu hỏi cần suy luận "không có thông tin riêng".

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Trên bộ test hiện tại, đổi từ dense sang hybrid không tạo tác động dương rõ ràng. Biến có tiềm năng tác động lớn nhất về sau nhiều khả năng là rerank/select đúng chunk hoặc query transformation cho alias, vì context recall đã bão hòa ở mức cao.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Nhóm nên thử một reranker thật sự cho top-10 candidates, hoặc thêm query expansion cho các alias/tên cũ như `Approval Matrix`. Đồng thời nên tinh chỉnh prompt abstain để model vẫn được phép kết luận "không có quy trình riêng được đề cập" khi tài liệu có đủ evidence gián tiếp.
