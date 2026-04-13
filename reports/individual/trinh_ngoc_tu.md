# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trịnh Ngọc Tú  
**Vai trò trong nhóm:** Co - Tech Lead (cùng Long)
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

> - Sprint nào bạn chủ yếu làm?
> Tôi chủ yếu làm sprint triển khai trả lời RAG và kiểm thử nhanh.
> - Cụ thể bạn implement hoặc quyết định điều gì?
> Tôi implement file `rag_answer.py` để tải câu hỏi, gọi retrieval, ghép context, gọi LLM và trả về đáp án; đồng thời chạy test vài case để đánh giá chất lượng trả lời.
> - Công việc của bạn kết nối với phần của người khác như thế nào?
> Đầu ra của `rag_answer.py` dùng trực tiếp dữ liệu từ phần indexing/retrieval của team, còn kết quả test giúp nhóm tinh chỉnh retrieval và prompt ở các bước sau.

_________________

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

> Chọn 1-2 concept từ bài học mà bạn thực sự hiểu rõ hơn sau khi làm lab.
> Ví dụ: chunking, hybrid retrieval, grounded prompt, evaluation loop.
> Giải thích bằng ngôn ngữ của bạn — không copy từ slide.
> Tôi chọn chunking và grounded prompt.
> - Chunking: Tôi hiểu rõ hơn việc chia tài liệu thành đoạn có kích thước và overlap hợp lý để retrieval không bỏ sót ý quan trọng nhưng cũng không quá dài gây loãng context.
> - Grounded prompt: Tôi hiểu cách viết prompt buộc model chỉ dựa trên context truy xuất được, từ đó giảm hallucination và giúp câu trả lời bám tài liệu hơn.

_________________

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

> Điều gì xảy ra không đúng kỳ vọng?
> Một số case trả lời chưa bám tài liệu vì đoạn context truy xuất bị lệch trọng tâm hoặc thiếu thông tin then chốt.
> Lỗi nào mất nhiều thời gian debug nhất?
> Lỗi tốn thời gian nhất là khi kết quả bị lạc đề; tôi phải lần lại toàn bộ pipeline retrieval và cách ghép context để tìm đoạn bị sai.
> Giả thuyết ban đầu của bạn là gì và thực tế ra sao?
> Ban đầu tôi nghĩ do prompt chưa đủ chặt, nhưng thực tế nguyên nhân chính là retrieval chọn nhầm= đoạn liên quan, nên cần tinh chỉnh retrieval trước rồi mới tối ưu prompt.

_________________

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

> Chọn 1 câu hỏi trong test_questions.json mà nhóm bạn thấy thú vị.
> Phân tích:
> - Baseline trả lời đúng hay sai? Điểm như thế nào?
> - Lỗi nằm ở đâu: indexing / retrieval / generation?
> - Variant có cải thiện không? Tại sao có/không?

**Câu hỏi:**
> Q07
**Phân tích:**
> Baseline trả lời gần đúng nhưng chưa nêu rõ tên mới của tài liệu, nên chỉ đạt mức trung bình về faithfulness/relevance.
> Lỗi chính nằm ở retrieval: không bắt được mapping “Approval Matrix” -> “Access Control SOP”, khiến generation phải suy diễn.
> Variant chưa cải thiện, thậm chí faithfulness giảm vì câu trả lời nhắc lại “Approval Matrix” như một tài liệu riêng thay vì chỉ rõ tên mới.

_________________

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

> 1-2 cải tiến cụ thể bạn muốn thử.
> Không phải "làm tốt hơn chung chung" mà phải là:
> "Tôi sẽ thử X vì kết quả eval cho thấy Y."
> Tôi sẽ thử hybrid retrieval + alias dictionary vì scorecard cho thấy các câu alias như q07 vẫn bị lệch, làm faithfulness giảm.
> Tôi sẽ thử tăng chất lượng retrieval bằng cách bổ sung synonym/alias mapping và tinh chỉnh top-k + rerank để bắt đúng tài liệu khi câu hỏi dùng tên cũ.
_________________

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
