# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Đặng Quang Minh
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500-800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi tập trung nhiều nhất ở Sprint 4 với vai trò Documentation Owner, nhưng công việc của tôi không chỉ là viết lại kết quả mà còn phải đọc kỹ toàn bộ pipeline để hiểu đúng những gì nhóm đã xây. Tôi rà soát `index.py`, `rag_answer.py`, `eval.py`, dữ liệu trong `data/docs/`, cùng các file kết quả trong `results/` để tổng hợp kiến trúc, quyết định chunking, retrieval strategy và kết quả A/B testing. Sau đó tôi điền lại `docs/architecture.md` và `docs/tuning-log.md` dựa trên số liệu thật như 29 chunks đã index, baseline dense đạt Faithfulness 3.70/5 và variant hybrid đạt 3.50/5. Ngoài phần tài liệu, tôi còn hỗ trợ dựng một giao diện demo bằng `Streamlit` để nhóm có thể trình bày trực quan hơn khi đánh giá pipeline.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn rằng một hệ thống RAG tốt không chỉ phụ thuộc vào LLM mà phụ thuộc rất mạnh vào cách chuẩn bị evidence trước khi model trả lời. Trước đây tôi nghĩ retrieval tốt nghĩa là chỉ cần tìm đúng tài liệu, nhưng qua scorecard tôi thấy đúng source vẫn chưa đủ; model còn phải lấy đúng đoạn và diễn giải đúng mức. Tôi cũng hiểu rõ hơn vai trò của chunking và grounded prompt. Chunking không phải chỉ là cắt văn bản cho đủ ngắn, mà là phải giữ được ranh giới tự nhiên để mỗi chunk vẫn có nghĩa độc lập. Grounded prompt cũng không đơn giản là thêm câu “chỉ trả lời từ context”, mà cần cân bằng giữa việc ép model không bịa và vẫn cho phép nó rút ra kết luận hợp lý từ bằng chứng có sẵn.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi ngạc nhiên nhất là variant hybrid không tốt hơn baseline dù trực giác ban đầu cho rằng kết hợp dense và BM25 sẽ giúp rõ rệt, nhất là với alias như “Approval Matrix”. Kết quả thực tế lại cho thấy Context Recall của baseline và variant đều đạt 5.00/5, nhưng Faithfulness và Completeness của variant còn giảm. Điều này khiến tôi nhận ra vấn đề chính không còn nằm ở recall tổng quát mà nằm ở việc chọn đúng evidence trọng tâm và tránh kéo thêm chunk nhiễu. Khó khăn lớn nhất là phân biệt lỗi đang nằm ở indexing, retrieval hay generation, vì nhiều câu nhìn bề ngoài giống retrieval fail nhưng thực ra retriever đã lấy đúng source rồi, chỉ có bước generation abstain quá sớm hoặc diễn giải dư. Việc đọc scorecard từng câu mới giúp tách đúng nguyên nhân.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q07 - Approval Matrix để cấp quyền hệ thống là tài liệu nào?`

**Phân tích:**

Tôi chọn `q07` vì đây là câu hỏi thể hiện rất rõ sự khác nhau giữa “tìm được tài liệu liên quan” và “trả lời đúng theo evidence”. Ở baseline dense, câu này được chấm Faithfulness 4, Relevance 4, Context Recall 5 và Completeness 5. Như vậy baseline đã retrieve được đúng nguồn là `it/access-control-sop.md` và câu trả lời nhìn chung đúng, dù vẫn hơi dài và có thêm phần mô tả mở rộng. Lỗi ở đây không nghiêm trọng và chủ yếu nằm ở generation: model trả lời nhiều hơn mức cần thiết thay vì chỉ nói ngắn gọn rằng “Approval Matrix for System Access” là tên cũ của “Access Control SOP”.

Ở variant hybrid, tôi kỳ vọng câu này sẽ cải thiện vì đây chính là bài toán alias. Thực tế Relevance tăng lên 5 nhưng Faithfulness lại giảm xuống 2. Nguyên nhân theo tôi là hybrid retrieval đã kéo được context liên quan, nhưng model lại diễn giải vượt ra ngoài bằng chứng trực tiếp, khiến câu trả lời bớt grounded. Trường hợp này cho thấy hybrid có thể giúp “đụng đúng vùng chủ đề”, nhưng nếu không có rerank mạnh hoặc prompt chặt hơn thì model vẫn có thể trả lời quá đà. Vì vậy, variant không thật sự cải thiện chất lượng tổng thể ở câu hỏi này.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ thử hai cải tiến cụ thể. Thứ nhất là thêm reranker thật sự cho top-10 candidates, vì kết quả eval cho thấy recall đã cao nhưng model vẫn chọn nhầm trọng tâm ở các câu như `q06` và `q07`. Thứ hai là tinh chỉnh prompt abstain để model có thể kết luận “không có quy trình riêng được đề cập” khi evidence gián tiếp đã đủ, thay vì mặc định từ chối như ở `q04` và `q10`.

---
