# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Phú Nghĩa (20233871)
**Nhóm:** [Tên nhóm]
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là hai vector embedding hướng gần giống nhau, nên hai đoạn văn thường biểu đạt nội dung hoặc ý nghĩa gần nhau. Điểm càng gần 1 thì hướng của hai vector càng tương đồng.

**Ví dụ có độ tương tự CAO:**
- Câu A: Khách hàng có thể yêu cầu hoàn tiền trong vòng 7 ngày.
- Câu B: Người mua được phép đề nghị trả lại tiền trong thời hạn một tuần.
- Tại sao tương đồng: Hai câu dùng từ khác nhau nhưng cùng nói về quyền yêu cầu hoàn tiền và cùng một thời hạn.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Đơn hàng sẽ được giao trong hai ngày làm việc.
- Câu B: Mạng nơ-ron sâu gồm nhiều lớp xử lý dữ liệu.
- Tại sao khác: Một câu nói về thời gian giao hàng, câu còn lại nói về kiến trúc học máy; chúng hầu như không có cùng chủ đề hay ý định.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine tập trung vào hướng của vector, tức là mẫu quan hệ ngữ nghĩa, và ít bị ảnh hưởng bởi độ lớn của vector. Khoảng cách Euclid phụ thuộc cả hướng lẫn độ lớn, nên hai embedding cùng ý nghĩa nhưng có norm khác nhau vẫn có thể bị xem là xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.111...)`.
> Đáp án: **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số chunk là `ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = 25`, tức tăng từ 23 lên 25. Overlap lớn hơn giữ lại nhiều ngữ cảnh ở ranh giới giữa hai chunk, giảm nguy cơ tách rời câu hoặc ý quan trọng, nhưng làm tăng số chunk, chi phí lưu trữ và lượng tính toán truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\r?\n+)` để tách tại khoảng trắng hoặc xuống dòng đứng sau dấu kết thúc câu, nhờ vậy dấu câu vẫn thuộc về câu phía trước. Sau khi loại khoảng trắng thừa và phần tử rỗng, các câu được gom tuần tự theo `max_sentences_per_chunk`; văn bản rỗng chỉ trả về danh sách rỗng và tham số số câu luôn được chặn tối thiểu là 1.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán lần lượt thử các separator theo độ ưu tiên `\n\n`, `\n`, `. `, khoảng trắng rồi ký tự; các phần nhỏ được ghép lại miễn là không vượt `chunk_size`, còn phần quá lớn được đưa xuống mức separator tiếp theo. Base case là đoạn đã đủ ngắn thì trả về ngay; nếu hết separator hoặc gặp separator rỗng, hàm cắt trực tiếp theo số ký tự để luôn tiến triển và bảo đảm giới hạn kích thước.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents` chuẩn hóa mỗi `Document` thành record gồm ID duy nhất, nội dung, bản sao metadata, `doc_id` và embedding; record được giữ trong bộ nhớ và đồng thời ghi sang ChromaDB nếu thư viện tùy chọn này khả dụng. `search` chỉ embedding câu hỏi một lần, tính dot product với từng embedding đã lưu, sắp xếp điểm giảm dần rồi trả tối đa `top_k` kết quả cùng nội dung, metadata và score.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc trước theo phép khớp chính xác tất cả cặp khóa–giá trị metadata, rồi mới xếp hạng các ứng viên còn lại; khi không có filter, hàm dùng cùng đường xử lý với `search`. `delete_document` tìm tất cả record có `metadata['doc_id']` tương ứng, xóa toàn bộ chunk đó khỏi store (và ChromaDB nếu đang dùng), đồng thời trả `True` chỉ khi thực sự tìm thấy dữ liệu để xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `answer` truy xuất top-k chunk rồi đánh số từng đoạn trong phần `Context` của prompt, sau đó chèn nguyên câu hỏi vào phần `Question`. Chỉ dẫn yêu cầu LLM chỉ dùng ngữ cảnh được cung cấp và phải nói không đủ thông tin nếu context thiếu; prompt hoàn chỉnh được chuyển cho `llm_fn` để sinh câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 42 items

tests/test_solution.py ..........................................       [100%]

============================== 42 passed in 0.08s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | cao / thấp | | |
| 2 | | | cao / thấp | | |
| 3 | | | cao / thấp | | |
| 4 | | | cao / thấp | | |
| 5 | | | cao / thấp | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** __ / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | Chờ Giai đoạn 2 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | Chờ Giai đoạn 2 / 10 |
| **Tổng phần cá nhân** | **45 / 60 (tạm tính sau Giai đoạn 1)** |
