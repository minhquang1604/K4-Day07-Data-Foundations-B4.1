# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Lâm Tùng Bách
**Nhóm:** [B4.1]
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Khi hai đoạn văn bản có độ tương tự cosine cao, các vector embedding của chúng hướng gần giống nhau. Điều này cho thấy hai đoạn thường có nội dung, chủ đề hoặc ý nghĩa ngữ nghĩa tương đồng, ngay cả khi chúng không sử dụng chính xác cùng một từ ngữ.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Khách hàng có thể hoàn trả sản phẩm trong vòng 30 ngày."
- Câu B: "Người mua được phép trả lại hàng trong thời hạn một tháng."
- Tại sao tương đồng: Hai câu dùng từ ngữ khác nhau nhưng đều diễn đạt cùng một chính sách: người mua có quyền trả hàng trong khoảng 30 ngày.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Đơn hàng của bạn sẽ được giao trong vòng ba ngày làm việc."
- Câu B: "Cây xanh hấp thụ khí carbon dioxide trong quá trình quang hợp."
- Tại sao khác: Hai câu thuộc hai chủ đề và mang hai ý nghĩa hoàn toàn khác nhau: giao hàng thương mại điện tử và quá trình sinh học của thực vật.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity tập trung vào góc, tức hướng ngữ nghĩa của hai vector, và gần như không bị ảnh hưởng bởi độ lớn của vector. Khoảng cách Euclid lại phụ thuộc cả hướng lẫn độ lớn, nên hai văn bản có ý nghĩa giống nhau vẫn có thể bị đánh giá là xa nhau khi độ dài hoặc chuẩn vector khác nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.111...) = 23`.
>
> **Đáp án: 23 chunks.**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi `overlap=100`, số chunk là `ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = ceil(24.75) = 25`; như vậy số lượng tăng từ 23 lên 25 chunks (tăng 2 chunks). Tăng overlap giúp giữ lại nhiều ngữ cảnh hơn tại ranh giới giữa hai chunk, tránh làm mất hoặc chia cắt ý quan trọng và có thể cải thiện khả năng truy xuất; đổi lại, dữ liệu bị lặp nhiều hơn, làm tăng chi phí lưu trữ và embedding.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\r?\n+)` để tách sau dấu kết thúc câu nhưng vẫn giữ lại dấu câu, sau đó loại khoảng trắng thừa và nhóm tối đa `max_sentences_per_chunk` câu. Hàm trả về danh sách rỗng khi đầu vào rỗng/chỉ có khoảng trắng và ép số câu tối đa về ít nhất 1 để tránh bước lặp không hợp lệ.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử các separator theo thứ tự ưu tiên `\n\n` → `\n` → `. ` → khoảng trắng → chuỗi rỗng, ghép các phần nhỏ vào một buffer cho đến giới hạn `chunk_size`, đồng thời giữ lại delimiter để không làm mất dấu câu hoặc ranh giới đoạn. Phần vượt giới hạn được tách đệ quy bằng separator tiếp theo; base case là văn bản đã vừa kích thước, còn khi hết separator thì cắt cứng theo số ký tự để luôn bảo đảm tiến triển.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents` sao chép metadata, tự bổ sung `doc_id` nếu thiếu, tạo embedding và mã lưu trữ duy nhất cho từng bản ghi; dữ liệu được lưu trong ChromaDB nếu thư viện khả dụng, nếu không sẽ dùng danh sách trong bộ nhớ. `search` nhúng câu truy vấn, tính dot product với từng embedding, sắp xếp điểm giảm dần và chỉ trả về tối đa `top_k` kết quả theo cấu trúc thống nhất gồm `id`, `content`, `metadata` và `score`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc trước các bản ghi có metadata khớp toàn bộ cặp khóa–giá trị, rồi mới tính độ tương tự trên tập ứng viên nhỏ hơn; nếu không truyền bộ lọc, hàm hoạt động giống `search`. `delete_document` tìm và xóa tất cả bản ghi có `metadata['doc_id']` tương ứng trên cả backend bộ nhớ và ChromaDB, sau đó trả về `True` chỉ khi thực sự có bản ghi bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `answer` truy xuất `top_k` chunk liên quan, đánh số từng đoạn rồi đưa chúng vào phần `NGỮ CẢNH` của prompt cùng câu hỏi ở phần `CÂU HỎI`. Prompt yêu cầu LLM chỉ dùng thông tin đã truy xuất và phải nói rõ khi ngữ cảnh không đủ; nếu store không có kết quả, thông báo thiếu ngữ cảnh vẫn được đưa vào prompt trước khi gọi `llm_fn`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.4, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: C:\Users\Admin\OneDrive\Desktop\VinAI\Day07-2A202601830_NguyenLamTungBach
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.12s ==============================
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
| Dự đoán độ tương tự (Similarity Predictions) | 4/ 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8/ 10 |
| **Tổng phần cá nhân** | **57/ 60** |
