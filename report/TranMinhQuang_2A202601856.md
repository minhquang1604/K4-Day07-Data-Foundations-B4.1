# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Minh Quang  
**MSSV:** 2A202601856  
**Nhóm/Lớp:** 2A  
**Ngày nộp:** 03/08/2026

> Phần này trình bày kết quả cá nhân. Thang điểm được đối chiếu theo `docs/SCORING.md`.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

Độ tương tự cosine cao cho biết hai vector embedding có hướng gần nhau, nghĩa là hai đoạn văn bản thường có nội dung hoặc ý nghĩa gần nhau. Giá trị càng gần 1 thì mức tương đồng càng cao; gần 0 là ít liên quan và gần -1 là ngược hướng.

**Ví dụ có độ tương tự CAO:**

- Câu A: “Người mua cần gửi yêu cầu đổi trả khi hàng bị lỗi.”
- Câu B: “Khách hàng có thể yêu cầu hoàn trả sản phẩm bị lỗi.”
- Tại sao tương đồng: Hai câu cùng nói về việc người mua yêu cầu đổi/trả một sản phẩm bị lỗi, chỉ khác cách diễn đạt.

**Ví dụ có độ tương tự THẤP:**

- Câu A: “Sản phẩm bị cấm không được đăng bán.”
- Câu B: “Hôm nay thời tiết tại Hà Nội có mưa.”
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau: quy định thương mại điện tử và thời tiết.

**Tại sao cosine similarity được ưu tiên hơn khoảng cách Euclid cho text embeddings?**

Cosine similarity tập trung vào hướng của vector, nên ít bị ảnh hưởng bởi độ lớn của vector hoặc độ dài văn bản. Trong khi đó, khoảng cách Euclid có thể thay đổi mạnh theo độ lớn vector dù hai văn bản vẫn có hướng biểu diễn ngữ nghĩa tương tự.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, `chunk_size=500`, `overlap=50`. Bao nhiêu chunks?**

Theo công thức của đề bài:

```text
số chunks = ceil((độ dài tài liệu - overlap) / (chunk_size - overlap))
           = ceil((10.000 - 50) / (500 - 50))
           = ceil(9.950 / 450)
           = ceil(22,111...)
           = 23 chunks
```

**Nếu overlap tăng lên 100 thì số chunks thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

```text
số chunks = ceil((10.000 - 100) / (500 - 100))
           = ceil(9.900 / 400)
           = ceil(24,75)
           = 25 chunks
```

Số chunk tăng từ 23 lên 25 vì bước trượt giảm từ 450 xuống 400 ký tự. Overlap lớn hơn giúp giữ thông tin nằm sát ranh giới giữa hai chunk, nhưng làm tăng dữ liệu trùng lặp, chi phí embedding và số bản ghi cần tìm kiếm.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk` — hướng tiếp cận:**

Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\r?\n+)` để tách tại khoảng trắng hoặc xuống dòng đứng sau dấu kết thúc câu, nhờ đó vẫn giữ dấu câu trong nội dung. Các câu được `strip()`, loại bỏ phần rỗng rồi ghép theo `max_sentences_per_chunk`; văn bản rỗng trả về `[]` và tham số số câu được bảo đảm tối thiểu là 1.

**`RecursiveChunker.chunk` / `_split` — hướng tiếp cận:**

Thuật toán thử lần lượt các separator `\n\n`, `\n`, `. `, khoảng trắng rồi chuỗi rỗng, ưu tiên giữ ranh giới cấu trúc lớn trước. Base case là đoạn hiện tại không vượt quá `chunk_size`; nếu không còn separator phù hợp hoặc gặp separator rỗng, hàm cắt cứng theo số ký tự để luôn tiến triển. Các phần nhỏ được ghép vào buffer đến sát giới hạn, còn phần quá lớn được xử lý đệ quy với separator tiếp theo.

### Lớp `EmbeddingStore`

**`add_documents` + `search` — hướng tiếp cận:**

Mỗi tài liệu được chuyển thành một record gồm ID, nội dung, bản sao metadata, embedding và một storage ID duy nhất; `doc_id` được bổ sung nếu chưa có. Store luôn giữ một bản trong bộ nhớ và có thể đồng bộ sang ChromaDB khi thư viện sẵn có. Khi tìm kiếm, truy vấn được embed, hệ thống tính dot product với các vector đã lưu, sắp xếp điểm giảm dần rồi lấy `top_k`; với các embedder đã chuẩn hóa vector như `MockEmbedder` và `LocalEmbedder`, dot product tương đương cosine similarity.

**`search_with_filter` + `delete_document` — hướng tiếp cận:**

`search_with_filter` lọc trước các record có metadata khớp chính xác tất cả cặp khóa–giá trị, sau đó mới tính điểm và xếp hạng; cách này tránh để kết quả ngoài phạm vi chiếm top-k. `delete_document` tìm mọi chunk có `metadata["doc_id"]` tương ứng, xóa chúng khỏi store trong bộ nhớ và xóa các storage ID tương ứng khỏi ChromaDB; hàm trả về `True` khi có dữ liệu bị xóa và `False` nếu không tìm thấy.

### Tác tử `KnowledgeBaseAgent`

**`answer` — hướng tiếp cận:**

Agent gọi `store.search(question, top_k)` rồi ghép các chunk thành các khối `[Context 1]`, `[Context 2]`, ... trong prompt. Prompt yêu cầu LLM chỉ trả lời dựa trên ngữ cảnh truy xuất và phải nói rõ khi dữ liệu không đủ, sau đó chèn câu hỏi ở phần riêng trước khi gọi `llm_fn(prompt)`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết quả kiểm thử

Lệnh chạy trong môi trường Python 3.11 của dự án:

```text
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED  [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED   [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED  [ 45%]
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

============================= 42 passed in 0.17s =============================
```

**Số lượng bài test vượt qua (pass): 42 / 42.**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tôi dự đoán theo nghĩa của câu trước khi chạy. Điểm thực tế được tính bằng `compute_similarity(MockEmbedder()(câu A), MockEmbedder()(câu B))`; quy ước điểm từ `0,5` trở lên là “cao”.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Người mua cần gửi yêu cầu đổi trả khi hàng bị lỗi. | Khách hàng có thể yêu cầu hoàn trả sản phẩm bị lỗi. | Cao | 0,2275 (thấp) | Không |
| 2 | Người bán phải cung cấp mô tả sản phẩm chính xác. | Thông tin giá và tình trạng hàng cần được đăng chính xác. | Cao | -0,0835 (thấp) | Không |
| 3 | Sản phẩm bị cấm không được đăng bán. | Hôm nay thời tiết tại Hà Nội có mưa. | Thấp | -0,0864 (thấp) | Có |
| 4 | Yêu cầu đổi trả cần kèm bằng chứng phù hợp. | Người mua phải cung cấp bằng chứng khi hàng không đúng mô tả. | Cao | -0,1300 (thấp) | Không |
| 5 | Người bán phản hồi yêu cầu đổi trả theo quy trình của sàn. | Mô hình học máy học từ dữ liệu huấn luyện. | Thấp | -0,1013 (thấp) | Có |

**Kết quả bất ngờ nhất và nhận xét:**

Cặp 4 gần như diễn đạt cùng một yêu cầu nhưng lại nhận điểm âm, trái với dự đoán ngữ nghĩa. Nguyên nhân là `MockEmbedder` tạo vector xác định từ hàm băm của toàn chuỗi để phục vụ unit test, không học ý nghĩa hay quan hệ giữa từ; vì vậy kết quả cho thấy muốn đánh giá semantic similarity thật sự cần dùng model đa ngữ như `paraphrase-multilingual-MiniLM-L12-v2`, còn mock chỉ phù hợp để kiểm tra luồng chương trình.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

### Thiết lập đánh giá

- Corpus hiện có: `returns-policy.md` và `seller-listing.md` trong `data/k4_ecommerce/`.
- Chunker: `SentenceChunker(max_sentences_per_chunk=2)`, tạo tổng cộng 5 chunks.
- Embedding backend: `MockEmbedder` mặc định, chạy in-memory.
- Mỗi truy vấn lấy top-3; câu 5 dùng `metadata_filter={"customer_role": "seller"}`.
- Vì hai file hiện là dữ liệu khởi động/template chứ chưa phải corpus 5–10 nguồn chính thức, kết quả dưới đây chỉ phản ánh lần chạy có thể tái lập trên trạng thái hiện tại của repository.

| # | Câu hỏi (Query) | Top-1 chunk truy xuất được (tóm tắt) | Score | Top-1 liên quan? | Câu trả lời của Agent (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | Người mua cần làm gì khi muốn đổi trả hàng bị lỗi hoặc không đúng mô tả? | Quy định người bán phải cung cấp thông tin đăng bán chính xác. | 0,0582 | Không | Không đủ ngữ cảnh liên quan trong top-3 để trả lời chắc chắn. |
| 2 | Yêu cầu đổi trả phải được gửi trong thời hạn nào? | Ghi chú metadata của tài liệu đổi trả, chưa chứa thời hạn. | 0,1167 | Không | Không đủ ngữ cảnh; chunk chứa quy định “thời hạn nêu trên trang sản phẩm/chính sách” không vào top-3. |
| 3 | Người bán có trách nhiệm gì khi nhận yêu cầu đổi trả? | Quy định về thông tin khi đăng bán, không phải xử lý đổi trả. | 0,2344 | Không | Chunk liên quan ở hạng 2: người bán phải phản hồi theo quy trình của sàn. |
| 4 | Khi đăng bán sản phẩm, người bán phải cung cấp chính xác những thông tin nào? | Người bán phải cung cấp chính xác giá, mô tả và tình trạng hàng. | 0,3351 | Có | Người bán chịu trách nhiệm bảo đảm giá, mô tả và tình trạng hàng là chính xác. |
| 5 | Người bán có được đăng sản phẩm bị hạn chế hoặc bị cấm không? | Chunk về trách nhiệm cung cấp thông tin sản phẩm chính xác. | 0,0204 | Không | Chunk liên quan ở hạng 2: sản phẩm bị hạn chế hoặc bị cấm không được đăng bán. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3? 3 / 5.**

**Phân tích ngắn:**

Các câu 3, 4 và 5 có bằng chứng liên quan trong top-3; câu 4 đưa đúng bằng chứng lên top-1. Câu 5 cho thấy metadata filter đã giới hạn đúng phạm vi người bán, nhưng mock embedding vẫn xếp chunk trực tiếp trả lời câu hỏi ở hạng 2. Hai thất bại ở câu 1 và 2 xuất phát chủ yếu từ mock embedding không biểu diễn ngữ nghĩa, đồng thời các câu ghi chú template bị đưa vào corpus và tạo nhiễu; cần loại phần hướng dẫn/template trước khi ingest và dùng local multilingual embedder để cải thiện.

**Điều hay nhất tôi học được khi đối chiếu các chiến lược:**

Chất lượng RAG không chỉ phụ thuộc vào code tìm kiếm mà còn phụ thuộc mạnh vào việc làm sạch corpus, chọn ranh giới chunk và model embedding. Metadata filter giúp thu hẹp đúng đối tượng, nhưng không thể thay thế một embedding model có khả năng biểu diễn ngữ nghĩa; do đó cần xem trực tiếp top-k và phân tích failure case thay vì chỉ nhìn vào score.

---

## Tự đánh giá (Phần cá nhân)

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 4 / 10 |
| **Tổng phần cá nhân** | **54 / 60** |

> Điểm retrieval tự đánh giá theo rubric: câu 1 = 0, câu 2 = 0, câu 3 = 1 (chunk liên quan không ở top-1), câu 4 = 2 (top-1 liên quan và trả lời đúng), câu 5 = 1 (chunk liên quan không ở top-1).
