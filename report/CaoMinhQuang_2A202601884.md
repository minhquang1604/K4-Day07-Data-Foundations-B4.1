# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Cao Minh Quang
**Nhóm:** B4.1
**Ngày:** 2026-08-03

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

Hai vector embedding gần như cùng hướng trong không gian nhiều chiều, tức hai
đoạn văn bản nhiều khả năng diễn đạt cùng một ý dù có thể dùng từ ngữ khác
nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Người mua có thể yêu cầu hoàn tiền."
- Câu B: "Khách hàng được trả hàng và nhận lại tiền."
- Tại sao tương đồng: cả hai cùng nói về quyền trả hàng/hoàn tiền của người
  mua, chỉ khác cách diễn đạt và chủ ngữ.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Người mua có 15 ngày để trả hàng."
- Câu B: "Máy học sử dụng dữ liệu để học quy luật."
- Tại sao khác: hai câu thuộc hai chủ đề hoàn toàn khác nhau — chính sách
  thương mại điện tử so với khái niệm machine learning.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

Cosine chỉ đo *hướng* của vector, bỏ qua độ dài, nên hai câu cùng ý nghĩa
nhưng khác độ dài/chi tiết vẫn cho điểm cao. Khoảng cách Euclid cộng dồn
chênh lệch trên từng chiều nên nhạy với độ lớn vector, dễ đánh giá sai hai
câu gần nghĩa nhưng có độ dài khác nhau là "không giống nhau".

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính: `ceil((10000 - 50) / (500 - 50))` = `ceil(9950 / 450)`
> = `ceil(22.11)`
> Đáp án: **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
`ceil((10000 - 100) / (500 - 100))` = `ceil(9900 / 400)` = `ceil(24.75)` =
**25 chunks** — tăng 2 chunk so với overlap=50. Overlap lớn hơn giúp một
câu/điều kiện nằm vắt ngang ranh giới giữa hai chunk ít bị cắt rời hoàn
toàn, giữ ngữ cảnh tốt hơn cho truy xuất, đổi lại tốn thêm dung lượng lưu
trữ và chi phí embed vì nội dung bị lặp lại nhiều hơn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Mô tả dưới đây bám sát đúng những gì tôi đã lập trình trong
`src/CaoMinhQuang/`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
Tôi tách câu bằng regex `(?<=[.!?])(?:\n| )`: lookbehind giữ nguyên dấu câu
`.`/`!`/`?` ở cuối mỗi câu, còn phần tách thực sự diễn ra ngay sau đó khi
gặp khoảng trắng hoặc ký tự xuống dòng. Sau khi `re.split`, tôi `strip()`
từng câu và loại bỏ câu rỗng, rồi gom các câu thành từng nhóm kích thước
`max_sentences_per_chunk` bằng `range(0, len(sentences), step)`, nối lại
bằng dấu cách. Trường hợp ngoại lệ: chuỗi rỗng (`if not text`) trả về danh
sách rỗng ngay từ đầu hàm, tránh regex chạy trên input rỗng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
Thuật toán thử lần lượt các separator theo thứ tự ưu tiên `["\n\n", "\n",
". ", " ", ""]`. Với separator hiện tại, tôi `split()` văn bản thành các
`parts` rồi gộp dần vào biến tích lũy `current_chunk` miễn còn dưới
`chunk_size`; khi gộp tiếp sẽ vượt giới hạn thì chốt `current_chunk` lại và
bắt đầu phần mới. Nếu một `part` tự nó đã dài hơn `chunk_size`, tôi gọi đệ
quy `_split(part, rest)` với danh sách separator còn lại (ưu tiên thấp
hơn) — đây là bước đệ quy chính. **Base case:** khi `len(current_text) <=
chunk_size` thì trả về `[current_text]` luôn; khi hết separator hoặc gặp
separator rỗng (`""`), cắt cứng theo từng đoạn đúng `chunk_size` ký tự.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
`__init__` thử `import chromadb` và tạo collection; nếu không cài được
ChromaDB (hoặc khởi tạo lỗi), tự rơi về danh sách in-memory `self._store`
nên code vẫn chạy được khi máy chưa có ChromaDB. `add_documents` gọi
`self._embedding_fn` để nhúng nội dung từng `Document` rồi lưu record gồm
`id`, `content`, `embedding`, `metadata` (có `doc_id` mặc định bằng
`metadata.setdefault`). `search` nhúng câu hỏi, tính dot product giữa
embedding câu hỏi với từng embedding đã lưu, sắp xếp giảm dần theo `score`
và cắt lấy `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
Filter chạy **trước** similarity search: tôi lọc `self._store` bằng
`all(record["metadata"].get(key) == value for key, value in
metadata_filter.items())` để giữ lại các record khớp toàn bộ điều kiện, rồi
mới gọi lại hàm search dot-product trên tập đã lọc. `delete_document` xóa
bằng cách tạo lại `self._store` chỉ giữ những record có
`metadata["doc_id"] != doc_id`, và trả `True`/`False` dựa vào việc kích
thước danh sách có giảm hay không sau khi lọc.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
`answer` gọi `store.search(question, top_k=top_k)` để lấy các chunk liên
quan nhất, nối `content` của chúng bằng `"\n\n"` làm `context`. Prompt được
dựng theo cấu trúc cố định: hướng dẫn LLM chỉ trả lời dựa trên context, nêu
rõ nếu context không đủ thông tin để trả lời, sau đó chèn `Context:`,
`Question:` và `Answer:`. Cách đưa context vào prompt như vậy giúp kiểm tra
được grounding — có thể so khớp câu trả lời với đúng chunk đã được đưa vào.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

Repo tổ chức mỗi sinh viên một package con (`src/CaoMinhQuang/`) thay vì một
`src/` dùng chung, nên tôi trỏ bộ test vào đúng package của mình qua biến
môi trường `LAB_SOLUTION_PACKAGE`. Ban đầu `test_src_package_exists` báo
FAILED vì thư mục `src/` (thư mục cha chứa các package con) còn thiếu file
`src/__init__.py`; tôi bổ sung file rỗng đó (chỉ đóng vai trò marker cho
thư mục cha, không export gì) và test pass:

```
$ LAB_SOLUTION_PACKAGE=src.CaoMinhQuang .venv/bin/python -m pytest tests/ -v
...
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::... (6 tests) PASSED
tests/test_solution.py::TestSentenceChunker::... (4 tests) PASSED
tests/test_solution.py::TestRecursiveChunker::... (4 tests) PASSED
tests/test_solution.py::TestEmbeddingStore::... (8 tests) PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::... (2 tests) PASSED
tests/test_solution.py::TestComputeSimilarity::... (4 tests) PASSED
tests/test_solution.py::TestCompareChunkingStrategies::... (3 tests) PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::... (3 tests) PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::... (3 tests) PASSED

============================== 42 passed in 0.04s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Điểm thực tế dưới đây do nhóm chạy bằng `text-embedding-3-small` (embedding
thật) trên cùng 5 cặp câu, theo `report/REPORT_NHOM.md`.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có thể yêu cầu hoàn tiền. | Khách hàng được trả hàng và nhận lại tiền. | cao | 0.582 | Có |
| 2 | Người bán phải đăng ảnh thật của sản phẩm. | Ảnh sản phẩm do người bán tự chụp là bắt buộc. | cao | 0.700 | Có |
| 3 | Phí hoàn hàng cùng tỉnh là 25.000 Xu. | Đơn khác tỉnh được hỗ trợ 40.000 Xu. | cao | 0.651 | Có |
| 4 | Người mua có 15 ngày để trả hàng. | Máy học sử dụng dữ liệu để học quy luật. | thấp | 0.212 | Có |
| 5 | Tài khoản đảm bảo giữ tiền thanh toán. | Tiền được lưu trước khi chuyển cho người bán. | cao | 0.485 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
Bất ngờ nhất là cặp 5: cùng nói về việc giữ tiền trước khi chuyển cho người
bán nhưng điểm chỉ 0.485, thấp hơn hẳn cặp 2 và 3 dù ý nghĩa gần nhau không
kém. Câu A dùng thuật ngữ cụ thể "tài khoản đảm bảo" còn câu B diễn đạt
chung chung hơn — điều này cho thấy embedding vẫn nhạy với cách diễn đạt và
thuật ngữ cụ thể chứ không chỉ với ý nghĩa trừu tượng, nên khi đánh giá
retrieval không thể chỉ nhìn con số mà cần đọc lại nội dung chunk.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

**Chiến lược:** `SentenceChunker(max_sentences_per_chunk=5)` trên corpus 6
tài liệu của nhóm (`data/shopee_policy/`) — 218 chunks, độ dài trung bình
755 ký tự, theo baseline nhóm đã đo trong `report/REPORT_NHOM.md`. Query 3
dùng filter `customer_role=buyer`, query 4 dùng filter `customer_role=
seller`, đúng theo `docs/SHOPEE_POLICY_BENCHMARKS.md`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Hạn trả hàng và ngoại lệ thực phẩm | `returns`, mục 3.2 | 0.719 | Có, top-1 | 15 ngày; thực phẩm tươi sống/đông lạnh: 24 giờ |
| 2 | Thời hạn phản hồi của người bán | `guarantee` (nhiễu ở top-1); evidence đúng ở top-3 | 0.566 | Có, nhưng không ở top-1 | Phản hồi trong 2 ngày lịch |
| 3 | Mức hỗ trợ phí tự gửi hàng hoàn | `return-shipping`, nhưng chunk thiếu 1 trong 2 mức Xu | 0.660 | Có nhưng chưa đủ evidence | Không nêu đủ cả 25.000 và 40.000 Xu |
| 4 | Điều kiện ảnh thật khi đăng bán | `listing`, evidence chính xác ở top-3 | 0.641 | Có, top-1 | Ảnh tự chụp, sản phẩm chiếm ít nhất 40% ảnh |
| 5 | Nơi giữ tiền và trường hợp hoàn tiền | `terms`, mục Tài Khoản Đảm Bảo | 0.610 | Có, top-1 | Tài Khoản Đảm Bảo; hoàn khi yêu cầu được chấp thuận |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4 / 5** — Q3
là failure case: đúng document nhưng top-3 không giữ đồng thời cả hai mức
phí (25.000 Xu cùng tỉnh / 40.000 Xu khác tỉnh) trong cùng một chunk, nên
câu trả lời thiếu ý quan trọng dù chunk đúng chủ đề vẫn nằm trong top-3.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
So với `FixedSizeChunker` (baseline nhóm: 3/5 evidence) và `RecursiveChunker`
(3/5 evidence), `SentenceChunker` giữ nguyên câu nên tốt hơn ở các câu hỏi
mà gold answer nằm gọn trong 1–2 câu liên tiếp. Nhưng qua trường hợp Q3, tôi
học được rằng chunk theo số câu cố định không đảm bảo các số liệu thuộc
cùng một điều khoản (như hai mức phí hoàn hàng) luôn nằm chung một chunk —
nếu làm lại tôi sẽ thử chunk theo heading/mục để giữ nguyên cả đoạn cùng
chủ đề thay vì chỉ đếm số câu.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
