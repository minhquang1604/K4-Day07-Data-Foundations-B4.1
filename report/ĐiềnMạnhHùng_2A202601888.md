# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Điền Mạnh Hùng

**Nhóm:** B4.1
**Ngày:** 2026-08-03

> Các kết quả retrieval bên dưới dùng corpus `data/shopee_policy/`, OpenAI
> `text-embedding-3-small`, và cùng bộ benchmark trong
> `docs/SHOPEE_POLICY_BENCHMARKS.md`.

## 1. Khởi động (Warm-up)

### Độ tương tự cosine

Cosine cao nghĩa là hai embedding cùng hướng, thường biểu diễn hai đoạn văn có
ý nghĩa gần nhau. Cosine đo hướng thay vì độ dài vector nên phù hợp với text
embedding hơn khoảng cách Euclid, vốn dễ bị ảnh hưởng bởi độ lớn vector.

- **Ví dụ cao:** “Người mua có thể yêu cầu hoàn tiền.” và “Khách hàng được trả
  hàng và nhận lại tiền.” Cả hai cùng nói về quyền trả hàng/hoàn tiền.
- **Ví dụ thấp:** “Người mua có 15 ngày để trả hàng.” và “Máy học sử dụng dữ
  liệu để học quy luật.” Hai câu thuộc hai chủ đề khác nhau.

### Bài toán chunking

- `chunk_size=500`, `overlap=50`: `ceil((10000 - 50) / (500 - 50))`
  = `ceil(22.11)` = **23 chunks**.
- `overlap=100`: `ceil((10000 - 100) / (500 - 100))`
  = `ceil(24.75)` = **25 chunks**. Overlap lớn hơn tạo thêm chunk nhưng giữ
  được ngữ cảnh ở ranh giới, giảm nguy cơ một ý bị cắt rời.

## 2. Hướng tiếp cận của tôi

### Chunking functions

- **`SentenceChunker.chunk`:** dùng regex tách sau `.`, `!`, `?` khi tiếp theo
  là whitespace/kết thúc chuỗi; giữ dấu câu và gom tối đa số câu cấu hình.
  Chuỗi rỗng hoặc chỉ có whitespace trả về danh sách rỗng.
- **`RecursiveChunker.chunk` / `_split`:** ưu tiên `\n\n`, `\n`, `. `, khoảng
  trắng, rồi cắt cứng. Khi một mảnh vượt kích thước, nó được xử lý đệ quy với
  separator ưu tiên thấp hơn; khi không còn separator, cắt theo `chunk_size`.
- **`compute_similarity`:** tính `dot(a,b) / (||a|| * ||b||)` và trả `0.0`
  nếu một vector có độ lớn bằng 0. `ChunkingStrategyComparator` trả số chunk,
  độ dài trung bình và danh sách chunk cho cả ba chiến lược.

### EmbeddingStore và agent

- **Store:** mỗi record gồm id, content, metadata, embedding; `doc_id` được
  bổ sung nếu thiếu. `search` xếp hạng dot product giảm dần; filter metadata
  chạy trước khi xếp hạng; `delete_document` xóa tất cả record cùng `doc_id`.
- **Agent:** lấy top-k chunk, đưa source/score/content vào context và yêu cầu
  LLM chỉ trả lời theo context hoặc nêu thiếu dữ liệu. Điều này giúp kiểm tra
  grounding và truy vết nguồn.

## 3. Hoàn thiện code

```text
.venv/bin/python -m pytest tests/ -v
42 passed in 0.02s
```

**Số lượng bài test vượt qua:** **42 / 42**

## 4. Dự đoán độ tương tự

Các score được tạo bằng `text-embedding-3-small` rồi gọi
`compute_similarity()`.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Người mua có thể yêu cầu hoàn tiền. | Khách hàng được trả hàng và nhận lại tiền. | cao | 0.582 | Có |
| 2 | Người bán phải đăng ảnh thật của sản phẩm. | Ảnh sản phẩm do người bán tự chụp là bắt buộc. | cao | 0.700 | Có |
| 3 | Phí hoàn hàng cùng tỉnh là 25.000 Xu. | Đơn khác tỉnh được hỗ trợ 40.000 Xu. | cao | 0.651 | Có |
| 4 | Người mua có 15 ngày để trả hàng. | Máy học sử dụng dữ liệu để học quy luật. | thấp | 0.212 | Có |
| 5 | Tài khoản đảm bảo giữ tiền thanh toán. | Tiền được lưu trước khi chuyển cho người bán. | cao | 0.485 | Có |

Điều đáng chú ý là cặp 5 chỉ ở mức trung bình dù cùng ý chính; cách diễn đạt
và chi tiết “tài khoản đảm bảo” vẫn tác động đến embedding. Vì vậy benchmark
cần kiểm tra trực tiếp chunk, không chỉ nhìn score.

## 5. Kết quả truy xuất của tôi

**Chiến lược:** `SentenceChunker(max_sentences_per_chunk=5)`; 218 chunks,
độ dài trung bình 755 ký tự. Query 3 dùng filter `buyer`, query 4 dùng filter
`seller`.

| # | Query | Top-1 (tóm tắt) | Score | Đủ evidence trong top-3? | Agent answer (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | Hạn trả hàng và ngoại lệ thực phẩm | `returns`, mục 3.2 | 0.719 | Có | 15 ngày; thực phẩm tươi sống/đông lạnh: 24 giờ |
| 2 | Thời hạn phản hồi của người bán | `guarantee` là nhiễu; evidence đúng ở top-3 | 0.566 | Có | Phản hồi trong 2 ngày lịch |
| 3 | Mức hỗ trợ phí tự gửi hàng hoàn | `return-shipping`, nhưng chunk top-3 thiếu mức Xu | 0.660 | Không đủ | Failure case: không trả được đủ 25.000/40.000 Xu |
| 4 | Điều kiện ảnh thật khi đăng bán | `listing`, evidence chính xác trong top-3 | 0.641 | Có | Ảnh tự chụp, sản phẩm chiếm ít nhất 40% ảnh |
| 5 | Nơi giữ tiền và trường hợp hoàn tiền | `terms`, mục Tài Khoản Đảm Bảo | 0.610 | Có | Tài Khoản Đảm Bảo; hoàn khi yêu cầu được chấp thuận |

**Evidence đầy đủ trong top-3:** **4 / 5**. Theo rubric, điểm retrieval/agent
ước tính là **7 / 10**: Q1, Q4, Q5 đạt 2 điểm; Q2 đạt 1 điểm vì evidence đúng
không ở top-1; Q3 đạt 0 điểm vì thiếu chi tiết định lượng.

Điều học được: chunk theo câu giữ trọn điều kiện và số liệu tốt hơn trong phần
lớn benchmark, nhưng một chủ đề có nhiều mức phí liền nhau vẫn có thể bị tách
khỏi nhau. Cần thử chunk theo heading/bảng hoặc gom FAQ pair cho dữ liệu chính
sách ở lần tiếp theo.

## Tự đánh giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code | 30 / 30 |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | 7 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
