# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** B4.1

**Thành viên:** 

| STT | Họ và tên            | Mã học viên |
| :-: | -------------------- | ----------- |
|  1  | Trần Phú Nghĩa       | 2A202601298 |
|  2  | Nguyễn Lâm Tùng Bách | 2A202601830 |
|  3  | Cao Minh Quang       | 2A202601884 |
|  4  | Trần Minh Quang      | 2A202601806 |
|  5  | Điền Mạnh Hùng       | 2A202601888 |

**Ngày:** 2026-08-03

## 1. Lựa chọn tài liệu

**Phạm vi:** chính sách trả hàng/hoàn tiền, bảo vệ người mua và nghĩa vụ đăng
bán của Shopee Việt Nam. Tất cả là trang công khai của Trung tâm trợ giúp
Shopee, được crawl chậm sau khi kiểm tra `robots.txt`.

| # | Tài liệu | Ngày lấy / phiên bản | Số ký tự | Metadata |
|---|---|---|---:|---|
| 1 | Chính sách trả hàng và hoàn tiền | 2026-08-03 / not-stated | 26,198 | both, returns, vi |
| 2 | Quy định về đăng bán sản phẩm | 2026-08-03 / not-stated | 28,830 | seller, listing, vi |
| 3 | Điều khoản dịch vụ | 2026-08-03 / not-stated | 110,814 | both, payment-and-escrow, vi |
| 4 | Shopee Đảm Bảo | 2026-08-03 / not-stated | 2,146 | buyer, buyer-protection, vi |
| 5 | Phương thức gửi hàng hoàn trả và phí hoàn trả | 2026-08-03 / not-stated | 8,439 | buyer, return-shipping, vi |
| 6 | Điều khoản dịch vụ Shopee Mall | 2026-08-03 / not-stated | 44,464 | seller, mall-returns, vi |

Manifest có URL gốc và căn cứ sử dụng `public-page` tại
`data/shopee_policy/sources.csv`. Corpus không chứa dữ liệu cá nhân, thông tin
đăng nhập hay nguồn nội bộ.

| Trường metadata | Ví dụ | Tác dụng |
|---|---|---|
| `doc_id`, `title` | `shopee-return-shipping` | Truy vết document/chunk |
| `source_url`, `retrieved_at`, `document_version` | URL gốc, `2026-08-03`, `not-stated` | Kiểm chứng nguồn và độ mới |
| `customer_role` | buyer / seller / both | Lọc theo đối tượng |
| `category`, `language`, `source_type` | returns, vi, official-help-center | Thu hẹp ngữ cảnh và phân tích corpus |

## 2. Thiết kế chiến lược

### Baseline trên toàn bộ corpus

| Chiến lược | Tham số | Số chunk | Độ dài TB | Nhận xét |
|---|---|---:|---:|---|
| FixedSize | 750, overlap 100 | 257 | 741.8 | Số chunk ít nhưng dễ cắt điều kiện/số liệu |
| Sentence | 5 câu/chunk | 218 | 755.0 | Chunk dễ đọc, giữ câu hoàn chỉnh |
| Recursive | 750 ký tự | 305 | 540.8 | Ưu tiên ranh giới đoạn/câu, nhiều chunk hơn |

**Chiến lược cá nhân được chọn:** `SentenceChunker(max_sentences_per_chunk=5)`
vì các chính sách có nhiều điều kiện và ngoại lệ được viết theo câu. Model
embedding dùng trong đánh giá là `text-embedding-3-small`; đây là embedding
thật, không phải mock.

| Chiến lược | Evidence đầy đủ trong top-3 | Điểm rubric ước tính | Điểm mạnh | Điểm yếu |
|---|---:|---:|---|---|
| FixedSize | 3 / 5 | 6 / 10 | Tìm tốt Q1–Q3 | Mất evidence Q4–Q5 |
| Sentence | 4 / 5 | 7 / 10 | Giữ đủ điều kiện Q1, Q2, Q4, Q5 | Mất hai mức phí ở Q3 |
| Recursive | 3 / 5 | 6 / 10 | Chunk ngắn, có cấu trúc | Q4–Q5 thiếu evidence ở top-3 |

Điểm trong bảng là đánh giá theo evidence + câu trả lời grounded trên cùng 5
query, không thay thế đánh giá của giảng viên. Sentence là lựa chọn tốt nhất
cho corpus này, nhưng chưa đủ để coi là tối ưu tuyệt đối.

## 3. Benchmark và chất lượng truy xuất

Toàn bộ gold answer và filter được lưu tại `docs/SHOPEE_POLICY_BENCHMARKS.md`.

| # | Câu hỏi | Gold answer rút gọn | Chiến lược tốt nhất | Kết quả |
|---|---|---|---|---|
| 1 | Hạn trả hàng / ngoại lệ thực phẩm | 15 ngày; thực phẩm 24 giờ | Sentence | Đạt |
| 2 | Phản hồi của người bán | 2 ngày lịch | Sentence | Đạt, evidence không top-1 |
| 3 | Phí tự gửi hàng hoàn | 25.000 / 40.000 Xu | FixedSize | Đạt; Sentence là failure case |
| 4 | Ảnh thật khi đăng bán | Ảnh tự chụp, ít nhất 40% diện tích | Sentence + filter seller | Đạt |
| 5 | Tài khoản giữ tiền / hoàn tiền | Tài Khoản Đảm Bảo; yêu cầu được chấp thuận | Sentence | Đạt |

Filter metadata giúp thu hẹp theo `buyer` ở Q3 và `seller` ở Q4. Q4 cải thiện
rõ vì loại các tài liệu buyer; Q3 vẫn cần chiến lược chunk tốt hơn vì filter
không thể bù cho việc hai mức phí bị tách khỏi câu hỏi trong ranking.

## 4. Demo và bài học

- Demo: ingest 6 tài liệu → 218 sentence chunks → filter buyer/seller →
  grounded agent với context có source/score.
- Failure case: Q3 của SentenceChunker trả về document đúng nhưng top-3 không
  chứa đồng thời `25,000` và `40,000`, nên agent không thể đưa gold answer đủ.
- Cải thiện đề xuất: thêm `HeadingChunker`/`FAQChunker` để giữ nguyên section
  “phí trả hàng”, hoặc giảm số câu mỗi chunk và thêm overlap theo heading.

## Tự đánh giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Lựa chọn tài liệu | 10 / 10 |
| Thiết kế chiến lược | 13 / 15 |
| Chất lượng truy xuất | 8 / 10 |
| Thuyết trình | 5 / 5 |
| **Tổng phần nhóm** | **36 / 40** |
