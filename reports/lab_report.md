# Báo cáo thực hành — Lab 19: Production-Grade GraphRAG vs Flat RAG

**Học viên:** Phạm Tiến Anh  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026

## 1. Thuyết minh kỹ thuật và phân tích lỗi

### 1.1 Coreference Resolution

Pipeline dùng quy tắc conservative: chỉ thay thế đại từ hoặc cụm “the company” khi tiền ngữ rõ ràng trong cùng chunk. Trường hợp mơ hồ được giữ nguyên và ghi vào `unresolved_mentions`. Nếu phân giải sai, NER/RE có thể tạo false edge, chẳng hạn gán quan hệ `INVESTED_IN` hoặc `ACQUIRED` cho nhầm công ty. Vì vậy hệ thống ưu tiên giữ nguyên văn bản khi không chắc chắn.

### 1.2 Entity Resolution và Lexical Guard

Ngưỡng vector được chọn là cosine similarity `0.90`. Candidate được tìm bằng FAISS `IndexFlatIP`, sau đó phải vượt lexical guard với tên đã bỏ hậu tố doanh nghiệp và `SequenceMatcher >= 0.72`. Manual aliases được ưu tiên, ví dụ `MSFT -> Microsoft`, `GOOGL -> Google`, `AAPL -> Apple`.

Lexical guard ngăn false merge giữa các tên gần nghĩa nhưng khác thực thể, ví dụ `Apple` và `Apple Music`, hoặc `Sam Altman` và `Steve Altman`. Các quyết định được lưu trong `entity_resolution_audit_df` với `MERGE_MANUAL`, `MERGE_VECTOR` hoặc `REJECT_GUARD`.

### 1.3 Super-node và provenance

Graph traversal dùng `max_hops=2`, `SUPER_NODE_DEGREE=100`, `SUPER_NODE_EDGE_CAP=50`, `GLOBAL_EDGE_CAP=250` và giới hạn context 14.000 ký tự. Với node degree > 100, hệ thống lấy tối đa 50 cạnh mới nhất theo `published_date`. Ưu điểm là tránh bùng nổ graph/token và ưu tiên thông tin mới; rủi ro là có thể bỏ sót một sự kiện lịch sử quan trọng.

Top-3 degree thực tế được xuất trong `outputs/top_degree_entities.csv` nếu cell 5.1 đã chạy. Không suy đoán các con số degree vì chúng phụ thuộc dataset/runtime. Mọi edge được nạp bằng `UNWIND` theo batch và có `source_chunk_id`, `published_date`, `evidence`, `confidence`; sanity check yêu cầu `invalid_provenance_edges = 0`.

### 1.4 Benchmark Flat RAG và GraphRAG

Số liệu dưới đây lấy từ `outputs/graphrag_vs_flatrag_summary.csv`:

| Nhóm | Metric | Flat RAG | GraphRAG |
|---|---|---:|---:|
| factoid | Comprehensiveness | 5.0 | 5.0 |
| factoid | Faithfulness | 5.0 | 5.0 |
| factoid | Multi-hop reasoning | 5.0 | 5.0 |
| factoid | Latency (s) | 1.081 | 0.968 |
| factoid | Token usage | 551.5 | 472.5 |
| multi-hop | Comprehensiveness | 5.0 | 5.0 |
| multi-hop | Faithfulness | 5.0 | 5.0 |
| multi-hop | Multi-hop reasoning | 5.0 | 5.0 |
| multi-hop | Latency (s) | 1.571 | 1.319 |
| multi-hop | Token usage | 641.0 | 533.0 |
| cross-doc | Comprehensiveness | 4.5 | 4.5 |
| cross-doc | Faithfulness | 4.5 | 5.0 |
| cross-doc | Multi-hop reasoning | 4.5 | 4.5 |
| cross-doc | Latency (s) | 1.828 | 2.416 |
| cross-doc | Token usage | 649.5 | 1378.0 |

GraphRAG cải thiện faithfulness ở nhóm cross-doc nhưng tốn token và latency hơn. Sample chỉ có 5 câu nên chưa đủ để khẳng định GraphRAG luôn tốt hơn.

**Flat RAG yếu hơn:** `G5000-07` về ServiceNow, Now Assist và AI Lighthouse. Flat RAG đạt 4.0 ở comprehensiveness/faithfulness/reasoning, GraphRAG đạt 5.0. Graph traversal giúp nối evidence về tính năng platform với chương trình hợp tác NVIDIA–Accenture.

**GraphRAG khó khăn:** `G5000-20` về Options Technology. GraphRAG vẫn đúng nhưng đạt 4.0 ở comprehensiveness và reasoning, đồng thời dùng 1.116 token so với 593 của Flat RAG. Nguyên nhân là graph context mang thêm provenance/quan hệ dù câu hỏi chỉ cần hai mốc thời gian. Có thể khắc phục bằng relation filtering và reranking theo câu hỏi.

### 1.5 Trade-offs, Agent control và scale

Flat RAG có indexing đơn giản, latency thấp và phù hợp factoid. GraphRAG có thêm chi phí extraction, entity resolution, Neo4j ingestion và traversal nhưng hỗ trợ provenance, multi-hop và cross-document tốt hơn. Khi scale 350MB/~100.000 bài báo, bottleneck đầu tiên là LLM extraction và entity resolution. Giải pháp là streaming, async batch queue, cache theo hash, FAISS/HNSW có blocking, Neo4j `UNWIND`, checkpoint/resume và community partitioning.

Một quyết định kiểm soát AI Agent là không dùng pairwise cosine `O(N²)` trên toàn dataset vì nguy cơ OOM và false merge; thay vào đó dùng ANN, lexical guard và Union-Find.

## 2. Reflection và Action Plan

### 2.1 Mapping bài giảng vào code

| Khái niệm | Module | Hàm/khối code |
|---|---|---|
| Conservative Coreference | M1 | `resolve_coref_batch`, `run_coref` |
| Schema/Allowlist Guard | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` |
| Bulk Cypher | M2 | `bulk_insert_nodes`, `bulk_insert_edges` |
| Entity Resolution | M3 | `build_resolution_map`, `UF` |
| Super-node cap | M4 | `retrieve_graph_context` |
| LLM-as-a-Judge | M5 | `judge_answer`, `run_evaluation` |

### 2.2 Debugging và bài học

Lỗi khó nhất là schema dữ liệu thực tế không khớp danh sách cột loader dự kiến. Cách xử lý là chuẩn hóa cột linh hoạt, tự dò cột văn bản dài nhất khi cần, rồi kiểm tra lần lượt `news_df`, `chunks_df`, `raw_triples_df` và `triples_df`. Bài học thứ hai là nhiều cell chỉ định nghĩa hàm; phải chạy các dòng thực thi cuối cell theo dependency order.

### 2.3 Action Plan

Với hệ thống hỏi đáp tài liệu doanh nghiệp, GraphRAG phù hợp khi câu hỏi cần nối người, tổ chức, sản phẩm, sự kiện và thời gian; câu hỏi tìm một đoạn văn đơn lẻ chỉ cần Flat RAG.

**Nodes:** `Document`, `Chunk`, `Person`, `Organization`, `Product`, `Technology`, `Event`, `Date`.  
**Relations:** `MENTIONS`, `WORKS_AT`, `FOUNDED`, `ACQUIRED`, `USES`, `DEVELOPED`, `PARTNERED_WITH`, `OCCURRED_ON`.

Entity Resolution sẽ dùng alias thủ công, ANN, lexical guard và audit log. Super-node sẽ dùng degree cap, temporal reranking, relation filtering và community partitioning.

## 3. Tự đánh giá

| Tiêu chí | Điểm |
|---|---:|
| Hiểu GraphRAG | 4/5 |
| Kiểm soát AI Coding Agent | 4/5 |
| Chất lượng Knowledge Graph | 4/5 |
| Debug và phân tích hệ thống | 4/5 |

## 4. Tệp kết quả

- `outputs/graphrag_eval_results.csv`
- `outputs/graphrag_vs_flatrag_summary.csv`
- `outputs/entity_resolution_audit.csv`
- `outputs/top_degree_entities.csv` nếu đã xuất từ cell 5.1
- `outputs/graph_integrity_summary.csv` nếu đã xuất từ cell 5.1
