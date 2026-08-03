"""
demo_app.py — Web UI demo cho Lab 7 (Cao Minh Quang, nhóm B4.1).

Chạy trực tiếp code trong `src/CaoMinhQuang/`: chunking -> embedding ->
EmbeddingStore.search / search_with_filter -> KnowledgeBaseAgent.

Chỉ dùng thư viện chuẩn của Python (http.server) + python-dotenv đã có sẵn
trong requirements.txt — KHÔNG cần cài thêm Flask/FastAPI.

    python3 demo_app.py                 # mở http://127.0.0.1:8000
    python3 demo_app.py --port 8080
    LAB_DATA_DIR=data/k4_ecommerce python3 demo_app.py

Backend nhúng lấy theo EMBEDDING_PROVIDER trong .env (mock | local | openai),
giống main.py. Embedding được cache xuống .demo_cache/ nên lần build sau
gần như tức thì và không tốn thêm chi phí API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import unicodedata
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from ingest import parse_front_matter
from src.CaoMinhQuang import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "demo"
CACHE_DIR = ROOT / ".demo_cache"
DEFAULT_DATA_DIR = os.getenv("LAB_DATA_DIR", "data/shopee_policy")

# 5 câu hỏi đánh giá của nhóm — xem docs/SHOPEE_POLICY_BENCHMARKS.md.
# `expect` là các mảnh gold answer dùng để tự động kiểm tra evidence trong chunk.
BENCHMARKS = [
    {
        "id": 1,
        "query": "Người mua có bao lâu để gửi yêu cầu trả hàng/hoàn tiền sau khi đơn giao thành công? Thực phẩm tươi sống hoặc đông lạnh thì sao?",
        "filter": None,
        "gold": "15 ngày với sản phẩm thông thường; 24 giờ với thực phẩm tươi sống và đông lạnh.",
        "expect": ["15 ngày", "24 giờ"],
    },
    {
        "id": 2,
        "query": "Khi không đồng ý với quyết định hoàn tiền, người bán phải phản hồi trong bao lâu?",
        "filter": None,
        "gold": "Trong vòng 2 ngày lịch kể từ ngày nhận thông báo của Shopee.",
        "expect": ["2 ngày"],
    },
    {
        "id": 3,
        "query": "Nếu người mua tự sắp xếp gửi hàng hoàn cho đơn không thuộc Shopee Mall, mức hỗ trợ phí là bao nhiêu?",
        "filter": "buyer",
        "gold": "25.000 Xu nếu cùng tỉnh/thành phố, 40.000 Xu nếu khác tỉnh/thành phố.",
        "expect": ["25.000", "40.000"],
    },
    {
        "id": 4,
        "query": "Người bán cần bảo đảm gì về ảnh thật của sản phẩm khi đăng bán?",
        "filter": "seller",
        "gold": "Ít nhất một ảnh thật do người bán tự chụp; sản phẩm chiếm ít nhất 40% diện tích ảnh.",
        "expect": ["40%"],
    },
    {
        "id": 5,
        "query": "Khoản tiền người mua đã thanh toán được lưu ở đâu trước khi chuyển cho người bán?",
        "filter": None,
        "gold": "Trong Tài Khoản Đảm Bảo của Shopee; hoàn cho người mua nếu yêu cầu trả hàng được chấp thuận.",
        "expect": ["tài khoản đảm bảo"],
    },
]

STRATEGIES = {
    "sentence": {
        "label": "SentenceChunker",
        "note": "Gom 5 câu / chunk — chiến lược cá nhân tôi chọn",
        "param": {"key": "max_sentences", "label": "Số câu mỗi chunk", "default": 5, "min": 1, "max": 20},
        "build": lambda p: SentenceChunker(max_sentences_per_chunk=int(p.get("max_sentences", 5))),
    },
    "fixed": {
        "label": "FixedSizeChunker",
        "note": "750 ký tự, overlap 100 — baseline",
        "param": {"key": "chunk_size", "label": "Kích thước chunk", "default": 750, "min": 100, "max": 3000},
        "build": lambda p: FixedSizeChunker(chunk_size=int(p.get("chunk_size", 750)), overlap=100),
    },
    "recursive": {
        "label": "RecursiveChunker",
        "note": "750 ký tự, tách theo đoạn → câu → từ",
        "param": {"key": "chunk_size", "label": "Kích thước chunk", "default": 750, "min": 100, "max": 3000},
        "build": lambda p: RecursiveChunker(chunk_size=int(p.get("chunk_size", 750))),
    },
}


# --------------------------------------------------------------------------- #
# Embedding backend (giống main.py) + cache xuống đĩa
# --------------------------------------------------------------------------- #

class CachedEmbedder:
    """Bọc một embedder thật, nhớ lại vector đã tính để khỏi gọi API hai lần."""

    def __init__(self, inner, backend_name: str) -> None:
        self._inner = inner
        self._backend_name = backend_name
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", backend_name)
        CACHE_DIR.mkdir(exist_ok=True)
        self._path = CACHE_DIR / f"emb_{safe}.json"
        self._lock = threading.Lock()
        self._dirty = False
        try:
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}

    def __call__(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vector = self._inner(text)
        with self._lock:
            self._cache[key] = vector
            self._dirty = True
        return vector

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._path.write_text(json.dumps(self._cache), encoding="utf-8")
            self._dirty = False


def select_embedder():
    """Chọn backend nhúng theo EMBEDDING_PROVIDER, có fallback về mock."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    note = ""
    embedder = _mock_embed

    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as exc:
            note = f"Local embedder không sẵn sàng ({exc.__class__.__name__}); đang dùng mock."
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as exc:
            note = f"OpenAI embedder không sẵn sàng ({exc.__class__.__name__}); đang dùng mock."

    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    is_mock = embedder is _mock_embed
    return CachedEmbedder(embedder, backend), backend, is_mock, note


def make_llm():
    """LLM cho KnowledgeBaseAgent: dùng OpenAI nếu có key, không thì trả stub."""
    if not os.getenv("OPENAI_API_KEY"):
        return _stub_llm, False
    try:
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        def llm(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""

        return llm, True
    except Exception:
        return _stub_llm, False


def _stub_llm(prompt: str) -> str:
    return (
        "[Chưa nối LLM thật] Prompt đã được dựng đầy đủ với context truy xuất được. "
        "Đặt OPENAI_API_KEY trong .env để agent sinh câu trả lời grounded.\n\n"
        f"— Độ dài prompt: {len(prompt)} ký tự."
    )


# --------------------------------------------------------------------------- #
# Corpus + kho vector theo từng chiến lược
# --------------------------------------------------------------------------- #

def load_corpus(data_dir: str) -> list[Document]:
    """Đọc .md/.txt + YAML front matter thành Document của src/CaoMinhQuang."""
    documents: list[Document] = []
    for path in sorted(Path(data_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = str(metadata.get("doc_id") or path.stem)
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("source", str(path))
        documents.append(Document(id=doc_id, content=body, metadata=metadata))
    return documents


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _evidence_pattern(term: str) -> re.Pattern:
    """Regex khoan dung để tìm một mảnh gold answer trong chunk.

    Corpus viết "25,000" còn gold answer trong benchmark viết "25.000", nên
    giữa hai chữ số bất kỳ ta cho phép có/không có dấu phân cách. Phần chữ so
    khớp không phân biệt hoa thường và đã chuẩn hoá Unicode về NFC.
    """
    clean = re.sub(r"(?<=\d)[.,](?=\d)", "", _nfc(term))
    parts: list[str] = []
    for index, char in enumerate(clean):
        parts.append(re.escape(char))
        if char.isdigit() and index + 1 < len(clean) and clean[index + 1].isdigit():
            parts.append("[.,  ]?")
    return re.compile("".join(parts), re.IGNORECASE)


class DemoState:
    """Giữ corpus, embedder và các store đã build (mỗi chiến lược một store)."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.documents = load_corpus(data_dir)
        self.embedder, self.backend, self.is_mock, self.backend_note = select_embedder()
        self.llm_fn, self.llm_real = make_llm()
        self.builds: dict[str, dict] = {}
        self.lock = threading.Lock()

    # -- build ------------------------------------------------------------- #

    def build_key(self, strategy: str, params: dict) -> str:
        param_key = STRATEGIES[strategy]["param"]["key"]
        return f"{strategy}:{params.get(param_key, STRATEGIES[strategy]['param']['default'])}"

    def get_build(self, strategy: str, params: dict) -> dict:
        return self.builds.get(self.build_key(strategy, params), {"state": "empty"})

    def start_build(self, strategy: str, params: dict) -> dict:
        key = self.build_key(strategy, params)
        with self.lock:
            existing = self.builds.get(key)
            if existing and existing["state"] in {"running", "ready"}:
                return existing
            job = {"state": "running", "done": 0, "total": 0, "key": key}
            self.builds[key] = job

        thread = threading.Thread(target=self._build, args=(strategy, params, job), daemon=True)
        thread.start()
        return job

    def _build(self, strategy: str, params: dict, job: dict) -> None:
        try:
            chunker = STRATEGIES[strategy]["build"](params)
            chunk_docs: list[Document] = []
            for doc in self.documents:
                for index, piece in enumerate(chunker.chunk(doc.content)):
                    meta = dict(doc.metadata)
                    meta["doc_id"] = doc.id
                    meta["chunk_index"] = index
                    chunk_docs.append(
                        Document(id=f"{doc.id}::chunk_{index}", content=piece, metadata=meta)
                    )

            job["total"] = len(chunk_docs)
            store = EmbeddingStore(collection_name=f"demo_{job['key']}", embedding_fn=self.embedder)

            # Nạp theo lô để cập nhật tiến độ cho UI.
            for start in range(0, len(chunk_docs), 10):
                store.add_documents(chunk_docs[start : start + 10])
                job["done"] = min(start + 10, len(chunk_docs))

            self.embedder.flush()

            lengths = [len(c.content) for c in chunk_docs]
            job.update(
                state="ready",
                store=store,
                chunks=chunk_docs,
                stats={
                    "count": len(chunk_docs),
                    "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
                    "min_length": min(lengths) if lengths else 0,
                    "max_length": max(lengths) if lengths else 0,
                },
            )
        except Exception as exc:  # hiển thị lỗi lên UI thay vì chết âm thầm
            job.update(state="error", error=f"{exc.__class__.__name__}: {exc}")

    # -- search ------------------------------------------------------------ #

    def search(self, strategy: str, params: dict, query: str, top_k: int, role: str | None,
               expect: list[str], with_agent: bool) -> dict:
        job = self.get_build(strategy, params)
        if job.get("state") != "ready":
            return {"error": "chưa build xong", "state": job.get("state", "empty")}

        store: EmbeddingStore = job["store"]
        metadata_filter = {"customer_role": role} if role else None

        if metadata_filter:
            results = store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
            pool = sum(1 for c in job["chunks"] if c.metadata.get("customer_role") == role)
        else:
            results = store.search(query, top_k=top_k)
            pool = len(job["chunks"])

        patterns = [(term, _evidence_pattern(term)) for term in expect]
        payload = []
        found: set[str] = set()
        for rank, result in enumerate(results, start=1):
            body = _nfc(result["content"])
            hits: list[str] = []
            marks: list[str] = []
            for term, pattern in patterns:
                surfaces = pattern.findall(body)
                if surfaces:
                    hits.append(term)
                    marks.extend(dict.fromkeys(surfaces))  # dạng chữ thật trong chunk, để bôi vàng
            found.update(hits)
            meta = result["metadata"]
            payload.append({
                "rank": rank,
                "score": round(float(result["score"]), 4),
                "content": body,
                "length": len(body),
                "marks": sorted(set(marks), key=len, reverse=True),
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title", meta.get("doc_id")),
                "customer_role": meta.get("customer_role"),
                "category": meta.get("category"),
                "chunk_index": meta.get("chunk_index"),
                "source_url": meta.get("source_url"),
                "hits": hits,
            })

        answer = ""
        if with_agent and results:
            agent = KnowledgeBaseAgent(store=store, llm_fn=self.llm_fn)
            try:
                answer = agent.answer(query, top_k=top_k)
            except Exception as exc:
                answer = f"[Lỗi gọi LLM] {exc.__class__.__name__}: {exc}"

        return {
            "results": payload,
            "answer": answer,
            "stats": job["stats"],
            "pool": pool,
            "total_chunks": len(job["chunks"]),
            "expect": expect,
            "found": [term for term in expect if term in found],
            "missing": [term for term in expect if term not in found],
        }


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    state: DemoState = None  # gán ở main()

    def log_message(self, fmt, *args):  # bớt ồn trên terminal khi demo
        if "/api/" in self.path or self.path == "/favicon.ico":
            return
        super().log_message(fmt, *args)

    # -- helpers ----------------------------------------------------------- #

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _params(self, data: dict) -> dict:
        return {k: v for k, v in data.items() if k in {"max_sentences", "chunk_size"}}

    # -- routes ------------------------------------------------------------ #

    def do_GET(self):
        route = urlparse(self.path)
        if route.path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif route.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif route.path == "/api/status":
            self._send(self._status())
        elif route.path == "/api/build":
            query = parse_qs(route.query)
            strategy = query.get("strategy", ["sentence"])[0]
            params = {k: v[0] for k, v in query.items() if k in {"max_sentences", "chunk_size"}}
            job = self.state.get_build(strategy, params)
            self._send({
                "state": job.get("state", "empty"),
                "done": job.get("done", 0),
                "total": job.get("total", 0),
                "stats": job.get("stats"),
                "error": job.get("error"),
            })
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        route = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send({"error": "JSON không hợp lệ"}, 400)
            return

        strategy = data.get("strategy", "sentence")
        if strategy not in STRATEGIES:
            self._send({"error": f"chiến lược không hợp lệ: {strategy}"}, 400)
            return
        params = self._params(data)

        if route.path == "/api/build":
            job = self.state.start_build(strategy, params)
            self._send({"state": job.get("state"), "done": job.get("done", 0), "total": job.get("total", 0)})
        elif route.path == "/api/search":
            query = (data.get("query") or "").strip()
            if not query:
                self._send({"error": "câu hỏi trống"}, 400)
                return
            result = self.state.search(
                strategy=strategy,
                params=params,
                query=query,
                top_k=max(1, min(10, int(data.get("top_k", 3)))),
                role=data.get("role") or None,
                expect=data.get("expect") or [],
                with_agent=bool(data.get("with_agent", True)),
            )
            self._send(result)
        else:
            self.send_error(404, "Not found")

    def _status(self) -> dict:
        state = self.state
        corpus = [
            {
                "doc_id": doc.id,
                "title": doc.metadata.get("title", doc.id),
                "chars": len(doc.content),
                "customer_role": doc.metadata.get("customer_role"),
                "category": doc.metadata.get("category"),
                "source_url": doc.metadata.get("source_url"),
            }
            for doc in state.documents
        ]
        roles = sorted({d["customer_role"] for d in corpus if d["customer_role"]})
        return {
            "backend": state.backend,
            "is_mock": state.is_mock,
            "backend_note": state.backend_note,
            "llm_real": state.llm_real,
            "data_dir": state.data_dir,
            "corpus": corpus,
            "total_chars": sum(d["chars"] for d in corpus),
            "roles": roles,
            "benchmarks": BENCHMARKS,
            "strategies": [
                {"id": key, "label": value["label"], "note": value["note"], "param": value["param"]}
                for key, value in STRATEGIES.items()
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Web UI demo cho Lab 7")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"Không tìm thấy thư mục dữ liệu: {args.data_dir}")
        return 1

    Handler.state = DemoState(args.data_dir)
    state = Handler.state

    print("=" * 62)
    print("  Lab 7 — RAG Demo | Cao Minh Quang, nhóm B4.1")
    print("=" * 62)
    print(f"  Dữ liệu     : {args.data_dir} ({len(state.documents)} tài liệu)")
    print(f"  Embedding   : {state.backend}")
    if state.backend_note:
        print(f"                {state.backend_note}")
    if state.is_mock:
        print("  CẢNH BÁO    : mock embeddings — kết quả KHÔNG có ý nghĩa ngữ nghĩa.")
        print("                Đặt EMBEDDING_PROVIDER=openai + OPENAI_API_KEY trong .env.")
    print(f"  Agent LLM   : {'OpenAI chat' if state.llm_real else 'chưa nối (stub)'}")
    print(f"  Mở trình duyệt tại: http://{args.host}:{args.port}")
    print("=" * 62)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng demo.")
        state.embedder.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
