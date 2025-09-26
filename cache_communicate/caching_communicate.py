import os
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
from models.embedder import get_embedding_model_HF


# Thư mục lưu trữ dữ liệu bộ nhớ đệm
STORAGE_PATH = "cache_communicate/data_store"
os.makedirs(STORAGE_PATH, exist_ok=True)

# Ngưỡng xác định mức độ tương đồng
SIM_THRESHOLD = 0.85

# Mô hình sinh embedding
embedder = get_embedding_model_HF()


def _generate_id(text: str) -> str:
    """Tạo ID duy nhất từ văn bản."""
    content = text.strip().lower().encode("utf-8")
    return hashlib.md5(content).hexdigest()


def _get_vector(text: str) -> np.ndarray:
    """Sinh vector embedding từ văn bản."""
    if hasattr(embedder, "encode"):  # HuggingFace SentenceTransformer
        vec = embedder.encode(text)
    # elif hasattr(embedder, "embed_query"):  # LangChain Embeddings
    #     vec = embedder.embed_query(text)
    else:
        raise AttributeError("Embedder không hỗ trợ encode hoặc embed_query.")
    return np.array(vec).flatten()  # luôn đảm bảo vector 1D


def _similarity_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính toán độ tương đồng cosine giữa hai vector."""
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def retrieve_from_cache(query: str) -> Optional[Dict[str, Any]]:
    """Tìm kiếm phản hồi từ bộ nhớ đệm, có thể dùng tìm kiếm ngữ nghĩa."""
    query_id = _generate_id(query)
    file_path = os.path.join(STORAGE_PATH, f"{query_id}.json")

    # Kiểm tra khớp tuyệt đối
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Dò tìm tương đồng ngữ nghĩa
    input_vector = _get_vector(query)
    top_match = None
    top_score = 0.0

    for f_name in os.listdir(STORAGE_PATH):
        if not f_name.endswith(".json"):
            continue

        file_full_path = os.path.join(STORAGE_PATH, f_name)
        with open(file_full_path, "r", encoding="utf-8") as f:
            try:
                item = json.load(f)
                stored_vector = np.array(item.get("embedding", []))
                if stored_vector.size == 0:
                    continue
                score = _similarity_score(input_vector, stored_vector)

                if score >= SIM_THRESHOLD and score > top_score:
                    top_match = item
                    top_score = score
            except Exception:
                continue  # Bỏ qua file lỗi

    return top_match


def store_response(query: str, result: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
    """Lưu kết quả vào bộ nhớ đệm."""
    entry_id = _generate_id(query)
    entry = {
        "prompt": query,
        "response": result,
        "embedding": _get_vector(query).tolist(),
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": extra_info or {}
    }

    file_path = os.path.join(STORAGE_PATH, f"{entry_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


def purge_cache() -> None:
    """Xóa toàn bộ bộ nhớ đệm."""
    for f in os.listdir(STORAGE_PATH):
        if f.endswith(".json"):
            os.remove(os.path.join(STORAGE_PATH, f))
