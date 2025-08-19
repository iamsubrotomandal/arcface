import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import threading

class FaceDB:
    """Simple on-disk face embedding database (cosine similarity search).

    Files:
      embeddings: .npy float32 (N, D)
      metadata:   .json {"ids": [...], "version": 1}
    """
    def __init__(self, root: str = "data/facedb", dim: Optional[int] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.root / "embeddings.npy"
        self.meta_path = self.root / "metadata.json"
        self._lock = threading.Lock()
        self.dim = dim
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.ids: List[str] = []
        self._load()
        if self.dim is None and self.embeddings.size > 0:
            self.dim = self.embeddings.shape[1]

    def _load(self):
        if self.emb_path.exists() and self.meta_path.exists():
            try:
                self.embeddings = np.load(self.emb_path)
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                self.ids = meta.get('ids', [])
            except Exception:
                self.embeddings = np.empty((0, 0), dtype=np.float32)
                self.ids = []
        if len(self.ids) != (0 if self.embeddings.size == 0 else self.embeddings.shape[0]):
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.ids = []

    def _save(self):
        tmp_emb = self.emb_path.with_suffix('.tmp.npy')
        tmp_meta = self.meta_path.with_suffix('.tmp.json')
        np.save(tmp_emb, self.embeddings)
        with open(tmp_meta, 'w', encoding='utf-8') as f:
            json.dump({"ids": self.ids, "version": 1}, f)
        os.replace(tmp_emb, self.emb_path)
        os.replace(tmp_meta, self.meta_path)

    def count(self) -> int:
        return len(self.ids)

    def enroll(self, person_id: str, embedding: np.ndarray, allow_update: bool = True) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        if self.dim is None:
            self.dim = emb.shape[1]
        if emb.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {emb.shape[1]}")
        with self._lock:
            if person_id in self.ids:
                idx = self.ids.index(person_id)
                if allow_update:
                    old = self.embeddings[idx:idx+1]
                    new_vec = (old + emb) / 2.0
                    new_vec /= (np.linalg.norm(new_vec, axis=1, keepdims=True) + 1e-9)
                    self.embeddings[idx] = new_vec[0]
            else:
                if self.embeddings.size == 0:
                    self.embeddings = emb
                else:
                    self.embeddings = np.vstack([self.embeddings, emb])
                self.ids.append(person_id)
            self._save()

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b.T

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings.size == 0:
            return []
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = self._cosine(emb, self.embeddings)[0]
        k = min(top_k, sims.shape[0])
        idxs = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in idxs]

    def match(self, embedding: np.ndarray, threshold: float = 0.35) -> Tuple[Optional[str], float]:
        results = self.search(embedding, top_k=1)
        if not results:
            return None, 0.0
        pid, score = results[0]
        if score >= threshold:
            return pid, score
        return None, score

    def export(self) -> Dict[str, Any]:
        return {"count": self.count(), "dim": self.dim, "ids": list(self.ids)}

__all__ = ["FaceDB"]
