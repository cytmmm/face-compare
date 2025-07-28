import os
import insightface
import numpy as np
import faiss
import pickle

INDEX_PATH = "face_data/index.faiss"
IDS_PATH = "face_data/user_ids.pkl"


class FaceEngine:
    def __init__(self, vector_dim=512):
        self.dim = vector_dim
        self.model = insightface.app.FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=0)
        self.index = faiss.IndexFlatL2(self.dim)
        self.user_ids = []

        self._load_index()

    def extract(self, image_path):
        import cv2
        img = cv2.imread(image_path)
        faces = self.model.get(img)
        if not faces:
            return None
        return faces[0].embedding.astype('float32')

    def rebuild_index(self, all_entries):
        if not all_entries:
            self.index = faiss.IndexFlatL2(self.dim)
            self.user_ids = []
        else:
            vectors = np.array([v['vector'] for v in all_entries]).astype('float32')
            self.user_ids = [v['user_id'] for v in all_entries]
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(vectors)

        self._save_index()

    def search(self, query_vec, k=1):
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(query_vec.reshape(1, -1), k)
        return [
            {"user_id": self.user_ids[i], "distance": float(d)}
            for d, i in zip(D[0], I[0])
            if i < len(self.user_ids)
        ]

    def _save_index(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(IDS_PATH, "wb") as f:
            pickle.dump(self.user_ids, f)

    def _load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(IDS_PATH, "rb") as f:
                self.user_ids = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.user_ids = []
