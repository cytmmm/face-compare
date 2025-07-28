import sqlite3
import numpy as np
import os
import base64
import json

DB_PATH = "face_vectors.db"

class FaceDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                user_id TEXT PRIMARY KEY,
                vector TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def encode_vec(self, vec):
        return base64.b64encode(vec.tobytes()).decode()

    def decode_vec(self, s):
        raw = base64.b64decode(s)
        return np.frombuffer(raw, dtype='float32')

    def insert(self, user_id, vector):
        vec_str = self.encode_vec(vector)
        self.conn.execute("REPLACE INTO faces (user_id, vector) VALUES (?, ?)", (user_id, vec_str))
        self.conn.commit()

    def delete(self, user_id):
        cur = self.conn.execute("DELETE FROM faces WHERE user_id = ?", (user_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def get_all_users(self):
        rows = self.conn.execute("SELECT user_id FROM faces").fetchall()
        return [row[0] for row in rows]

    def get_all_vectors(self):
        rows = self.conn.execute("SELECT user_id, vector FROM faces").fetchall()
        return [{"user_id": row[0], "vector": self.decode_vec(row[1])} for row in rows]