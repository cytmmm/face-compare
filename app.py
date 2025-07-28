from fastapi import FastAPI, UploadFile, File, HTTPException
from face_engine import FaceEngine
from db import FaceDB
import shutil
import os

app = FastAPI()
engine = FaceEngine()
db = FaceDB()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/register")
def register(user_id: str, file: UploadFile = File(...)):
    image_path = os.path.join(UPLOAD_DIR, f"{user_id}.jpg")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vec = engine.extract(image_path)
    if vec is None:
        raise HTTPException(status_code=400, detail="No face detected")

    db.insert(user_id, vec)
    engine.rebuild_index(db.get_all_vectors())
    return {"status": "registered", "user_id": user_id}


@app.post("/compare")
def compare(file: UploadFile = File(...)):
    image_path = os.path.join(UPLOAD_DIR, "query.jpg")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vec = engine.extract(image_path)
    if vec is None:
        raise HTTPException(status_code=400, detail="No face detected")

    top_matches = engine.search(vec, k=2)
    return {"results": top_matches}


@app.get("/users")
def list_users():
    return db.get_all_users()


@app.delete("/delete/{user_id}")
def delete_user(user_id: str):
    if not db.delete(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    engine.rebuild_index(db.get_all_vectors())
    return {"status": "deleted", "user_id": user_id}


@app.post("/rebuild_index")
def rebuild_index():
    engine.rebuild_index(db.get_all_vectors())
    return {"status": "faiss index rebuilt"}