from fastapi import FastAPI, File, UploadFile, Form
import cv2
import dlib
import numpy as np
import sqlite3
import io
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# Load Dlib models
shape_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Initialize database
conn = sqlite3.connect("face_database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
""")
conn.commit()

# Compute cosine similarity
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# Process image and extract face encoding
def process_image(image):
    image = Image.open(io.BytesIO(image))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    faces = detector(frame)
    if not faces:
        return None
    
    shape = sp(frame, faces[0])
    encoding = np.array(facerec.compute_face_descriptor(frame, shape))
    return encoding

# Train user
@app.post("/train")
async def train_user(name: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    encoding = process_image(image_data)

    if encoding is None:
        return {"message": "No face detected. Try another image."}

    cursor.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
    conn.commit()
    
    return {"message": f"User {name} registered successfully!"}

# Recognize user
@app.post("/recognize")
async def recognize_user(file: UploadFile = File(...)):
    image_data = await file.read()
    test_encoding = process_image(image_data)

    if test_encoding is None:
        return {"message": "No face detected. Try another image."}

    cursor.execute("SELECT name, encoding FROM users")
    users = cursor.fetchall()

    best_match_name = "Unknown"
    best_match_score = -1

    for name, db_encoding in users:
        db_encoding = np.frombuffer(db_encoding, dtype=np.float64)
        similarity = cosine_similarity(test_encoding, db_encoding)

        if similarity > best_match_score:
            best_match_score = similarity
            best_match_name = name

    if best_match_score > 0.75:
        return {"name": best_match_name, "match_percentage": f"{best_match_score*100:.2f}%"}
    
    return {"name": "Unknown", "match_percentage": "0%"}

# Run the app (for local testing)
# Command to run: `uvicorn filename:app --host 0.0.0.0 --port 8000`
