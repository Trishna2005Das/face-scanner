import cv2
import dlib
import numpy as np
import sqlite3
import time
import os

# Print the OpenCV version
print("OpenCV version:", cv2.__version__)

# Define file paths
shape_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
try:
    sp = dlib.shape_predictor(shape_predictor_path)
except RuntimeError:
    print(f"Error: Unable to open {shape_predictor_path}. Please ensure the file is in the correct location.")
    exit(1)

try:
    facerec = dlib.face_recognition_model_v1(face_recognition_model_path)
except RuntimeError:
    print(f"Error: Unable to open {face_recognition_model_path}. Please ensure the file is in the correct location.")
    exit(1)

# Initialize SQLite Database
conn = sqlite3.connect("face_database.db")
cursor = conn.cursor()

# Create Table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
""")
conn.commit()

# Capture images for training with real-time feedback
def capture_training_images(name):
    cap = cv2.VideoCapture(0)
    images = []
    start_time = time.time()

    print(f"Look into the camera for 20 seconds, {name}...")

    while time.time() - start_time < 20:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Capturing {len(images)}/40", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Feed", frame)

        if len(images) < 40 and (len(images) == 0 or time.time() - start_time >= len(images) * 0.5):
            images.append(frame.copy())
            print(f"Captured {len(images)}/40 images...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return images

# Train HOG model
def train_user(name):
    images = capture_training_images(name)
    encodings = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(gray)

        if faces:
            shape = sp(gray, faces[0])
            encoding = facerec.compute_face_descriptor(gray, shape)
            encodings.append(np.array(encoding))

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)  # Average face encoding

        # Store in SQLite
        cursor.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, avg_encoding.tobytes()))
        conn.commit()
        print(f"User {name} registered successfully!")

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


# Test Face Recognition with Bounding Box
def recognize_user():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(gray)

        for face in faces:
            shape = sp(gray, face)
            test_encoding = np.array(facerec.compute_face_descriptor(gray, shape))

            cursor.execute("SELECT name, encoding FROM users")
            users = cursor.fetchall()

            best_match_name = "Unknown"
            best_match_score = -1  # Cosine similarity ranges from -1 to 1

            for name, db_encoding in users:
                db_encoding = np.frombuffer(db_encoding, dtype=np.float64)
                similarity = cosine_similarity(test_encoding, db_encoding)
                match_percentage = similarity * 100  # Convert to percentage

                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_name = name

            if best_match_score > 0.75:  # Adjust threshold for better accuracy
                recognized_name = best_match_name
            else:
                recognized_name = "Unknown"

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display name above the face
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display match percentage below the face
            cv2.putText(frame, f"{best_match_score*100:.2f}%", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    operation = input("Enter operation (train/test): ").strip().lower()

    if operation == "train":
        user_name = input("Enter your name: ").strip()
        train_user(user_name)

    elif operation == "test":
        recognize_user()
