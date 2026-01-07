import cv2
import face_recognition
import os
import numpy as np

# -----------------------------
# Load Known Faces
# -----------------------------
known_face_encodings = []
known_face_names = []

path = "known_faces"

for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(path, file)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file)[0])

print("Authorized Faces Loaded")

# -----------------------------
# Start Camera
# -----------------------------
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.5
        )
        name = "Access Denied"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw rectangle
        color = (0, 255, 0) if name != "Access Denied" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    cv2.imshow("AI Face Recognition Entry Agent", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
