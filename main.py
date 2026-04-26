import cv2
import time
from scipy.spatial import distance
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
import winsound

# ------------------ EAR LOGIC ------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)
 
EAR_THRESHOLD = 0.25
DROWSY_TIME = 2  # seconds
eye_closed_start = None
alarm_playing = False

# ------------------ MEDIAPIPE TASKS ------------------

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ------------------ CAMERA ------------------

cap = cv2.VideoCapture(0)
print("Camera started successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔴 IMPORTANT FIX: wrap numpy array into MediaPipe Image
    mp_image = Image(
    image_format=ImageFormat.SRGB,
    data=rgb
)


    result = landmarker.detect(mp_image)

    status = "AWAKE"

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_eye = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in LEFT_EYE
        ]
        right_eye = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in RIGHT_EYE
        ]

        ear = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2

        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start >= DROWSY_TIME:
                status = "DROWSY"

                if not alarm_playing:
                    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC | winsound.SND_LOOP | winsound.SND_FILENAME)
                    alarm_playing = True
        else:
            eye_closed_start = None
            
            if alarm_playing:
                winsound.PlaySound(None, winsound.SND_PURGE)
                alarm_playing = False

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
    cv2.putText(
        frame,
        f"STATUS: {status}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("Drowsiness Detection - MediaPipe Tasks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# print("hello world  how are you guys my name is babul why are you guys going alone for the ride take me also i am feeling bored over here living alone please take me also you will enjoy outside and i will just lie over here doing nothing and i will have to eat that all days repeating boring food for the dinner but you guys will enjoy tasty food outside i think so you all are going to enjoy momos and tea.")
# print("Hello World this is babul this side")