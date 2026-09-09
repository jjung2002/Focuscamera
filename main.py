import cv2 #OpenCV의 Python 모듈 불러오기
import mediapipe as mp  #얼굴감지 파이프라인
import time
import math
import statistics

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#눈 크기 함수
def landmark_distance(point1, point2, width, height):
    x1 = point1.x * width
    y1 = point1.y * height

    x2 = point2.x * width
    y2 = point2.y * height

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#눈 확인 함수
def calculate_ear(face, eye_indices, width, height):
    p1 = face[eye_indices[0]]
    p2 = face[eye_indices[1]]
    p3 = face[eye_indices[2]]
    p4 = face[eye_indices[3]]
    p5 = face[eye_indices[4]]
    p6 = face[eye_indices[5]]

    vertical1 = landmark_distance(p2, p6, width, height)
    vertical2 = landmark_distance(p3, p5, width, height)
    horizontal = landmark_distance(p1, p4, width, height)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

selected_landmarks = RIGHT_EYE + LEFT_EYE  #눈 점

MODEL_PATH = "models/face_landmarker.task" #모델파일 위치

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1 #감지 얼굴 수
)
landmarker = vision.FaceLandmarker.create_from_options(options)

print("Face Landmarker loaded successfully.")

cap = cv2.VideoCapture(0) #웹캠

if not cap.isOpened():
    print("Camera failed to open.")
    exit()

print("Camera connected successfully.")

start_time = time.monotonic() #시간 타이머
last_timestamp_ms = -1

CALIBRATION_SECONDS = 5 #기준값을 측정할 시간

calibration_start = None #측정 시작 시간
ear_samples = [] #5초동안 측정한 ear저장
baseline_ear = None #최종 개인 기준 ear

closed_start_time = None
closed_duration = 0.0
last_closed_duration = 0.0

blink_count = 0
eye_event = "None"

while True:
    success, frame = cap.read() #sucess는 boolean, frame은 이미지데이터

    if not success:
        print("Failed to capture frame.")
        break

    # OpenCV BGR -> RGB    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #색상 순서 변경

    #Mediapipe를 위해 OpenCV의 Numpy이미지를 mp.Image형식으로 
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((time.monotonic() - start_time) * 1000) # 현재 프레임 시간 계산

    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1

    last_timestamp_ms = timestamp_ms

    #-----------핵심코드
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )
    if result.face_landmarks:
        status = "Face Detected"

        height, width, _ = frame.shape

        face = result.face_landmarks[0]

        right_ear = calculate_ear(
            face,
            RIGHT_EYE,
            width,
            height
        )

        left_ear = calculate_ear(
            face,
            LEFT_EYE,
            width,
            height
        )
        average_ear=(right_ear + left_ear) / 2

        EYE_CLOSED_THRESHOLD = 0.20

        BLINK_MAX_DURATION = 0.30
        LONG_CLOSURE_THRESHOLD = 1.00   

        if baseline_ear is None:

            if calibration_start is None:
                calibration_start = time.monotonic()
                print("EAR calibration started.")

            ear_samples.append(average_ear)

            calibration_elapsed = time.monotonic() - calibration_start

            if calibration_elapsed >= CALIBRATION_SECONDS:
                baseline_ear = statistics.median(ear_samples)
                print(
                    f"Calibration complete. "
                    f"Baseline EAR: {baseline_ear:.3f}"
                )

        eye_ratio=None

        if baseline_ear is not None and baseline_ear >0:
            eye_ratio = average_ear / baseline_ear

        eye_status = "Unknown"

        if eye_ratio is not None:
            if eye_ratio < EYE_CLOSED_THRESHOLD:
                eye_status = "Eyes Closed"
            else:
                eye_status = "Eyes Open"

        # 눈 감김 시간 측정
        if eye_status == "Eyes Closed":

            if closed_start_time is None:
                closed_start_time = time.monotonic()

            closed_duration = time.monotonic() - closed_start_time

            if closed_duration >= LONG_CLOSURE_THRESHOLD:
                eye_event = "Long Closure"

        else:
            if closed_start_time is not None:
                last_closed_duration = closed_duration

                if last_closed_duration <= BLINK_MAX_DURATION:
                    eye_event = "Blink"
                    blink_count += 1
                elif last_closed_duration < LONG_CLOSURE_THRESHOLD:
                    eye_event = "Extended Closure"

                else:
                    eye_event = "Long Closure"

            else:
                eye_event = "None"

            closed_start_time = None
            closed_duration = 0.0

        


#-------------print
        print(
            f"Right EAR: {right_ear:.3f}, "
            f"Left EAR: {left_ear:.3f}, "
            f"Average: {average_ear:.3f}"
        )

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if baseline_ear is None:
            calibration_text = "Calibrating..."
        else:
            calibration_text = f"Baseline: {baseline_ear:.3f}"

        cv2.putText(
            frame,
            calibration_text,
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        if eye_ratio is not None:
            cv2.putText(
                frame,
                f"Eye Ratio: {eye_ratio:.2f}",
                (30,160),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                eye_status,
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Closed Time: {closed_duration: .2f}s",
                (30,230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )
            cv2.putText(
                frame,
                f"Last Close: {last_closed_duration:.2f}s",
                (30,265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )
            cv2.putText(
                frame,
                f"Eye Event: {eye_event}",
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Blink Count: {blink_count}",
                (30, 335),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        

        for index in selected_landmarks:
            landmark = result.face_landmarks[0][index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x,y),
                4,
                (0,255,0),
                -1
            )

            cv2.putText(  #얼굴 위 텍스트 출력
                frame,
                str(index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        

    else:
        status = "No face"

        if baseline_ear is None:
            calibration_start = None
            ear_samples.clear()
    #-----------
    cv2.putText(  #화면 텍스트 출력
        frame,
        status,
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,  
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Focus Camera", frame)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q") or key ==27 : #q누르면 반복문 종료
        break

cap.release()
cv2.destroyAllWindows()