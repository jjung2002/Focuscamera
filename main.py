import cv2
import mediapipe as mp
import time
import math
import statistics

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 설정값
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

# EAR 계산에 사용할 눈 랜드마크
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

SELECTED_LANDMARKS = RIGHT_EYE + LEFT_EYE


HEAD_LANDMARKS = [1, 10, 152, 234, 454]

# 초기 사용자 눈 상태 측정 시간
CALIBRATION_SECONDS = 5

# 현재 프로토타입용 임계값
EYE_CLOSED_THRESHOLD = 0.20

# 실제 테스트 결과를 기반으로 잡은 임시 시간 기준
BLINK_MAX_DURATION = 0.30
LONG_CLOSURE_THRESHOLD = 1.00


# ============================================================
# 눈 분석 함수
# ============================================================

def landmark_distance(point1, point2, width, height):
    """
    두 MediaPipe 랜드마크 사이의 픽셀 기준 유클리드 거리를 계산한다.
    """

    x1 = point1.x * width
    y1 = point1.y * height

    x2 = point2.x * width
    y2 = point2.y * height

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def calculate_ear(face, eye_indices, width, height):
    """
    눈 랜드마크 6개를 이용해
    EAR(Eye Aspect Ratio)을 계산한다.
    """

    p1 = face[eye_indices[0]]
    p2 = face[eye_indices[1]]
    p3 = face[eye_indices[2]]
    p4 = face[eye_indices[3]]
    p5 = face[eye_indices[4]]
    p6 = face[eye_indices[5]]

    vertical1 = landmark_distance(
        p2, p6, width, height
    )

    vertical2 = landmark_distance(
        p3, p5, width, height
    )

    horizontal = landmark_distance(
        p1, p4, width, height
    )

    ear = (
        vertical1 + vertical2
    ) / (2.0 * horizontal)

    return ear


def draw_eye_landmarks(frame, face, width, height):
    """
    디버깅을 위해 눈 랜드마크 위치와 번호를 화면에 표시한다.
    """

    for index in SELECTED_LANDMARKS:

        landmark = face[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            str(index),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


def draw_head_landmarks(frame, face, width, height):
    """
    Head 방향 분석에 사용할 얼굴 기준점을 표시한다.
    """

    for index in HEAD_LANDMARKS:
        landmark = face[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            5,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1
        )
# ============================================================
# MediaPipe Face Landmarker 초기화
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)

print("Face Landmarker loaded successfully.")


# ============================================================
# 웹캠 초기화
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()

print("Camera connected successfully.")


# ============================================================
# 시간 관련 변수
# ============================================================

# MediaPipe VIDEO 모드 timestamp
start_time = time.monotonic()
last_timestamp_ms = -1


# ============================================================
# EAR 자동 Calibration 변수
# ============================================================

calibration_start = None
ear_samples = []
baseline_ear = None


# ============================================================
# 눈 감김 / Blink 관련 변수
# ============================================================

closed_start_time = None
closed_duration = 0.0
last_closed_duration = 0.0

blink_count = 0
eye_event = "None"


# ============================================================
# 메인 루프
# ============================================================

while True:

    # --------------------------------------------------------
    # 1. 카메라 프레임 읽기
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame.")
        break


    # --------------------------------------------------------
    # 2. OpenCV 이미지 → MediaPipe 이미지
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # 3. MediaPipe VIDEO timestamp 계산
    # --------------------------------------------------------

    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1

    last_timestamp_ms = timestamp_ms


    # --------------------------------------------------------
    # 4. 얼굴 랜드마크 검출
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # 기본 상태
    status = "No Face"

    average_ear = None
    eye_ratio = None
    eye_status = "Unknown"


    # --------------------------------------------------------
    # 5. 얼굴이 검출된 경우
    # --------------------------------------------------------

    if result.face_landmarks:

        status = "Face Detected"

        height, width, _ = frame.shape

        face = result.face_landmarks[0]


        # ----------------------------------------------------
        # 5-1. 양쪽 눈 EAR 계산
        # ----------------------------------------------------

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

        average_ear = (
            right_ear + left_ear
        ) / 2


        # ----------------------------------------------------
        # 5-2. 사용자 EAR 자동 Calibration
        # ----------------------------------------------------

        if baseline_ear is None:

            if calibration_start is None:

                calibration_start = time.monotonic()

                print(
                    "EAR calibration started."
                )

            ear_samples.append(
                average_ear
            )

            calibration_elapsed = (
                time.monotonic()
                - calibration_start
            )

            if (
                calibration_elapsed
                >= CALIBRATION_SECONDS
            ):

                baseline_ear = statistics.median(
                    ear_samples
                )

                print(
                    "Calibration complete. "
                    f"Baseline EAR: "
                    f"{baseline_ear:.3f}"
                )


        # ----------------------------------------------------
        # 5-3. Eye Ratio 계산
        # ----------------------------------------------------

        if (
            baseline_ear is not None
            and baseline_ear > 0
        ):

            eye_ratio = (
                average_ear
                / baseline_ear
            )


        # ----------------------------------------------------
        # 5-4. 눈 Open / Closed 판정
        # ----------------------------------------------------

        if eye_ratio is not None:

            if (
                eye_ratio
                < EYE_CLOSED_THRESHOLD
            ):
                eye_status = "Eyes Closed"

            else:
                eye_status = "Eyes Open"


        # ----------------------------------------------------
        # 5-5. 눈 감긴 시간 측정
        # ----------------------------------------------------

        if eye_status == "Eyes Closed":

            if closed_start_time is None:
                closed_start_time = (
                    time.monotonic()
                )

            closed_duration = (
                time.monotonic()
                - closed_start_time
            )

            if (
                closed_duration
                >= LONG_CLOSURE_THRESHOLD
            ):
                eye_event = "Long Closure"


        elif eye_status == "Eyes Open":

            if closed_start_time is not None:

                last_closed_duration = (
                    closed_duration
                )

                if (
                    last_closed_duration
                    <= BLINK_MAX_DURATION
                ):

                    eye_event = "Blink"
                    blink_count += 1

                elif (
                    last_closed_duration
                    < LONG_CLOSURE_THRESHOLD
                ):

                    eye_event = (
                        "Extended Closure"
                    )

                else:

                    eye_event = (
                        "Long Closure"
                    )

            else:
                eye_event = "None"

            closed_start_time = None
            closed_duration = 0.0


        # ----------------------------------------------------
        # 5-6. 눈 랜드마크 표시
        # ----------------------------------------------------

        draw_eye_landmarks(
            frame,
            face,
            width,
            height
        )

        draw_head_landmarks(
            frame,
            face,
            width,
            height
        )   


    # --------------------------------------------------------
    # 6. Calibration 중 얼굴이 사라진 경우
    # --------------------------------------------------------

    else:

        if baseline_ear is None:

            calibration_start = None
            ear_samples.clear()


    # ========================================================
    # 디버그 UI
    # ========================================================

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    if average_ear is not None:

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    if baseline_ear is None:
        calibration_text = "Calibrating..."

    else:
        calibration_text = (
            f"Baseline: {baseline_ear:.3f}"
        )

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
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
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
        f"Closed Time: "
        f"{closed_duration:.2f}s",
        (30, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Last Close: "
        f"{last_closed_duration:.2f}s",
        (30, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
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


    # --------------------------------------------------------
    # 화면 출력
    # --------------------------------------------------------

    cv2.imshow(
        "Focus Camera",
        frame
    )


    # q 또는 ESC로 종료
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# ============================================================
# 프로그램 종료
# ============================================================

cap.release()
cv2.destroyAllWindows()