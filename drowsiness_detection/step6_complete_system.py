import cv2
import numpy as np
import dlib
from step3_ear_calculation import calculate_ear
from step4_drowsiness_logic import DrowsinessDetector
from step5_alert_system import AlertSystem

# dlib 얼굴 랜드마크 모델 로드
predictor_path = "../shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 졸음 판단 및 알림 시스템 초기화
drowsiness_detector = DrowsinessDetector(ear_threshold=0.25, consec_frames=15)
alert_system = AlertSystem()

# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # 얼굴 랜드마크 검출
        shape = predictor(gray, face)
        shape_np = []
        for i in range(68):
            part = shape.part(i)
            shape_np.append((part.x, part.y))
        shape_np = np.array(shape_np)

        # 왼쪽, 오른쪽 눈 EAR 계산
        left_eye = shape_np[36:42]
        right_eye = shape_np[42:48]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # 졸음 판단
        is_drowsy, count = drowsiness_detector.update(avg_ear)

        # 눈 영역 하이라이트 (옵션)
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 졸음 알림 처리
        if is_drowsy:
            frame = alert_system.alert(frame)
        else:
            alert_system.reset()

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
