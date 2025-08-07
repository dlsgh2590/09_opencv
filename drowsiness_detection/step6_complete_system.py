import cv2
import dlib
import numpy as np

# EAR 계산 함수 임포트
from step3_ear_calculation import calculate_ear
# 졸음 판단 로직 클래스 임포트
from step4_drowsiness_logic import DrowsinessDetector
# 경고 시스템 클래스 임포트
from step5_alert_system import AlertSystem

# 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

# EAR 임계값 및 프레임 수 기준 설정 후 객체 생성
drowsiness_detector = DrowsinessDetector(ear_threshold=0.25, consec_frames=15)
alert_system = AlertSystem()

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 프레임을 못 받으면 종료

    # 프레임을 흑백으로 변환 (얼굴 검출은 흑백 영상에서 더 정확)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)

    for face in faces:
        # 얼굴 랜드마크 검출 (68개 점 추출)
        shape = predictor(gray, face)

        # (x, y) 형태의 numpy 배열로 변환
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        # 왼쪽, 오른쪽 눈 영역만 추출 (각각 6개 점)
        left_eye = shape_np[36:42]
        right_eye = shape_np[42:48]

        # 각각의 눈에 대해 EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # 두 눈의 EAR 평균 계산
        avg_ear = (left_ear + right_ear) / 2.0

        # 눈 하이라이트 (눈 주위 점에 원 그리기)
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # EAR 수치 프레임에 표시
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 졸음 판단 업데이트 (지속적 감김 상태인지 체크)
        is_drowsy, frame_count = drowsiness_detector.update(avg_ear)

        # 졸음 상태일 경우 경고 알림 수행
        if is_drowsy:
            frame = alert_system.alert(frame)  # 텍스트 + 비프음
        else:
            alert_system.reset()  # 경고 초기화

    # 결과 영상 출력
    cv2.imshow("Drowsiness Detection", frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 카메라 정리
cap.release()
cv2.destroyAllWindows()
