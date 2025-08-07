import cv2

# 얼굴 및 눈 검출용 Haar cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

print("[ESC] 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환 (검출용)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출

    for (x, y, w, h) in faces:
        # 얼굴 사각형 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 얼굴 영역 잘라서 눈 검출
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # 눈 바운딩 박스 (눈은 얼굴 내부 기준)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # 눈 중심점 좌표 계산
            cx = ex + ew // 2
            cy = ey + eh // 2

            # 눈 중심점 표시 (원)
            cv2.circle(roi_color, (cx, cy), 3, (0, 0, 255), -1)

    # 결과 출력
    cv2.imshow('Eye Detection', frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 정리
cap.release()
cv2.destroyAllWindows()