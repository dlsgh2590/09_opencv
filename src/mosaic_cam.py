import cv2

# 모자이크 강도 설정 (숫자가 클수록 더 강한 모자이크 효과)
rate = 15

win_title = '모자이크 캠'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠(기본 카메라) 열기
cap = cv2.VideoCapture(0)

# 수동 모자이크 기능을 위한 변수 초기화
manual_mode = False         # 수동 모드 활성화 여부
manual_coords = None        # 수동으로 선택한 ROI 좌표 저장 변수

# 사용법 안내 출력
print("[ESC] 종료 | [m] 수동 모자이크 모드 | [r] 수동 모드 영역 초기화")

# 메인 루프: 카메라가 열려있는 동안 반복
while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 프레임을 못 읽으면 루프 종료

    # 원본 프레임을 복사해서 모자이크 효과 적용할 버전 생성
    display_frame = frame.copy()

    # 1. 자동 얼굴 인식 및 모자이크 처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 얼굴 검출을 위한 흑백 변환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출 수행

    # 검출된 얼굴들 각각에 대해 모자이크 처리
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]  # 얼굴 영역 잘라내기
        small = cv2.resize(roi, (w // rate, h // rate))  # 축소
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # 확대 (모자이크)
        display_frame[y:y+h, x:x+w] = mosaic  # 모자이크 영역 원본 프레임에 덮어쓰기

    # 2. 수동 선택한 ROI 모자이크 처리 (m 키로 지정한 영역)
    if manual_mode and manual_coords is not None:
        x, y, w, h = manual_coords  # ROI 좌표 가져오기
        roi = frame[y:y+h, x:x+w]  # 선택한 영역 잘라내기
        small = cv2.resize(roi, (w // rate, h // rate))  # 축소
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # 확대
        display_frame[y:y+h, x:x+w] = mosaic  # 원본에 모자이크 덮기

        # 선택 영역 시각화 (노란색 사각형)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 3. 결과 프레임 화면에 출력
    cv2.imshow(win_title, display_frame)

    # 4. 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키 → 종료
        break
    elif key == ord('m'):  # m 키 → 수동 모자이크 영역 선택 모드
        print("ROI 영역을 선택하세요 (Enter/Space로 확정, Esc로 취소)")
        manual_coords = cv2.selectROI(win_title, frame, False)  # ROI 선택 UI 띄우기
        if manual_coords[2] == 0 or manual_coords[3] == 0:
            # 선택한 영역의 넓이나 높이가 0이면 → 선택 안 한 것 → 무시
            manual_coords = None
        else:
            # 정상 선택된 경우 수동 모드 활성화
            manual_mode = True
    elif key == ord('r'):  # r 키 → 수동 영역 초기화
        manual_coords = None
        manual_mode = False

# 모든 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()