import cv2
import winsound  # 윈도우용 비프음, 리눅스/mac은 다른 라이브러리 필요

class AlertSystem:
    def __init__(self, beep_freq=1000, beep_duration=500):
        """
        알림 시스템 초기화
        beep_freq: 비프음 주파수 (Hz)
        beep_duration: 비프음 지속 시간 (ms)
        """
        self.beep_freq = beep_freq
        self.beep_duration = beep_duration
        self.alert_on = False

    def alert(self, frame):
        """
        졸음 알림 표시 및 소리 재생
        :param frame: 경고 메시지 출력할 프레임
        :return: 경고가 표시된 프레임
        """
        # 화면에 경고 문구 그리기
        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

        # 비프음 재생 (윈도우 전용)
        if not self.alert_on:
            winsound.Beep(self.beep_freq, self.beep_duration)
            self.alert_on = True

        return frame

    def reset(self):
        """경고 상태 초기화"""
        self.alert_on = False