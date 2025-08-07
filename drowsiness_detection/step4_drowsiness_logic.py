class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, consec_frames=15):
        """
        졸음 판단 클래스
        :param ear_threshold: EAR 임계값 (이보다 작으면 졸음 의심)
        :param consec_frames: 몇 프레임 연속으로 임계값보다 작아야 졸음 판단할지
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.counter = 0      # 연속 낮은 EAR 카운터
        self.drowsy = False   # 졸음 상태 여부

    def update(self, ear):
        """
        EAR 값을 받아 졸음 상태 업데이트
        :param ear: 현재 프레임의 EAR 값
        :return: (drowsy_flag: bool, frame_count: int)
        """
        if ear < self.ear_threshold:
            self.counter += 1
            if self.counter >= self.consec_frames:
                self.drowsy = True
        else:
            self.counter = 0
            self.drowsy = False

        return self.drowsy, self.counter