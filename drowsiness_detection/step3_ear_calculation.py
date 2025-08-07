import numpy as np

def euclidean_dist(p1, p2):
    """두 점 사이의 유클리드 거리 계산"""
    return np.linalg.norm(p1 - p2)

def calculate_ear(eye):
    """
    EAR 계산
    eye: (6, 2) 형태의 numpy 배열 (왼쪽/오른쪽 눈의 랜드마크)
    """
    # 수직 거리 두 개
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    # 수평 거리 하나
    C = euclidean_dist(eye[0], eye[3])

    # EAR 공식 적용
    ear = (A + B) / (2.0 * C)
    return ear
