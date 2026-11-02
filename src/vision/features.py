import numpy as np

def extract_features(hand_landmarks):
    """
    Chuyển 21 landmark MediaPipe thành 63 đặc trưng float32.
    Chuẩn hóa bằng cách lấy landmark cổ tay (điểm 0) làm gốc,
    rồi chia cho độ dài vector từ điểm 0 đến điểm 9.
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks, got shape {points.shape}")

    points -= points[0]

    scale = np.linalg.norm(points[9])
    if scale > 1e-6:
        points /= scale

    return points.flatten().astype(np.float32)

