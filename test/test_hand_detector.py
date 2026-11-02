
import cv2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.vision.hand_detector import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    if not cap.isOpened():
        print("Cannot open webcam!")
        detector.close()
        return

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            features = detector.extract(frame)

            if features is not None:
                print("Features shape:", features.shape)

                cv2.putText(
                    frame,
                    "Hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("VisionSOS - Hand Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
