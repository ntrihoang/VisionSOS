
import cv2
import time

from src.vision.hand_detector import HandDetector
from src.recognition.recognizer import GestureRecognizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 3
SEQUENCE_TIMEOUT = 5.0
COOLDOWN = 2.0

SOS_DISPLAY_SECONDS = 3.0


# ==========================================
# 2. DISPLAY CONFIGURATION
# ==========================================

WINDOW_NAME = "VisionSOS"

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def main():

    # --------------------------------------
    # Initialize camera and components
    # --------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # 2. Khởi tạo các module
    detector = HandDetector()

    recognizer = GestureRecognizer(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stable_frames=STABLE_FRAMES,
        sequence_timeout=SEQUENCE_TIMEOUT,
        cooldown=COOLDOWN
    )

    # Timer for SOS notification
    sos_display_until = 0.0

    print("=" * 45)
    print("VISION SOS STARTED")
    print("=" * 45)
    print("Press Q: Quit")
    print("Press R: Reset recognition")
    print("=" * 45)

    try:

        # ==================================
        # 4. MAIN LOOP
        # ==================================

        while True:

            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Cannot read frame from webcam.")
                break

            # Lật ảnh để thao tác tự nhiên như soi gương
            frame = cv2.flip(frame, 1)

            # ----------------------------------
            # Hand detection and feature extraction
            # ----------------------------------
            #3. Phát hiện bàn tay và trích xuất features
            features = detector.extract(frame, draw=True)

            if features is not None:

                # ----------------------------------
                #4. Dự đoán cử chỉ và xác minh chuỗi SOS
                # ----------------------------------

                result = recognizer.process(features)

                label = result["label"]
                confidence = result["confidence"]
                state = result["state"]
                stage = result["stage"]

                # ----------------------------------
                # Check SOS confirmation
                # ----------------------------------

                if result["confirmed"]:
                    sos_display_until = (
                        time.monotonic() + SOS_DISPLAY_SECONDS
                    )

                    print("[SOS] Gesture sequence confirmed!")

                # ----------------------------------
                # Hiển thị kết quả
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"Gesture: {label}",
                    (30, 40),
                    FONT,
                    0.7,
                    COLOR_GREEN,
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}",
                    (30, 75),
                    FONT,
                    0.7,
                    COLOR_WHITE,
                    2
                )

                cv2.putText(
                    frame,
                    f"State: {state}",
                    (30, 110),
                    FONT,
                    0.7,
                    COLOR_YELLOW,
                    2
                )

                # ----------------------------------
                # Display sequence progress
                # ----------------------------------

                progress_messages = {
                    0: "Waiting: OPEN_PALM",
                    1: "Step 1/3: Waiting THUMB_TUCKED",
                    2: "Step 2/3: Waiting THUMB_ENCLOSED",
                    3: "SOS CONFIRMED"
                }

                progress = progress_messages.get(
                    stage,
                    "Waiting..."
                )

                cv2.putText(
                    frame,
                    progress,
                    (30, 150),
                    FONT,
                    0.6,
                    COLOR_YELLOW,
                    2
                )

            else:

                # ----------------------------------
                # No hand detected
                # ----------------------------------

                cv2.putText(
                    frame,
                    "No hand detected",
                    (30, 40),
                    FONT,
                    0.7,
                    COLOR_RED,
                    2
                )

            # ==================================
            # 5. SOS NOTIFICATION DISPLAY
            # ==================================
            # IMPORTANT:
            # This block must stay OUTSIDE the
            # features if/else condition.

            if time.monotonic() < sos_display_until:

                cv2.putText(
                    frame,
                    "SOS CONFIRMED!",
                    (30, 210),
                    FONT,
                    1.2,
                    COLOR_RED,
                    3
                )

            # ----------------------------------
            # Show frame
            # ----------------------------------

            cv2.imshow(WINDOW_NAME, frame)

            # ==================================
            # 6. KEYBOARD CONTROLS
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                print("[INFO] Exiting VisionSOS...")
                break

            # Reset
            elif key == ord("r"):

                recognizer.reset()

                # Clear SOS notification timer
                sos_display_until = 0.0

                print("[INFO] Recognition reset.")

    finally:

        # ==================================
        # 7. CLEANUP
        # ==================================

        cap.release()
        detector.close()
        cv2.destroyAllWindows()

        print("[INFO] Resources released.")
        print("[INFO] VisionSOS stopped.")


# ==========================================
# 8. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
