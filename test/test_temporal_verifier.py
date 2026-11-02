import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recognition.temporal_verifier import TemporalVerifier


def send_gesture(verifier, label, start_time, confidence=0.95):
    """
    Mô phỏng một cử chỉ được giữ ổn định trong 3 frame.
    """
    results = []

    for i in range(3):
        timestamp = start_time + i * 0.1

        result = verifier.update(
            label=label,
            timestamp=timestamp,
            confidence=confidence
        )

        results.append(result)

    return results[-1]


def test_correct_sequence():
    print("\nTEST 1: CORRECT SOS SEQUENCE")

    verifier = TemporalVerifier()

    result1 = send_gesture(verifier, "OPEN_PALM", 0.0)
    result2 = send_gesture(verifier, "THUMB_TUCKED", 1.0)
    result3 = send_gesture(verifier, "THUMB_ENCLOSED", 2.0)

    print("OPEN_PALM:", result1)
    print("THUMB_TUCKED:", result2)
    print("THUMB_ENCLOSED:", result3)

    assert result1["state"] == "IN_PROGRESS"
    assert result2["state"] == "IN_PROGRESS"
    assert result3["state"] == "SOS_CONFIRMED"
    assert result3["confirmed"] is True

    print("PASS")


def test_wrong_sequence():
    print("\nTEST 2: WRONG SEQUENCE")

    verifier = TemporalVerifier()

    result = send_gesture(verifier, "THUMB_TUCKED", 0.0)

    print("Result:", result)

    assert result["state"] == "RESET"
    assert result["confirmed"] is False

    print("PASS")


def test_low_confidence():
    print("\nTEST 3: LOW CONFIDENCE")

    verifier = TemporalVerifier()

    result = verifier.update(
        label="OPEN_PALM",
        timestamp=0.0,
        confidence=0.50
    )

    print("Result:", result)

    assert result["state"] == "LOW_CONFIDENCE"

    print("PASS")


def test_timeout():
    print("\nTEST 4: SEQUENCE TIMEOUT")

    verifier = TemporalVerifier(sequence_timeout=5.0)

    send_gesture(verifier, "OPEN_PALM", 0.0)

    # Đã quá 5 giây kể từ khi bắt đầu chuỗi
    result = verifier.update(
        label="THUMB_TUCKED",
        timestamp=5.5,
        confidence=0.95
    )

    print("Result:", result)

    assert result["state"] == "TIMEOUT"
    assert result["stage"] == 0

    print("PASS")


def test_cooldown():
    print("\nTEST 5: COOLDOWN")

    verifier = TemporalVerifier(cooldown=2.0)

    send_gesture(verifier, "OPEN_PALM", 0.0)
    send_gesture(verifier, "THUMB_TUCKED", 1.0)

    result = send_gesture(verifier, "THUMB_ENCLOSED", 2.0)

    assert result["confirmed"] is True

    # Thử gửi thêm cử chỉ ngay trong cooldown
    cooldown_result = verifier.update(
        label="OPEN_PALM",
        timestamp=4.0,
        confidence=0.95
    )

    print("Result:", cooldown_result)

    assert cooldown_result["state"] == "COOLDOWN"

    print("PASS")


if __name__ == "__main__":
    test_correct_sequence()
    test_wrong_sequence()
    test_low_confidence()
    test_timeout()
    test_cooldown()

    print("\nALL TESTS PASSED!")
