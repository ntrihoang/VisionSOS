
from collections import deque


class TemporalVerifier:
    """
    Xác minh chuỗi cử chỉ SOS theo thời gian.

    Chuỗi SOS:
        OPEN_PALM -> THUMB_TUCKED -> THUMB_ENCLOSED

    Chức năng:
        - Kiểm tra confidence.
        - Kiểm tra độ ổn định của cử chỉ.
        - Xác minh đúng thứ tự chuỗi SOS.
        - Xử lý timeout.
        - Chống kích hoạt SOS liên tục bằng cooldown.
    """

    def __init__(
        self,
        confidence_threshold=0.70,
        stable_frames=3,
        sequence_timeout=5.0,
        cooldown=2.0
    ):
        # Các tham số cấu hình
        self.confidence_threshold = confidence_threshold # Ngưỡng confidence để chấp nhận cử chỉ
        self.stable_frames = stable_frames # Quy định số frame cần ổn định để xác minh cử chỉ đúng
        self.sequence_timeout = sequence_timeout # thời gian tối đa để hoàn thành chuỗi SOS, nếu quá sẽ reset
        self.cooldown = cooldown # thời gian hồi SOS

        # Chuỗi cử chỉ SOS cần xác minh
        self.sequence = [
            "OPEN_PALM",
            "THUMB_TUCKED",
            "THUMB_ENCLOSED"
        ]
        # theo dõi xem người dùng đã đi qua được mấy bước trong chuỗi.
        self.stage = 0
        
        # Lưu các nhãn gần nhất để kiểm tra độ ổn định
        self.recent_labels = deque(maxlen=stable_frames)

        # Nhãn ổn định đã được xử lý gần nhất
        self.last_processed_label = None

        # Thời điểm bắt đầu chuỗi SOS
        self.sequence_start_time = None

        # Thời điểm xác nhận SOS gần nhất
        self.last_confirmation_time = None

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):
        """Reset tiến trình FSM, giữ trạng thái lịch sử nhận diện."""
        self.stage = 0
        self.sequence_start_time = None

    def reset_all(self):
        """Reset toàn bộ trạng thái của TemporalVerifier."""
        self.reset()
        self.recent_labels.clear()
        self.last_processed_label = None
        self.last_confirmation_time = None

    # ==================================================
    # HELPER: TẠO KẾT QUẢ
    # ==================================================

    def _result(self, state, confirmed=False):
        """Chuẩn hóa kết quả trả về."""
        return {
            "confirmed": confirmed,
            "state": state,
            "stage": self.stage
        }

    # ==================================================
    # MAIN UPDATE
    # ==================================================

    def update(self, label, timestamp, confidence):
        """
        Nhận dự đoán từ ML Predictor và xác minh chuỗi SOS.

        Args:
            label (str): Nhãn cử chỉ được dự đoán.
            timestamp (float): Thời điểm xử lý frame.
            confidence (float): Độ tin cậy của model.

        Returns:
            dict:
                confirmed: SOS đã được xác nhận hay chưa.
                state: Trạng thái xử lý.
                stage: Số bước SOS đã hoàn thành.
        """

        # 1. KIỂM TRA COOLDOWN - 
        if self.last_confirmation_time is not None:

            elapsed = timestamp - self.last_confirmation_time

            if elapsed < self.cooldown:
                return self._result("COOLDOWN")

            # Hết cooldown: chuẩn bị cho chuỗi SOS mới
            self.reset_all()

        # 2. KIỂM TRA SEQUENCE TIMEOUT - thời
        if self.sequence_start_time is not None:

            elapsed = timestamp - self.sequence_start_time

            if elapsed >= self.sequence_timeout:
                self.reset()
                self.recent_labels.clear()
                self.last_processed_label = None

                return self._result("TIMEOUT")

        # 3. KIỂM TRA CONFIDENCE
        if confidence < self.confidence_threshold:
            self.recent_labels.clear()
            self.last_processed_label = None

            return self._result("LOW_CONFIDENCE")

        # 4. THU THẬP NHÃN VÀ KIỂM TRA ĐỘ ỔN ĐỊNH
        self.recent_labels.append(label)

        # Chưa đủ số frame yêu cầu
        if len(self.recent_labels) < self.stable_frames:
            return self._result("COLLECTING")

        # Các frame phải có cùng một nhãn
        if len(set(self.recent_labels)) != 1:
            self.last_processed_label = None
            return self._result("UNSTABLE")

        # Lấy nhãn đã ổn định
        stable_label = self.recent_labels[-1]

        # 5. CHỐNG XỬ LÝ TRÙNG LẶP
        if stable_label == self.last_processed_label:
            return self._result("WAITING")

        self.last_processed_label = stable_label

        # 6. XỬ LÝ OTHER
        if stable_label == "OTHER":
            self.reset()
            self.recent_labels.clear()
            self.last_processed_label = None

            return self._result("RESET")

        # 7. KIỂM TRA NHÃN TIẾP THEO
        expected_label = self.sequence[self.stage]

        # 8. XỬ LÝ CỬ CHỈ ĐÚNG THỨ TỰ
        if stable_label == expected_label:

            # Bắt đầu tính thời gian từ cử chỉ đầu tiên
            if self.stage == 0:
                self.sequence_start_time = timestamp

            self.stage += 1

            # 9. KIỂM TRA HOÀN THÀNH SOS
            if self.stage == len(self.sequence):

                completed_stage = self.stage

                self.last_confirmation_time = timestamp

                # Reset tiến trình nhưng giữ cooldown
                self.reset()
                self.recent_labels.clear()
                self.last_processed_label = None

                return {
                    "confirmed": True,
                    "state": "SOS_CONFIRMED",
                    "stage": completed_stage
                }

            return self._result("IN_PROGRESS")

        # 10. XỬ LÝ SAI THỨ TỰ

        # Nếu gặp OPEN_PALM, bắt đầu lại chuỗi từ đầu
        if stable_label == self.sequence[0]:

            self.stage = 1
            self.sequence_start_time = timestamp

            return self._result("IN_PROGRESS")

        # Các cử chỉ sai thứ tự còn lại sẽ reset chuỗi
        self.reset()
        self.recent_labels.clear()
        self.last_processed_label = None

        return self._result("RESET")
