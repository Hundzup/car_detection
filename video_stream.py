# video_stream.py
import cv2
from model_utils import load_model
from detector import process_frame, intersects
from config import ROI, CLASSES, CONFIDENCE_THRESHOLD

class VideoCamera:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.model = load_model()
        self.roi = ROI
        self.classes = CLASSES
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        x1, y1, x2, y2 = self.roi
        detected = False

        results = self.model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                confidence = box.conf.item()
                if confidence < self.confidence_threshold:
                    continue
                if self.model.names[cls_id] in self.classes:
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (255, 0, 0), 2)
                    if intersects((x1b, y1b, x2b, y2b), self.roi):
                        detected = True

        # Рисуем ROI и статус
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        status_text = "Traffic" if detected else "No Traffic"
        color = (0, 0, 255) if detected else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if detected:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), status_text