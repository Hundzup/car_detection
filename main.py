import cv2
from model_utils import load_model
from detector import process_frame, intersects
from config import ROI, CLASSES, CONFIDENCE_THRESHOLD

def main(video_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        return
    
    # Цвета для визуализации
    roi_color = (0, 255, 0)  # Зеленый для ROI
    box_color = (255, 0, 0)  # Синий для объектов

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Рисуем ROI
        x1, y1, x2, y2 = ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

        results = model(frame)
        detected = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                confidence = box.conf.item()
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                if model.names[cls_id] in CLASSES:
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), box_color, 2)
                    if intersects((x1b, y1b, x2b, y2b), ROI):
                        detected = True
        
        # Вывод текста
        text = "traffic" if detected else "no traffic"
        cv2.putText(frame, 
            text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if detected:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Отображение кадра
        cv2.imshow('Traffic Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "cvtest.avi"
    main(video_path)