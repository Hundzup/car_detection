from config import CLASSES, ROI, CONFIDENCE_THRESHOLD

def intersects(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (xB > xA) and (yB > yA)

def process_frame(model, frame):
    print(f"Обработка кадра размером {frame.shape}")
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            if model.names[cls_id] in CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if intersects((x1, y1, x2, y2), ROI):
                    return True
    return False