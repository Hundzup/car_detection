from ultralytics import YOLO


def load_model(model_path='yolov8n.pt'):
    print("Загрузка модели...")
    
    return YOLO(model_path)