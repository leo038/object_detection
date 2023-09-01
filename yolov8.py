from ultralytics import YOLO


def load_model():
    model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)
    return model


def parse_result(result):
    """The return result is like [[x1, y1, x2, y2, confidence, class],...]"""
    result = result[0].boxes.boxes.cpu().numpy()
    return result
