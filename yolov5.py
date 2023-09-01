import torch


def parse_result(result):
    """the reuslt is a list, the length is batch size,for interface, we use batach=1. pred[0] is an 2D array, the length is target number, each target is represent as [x1, y1, x2, y2, confidence, class] """
    pred = result.pred
    pred = pred[0].cpu().numpy()

    return pred


def load_model():
    local_model_path = 'E:/work/codes/network/yolov5/'
    model_name = 'yolov5x'
    model = torch.hub.load(local_model_path, model_name, source='local')
    return model
