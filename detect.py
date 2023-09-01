import os
import time
import yaml
from yolov5 import load_model, parse_result
from visulize import visulize

with open("coco.yaml", 'r') as f:
    name_dict = yaml.safe_load(f)


def detect(image_file, display=True):
    model = load_model()
    if os.path.isfile(image_file):
        result = model(image_file)
        result = parse_result(result)
        if visulize:
            labels = ["{}:{:.2f}".format(name_dict[name], conf) for name, conf in zip(result[:, -1], result[:, -2])]
            visulize(img=image_file, boxes=result[:, :4], labels=labels)

    elif os.path.isdir(image_file):
        file_list = os.listdir(image_file)
        for frame_index, file_name in enumerate(file_list):
            img_file = os.path.join(image_file, file_name)
            time_start = time.time()
            result = model(img_file)
            time_end = time.time()
            print(f"Process {frame_index}/{len(file_list)}, FPS: {1 / (time_end - time_start)}, file name:{file_name} ")
            result = parse_result(result)
            if visulize:
                labels = ["{}:{:.2f}".format(name_dict[name], conf) for name, conf in zip(result[:, -1], result[:, -2])]
                visulize(img=img_file, boxes=result[:, :4], labels=labels)
            print(f"Process {frame_index}/{len(file_list)}, FPS: {1 / (time_end - time_start)}, file name:{file_name} ")


if __name__ == "__main__":
    img_dir = "E:\\dataset\\HT21\\train\\HT21-02\\img1\\"
    detect(img_dir)
