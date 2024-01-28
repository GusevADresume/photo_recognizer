import cv2
import numpy
import os
from pathlib import Path
from memory_profiler import profile


# @profile()
def find_phone(img_path: str) -> bool:
    YOLO_CFG = "Resources/yolov4.cfg"
    YOLO_WEIGHTS = "Resources/yolov4.weights"

    cv = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    l_names = cv.getLayerNames()
    out_l_indexes = cv.getUnconnectedOutLayers()
    out_l = [l_names[index - 1] for index in out_l_indexes]

    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    search_object = ['cell phone']

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    cv.setInput(blob)
    outs = cv.forward(out_l)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = numpy.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        class_index = class_indexes[box_index]
        if classes[class_index] in search_object:
            objects_count += 1

    return objects_count > 0


def check_all_img(start_path: str) -> list[dict]:
    paths = os.listdir(os.path.join(start_path))
    result = []

    for path in paths:
        imgs_list = (os.listdir(os.path.join(start_path, path)))
        for img in imgs_list:
            if Path(os.path.join(start_path, path, img)).suffix == '.jpg':
                result.append({img: find_phone(os.path.join(start_path, path, img))})
    return result


if __name__ == "__main__":
    img_name = input('Input path to image and name: ')
    print(find_phone(img_name))
