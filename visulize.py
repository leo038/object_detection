import random
import cv2


def visulize(img, boxes, labels=None, colors=None, update_time=5, caption_name="Image", save_name=None):
    """

    :param img: image file
    :param boxes: list of box, like [[x1, y1, x2, y2], ...]
    :param labels: list of label
    :param colors: list of color
    :param update_time:
    :param caption_name:
    :param save_name: if provided, the image will be saved
    :return:
    """
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    thickness = 1

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if colors is not None:
            color = colors[index]
        else:
            color =[random.randint(0, 255) for _ in range(3)]
        center = int(x1) + 5, int(y1) + 10
        # plot box
        cv2.rectangle(img=image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), thickness=thickness, color=color,
                      lineType=cv2.LINE_AA)
        # plot label
        if labels is not None:
            label = labels[index]
            thickness_text = max(thickness - 1, 1)
            # cv2.rectangle(image, (int(x1), int(y1)), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img=image, text=label, org=center, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                        color=(255, 255, 255), thickness=thickness_text, lineType=cv2.LINE_AA)

    cv2.imshow(caption_name, image)
    cv2.waitKey(delay=update_time)

    # save image
    if save_name is not None:
        cv2.imwrite(save_name, image)
