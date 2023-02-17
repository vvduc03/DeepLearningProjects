# import necessary package
# import necessary package
import cv2.data
import numpy as np
import cv2
import argparse
import imutils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from utils import load_checkpoint, cells_to_bboxes, non_max_suppression, plot_image, plot_couple_examples
from model import YOLOV3_Byhand

import torch
from torch import optim
import cv2.data
import numpy as np
import cv2
import argparse
import imutils

import config
from utils import load_checkpoint
from model import  YOLOV3_Byhand

import torch
from torch import optim

# create color space
color = [(255, 0, 0),
         (0, 255, 0),
         (30, 30, 30),
         (194, 194, 194),
         (13, 29, 152),
         (242, 20, 188),
         (18, 115, 222),
         (255, 162, 0),
         (9, 167, 121),
         (54, 1, 255),
         (10, 255, 255),
         (255, 255, 0),
         (255, 0, 255),
         (0, 255, 255),
         (150, 150, 150),
         (200, 150, 50),
         (50, 150, 200),
         (150, 20, 200),
         (70, 175, 20),
         (84, 150, 150),
         ]
IMAGE_SIZE, height, width= 416, 416, 416
# load model
model = YOLOV3_Byhand(num_classes=config.NUM_CLASSES).to('cuda')
model = load_checkpoint("checkpoint.pth", model, optimizer=optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY))

transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    # bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to('cuda')
# detection smile
video = cv2.VideoCapture(0)
while True:

    grabbed, frame = video.read()

    frame = imutils.resize(frame, width=416)

    frame_clone = frame.copy()

    """Plots predicted bounding boxes on the image"""
    class_labels = config.PASCAL_CLASSES

    frame_trans = transforms(image=frame)

    frame_trans = torch.unsqueeze(frame_trans['image'], dim=0).to('cuda')
    # frame_trans = torch.unsqueeze(frame, dim=0)

    model.eval()

    # scaled_anchors = (
    #         torch.tensor(config.ANCHORS)
    #         * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    # # ).to(config.DEVICE)

    with torch.no_grad():
        out = model(frame_trans)
        bboxes = [[] for _ in range(frame_trans.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

    frame_trans = torch.squeeze(frame_trans)

    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=0.8, threshold=0.8, box_format="midpoint",
    )

    # frame_trans = frame_trans.permute(1, 2, 0).detach()
    for box in nms_boxes:

        class_pred = box[0]
        acc = int(box[2] * 100)
        box = box[2:]

        x1 = int((box[0] - box[2] / 2) * 416)
        y1 = int((box[1] - box[3] / 2) * 416)
        x2 = x1 + int(box[2] * 416)
        y2 = y1 + int(box[3] * 416)

        cv2.rectangle(frame_clone,
                      (x2, y2),
                      (x1, y1),
                      color=color[int(class_pred)],
                      thickness=2
                      )

        cv2.putText(frame_clone,
                    config.PASCAL_CLASSES[int(class_pred)],
                    (x1, y1),
                    # (1, 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color=color[int(class_pred)],
                    thickness=2)

        cv2.putText(frame_clone,
                    config.PASCAL_CLASSES[int(class_pred)],
                    # (x1, y1),
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color=(0, 0, 0),
                    thickness=2)
        cv2.putText(frame_clone,
                    f"Acc: {acc}",
                    # (x1, y1),
                    (30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color=(0, 0, 0),
                    thickness=2)
    cv2.imshow('img', frame_clone)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()