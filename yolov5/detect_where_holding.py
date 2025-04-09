import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from norfair import Detection, Tracker, draw_tracked_objects
import os

def is_holding(person_box, obj_box):
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    center_x = (ox1 + ox2) / 2
    center_y = (oy1 + oy2) / 2
    return px1 <= center_x <= px2 and py1 + (py2 - py1) * 0.5 <= center_y <= py2

def convert_to_norfair_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([[int((x1 + x2) / 2), int((y1 + y2) / 2)]])

def detect_holding_with_tracking(weights, source, conf_thres=0.3, iou_thres=0.45, output_path="outputs/output_video.mp4"):
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(640, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    
    source_str = str(source).lower()
    is_video = source_str.endswith((".mp4", ".avi", ".mov"))
    is_image = source_str.endswith((".jpg", ".jpeg", ".png"))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = None
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)

    for i, (path, im, im0s, vid_cap, _) in enumerate(dataset):
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        holding_detections = []

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                persons, objects = [], []

                for *xyxy, conf, cls in det:
                    label = int(cls)
                    name = names[label]
                    if name == "person":
                        persons.append(xyxy)
                    elif name in ["cup", "bottle"]:
                        objects.append((xyxy, name))

                for p in persons:
                    found = False
                    for (o, obj_name) in objects:
                        if is_holding(p, o):
                            found = True
                            holding_detections.append(Detection(points=convert_to_norfair_bbox(p)))
                            cv2.rectangle(im0s, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 255, 0), 2)
                            cv2.putText(im0s, f"Holding {obj_name}", (int(p[0]), int(p[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if not found:
                        cv2.putText(im0s, "No person holding cup or bottle", (int(p[0]), int(p[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        tracked_objects = tracker.update(detections=holding_detections)
        draw_tracked_objects(im0s, tracked_objects)

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_video:
            if vid_writer is None:
                h, w = im0s.shape[:2]
                output_file = output_dir / Path(output_path).name
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                vid_writer = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))
            vid_writer.write(im0s)
        else:
            save_name = output_dir / f"image_output_{i}.jpg"
            cv2.imwrite(str(save_name), im0s)
            print(f"[INFO] Saved: {save_name}")

    if vid_writer:
        vid_writer.release()
        print(f"[INFO] Tracked video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov5m.pt")
    parser.add_argument("--source", default="video.mp4")  # Can be image, video, or folder
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", default="outputs/output_video.mp4")
    args = parser.parse_args()

    detect_holding_with_tracking(args.weights, args.source, args.conf, args.iou, args.output)
