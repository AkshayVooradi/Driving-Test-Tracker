import os
from IPython import display
from ultralytics import YOLO
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections
from supervision.video.sink import VideoSink
import math
from tqdm import tqdm  # Use standard tqdm instead of notebook version
import numpy as np
import sys
import cv2
import shutil


def object_and_lane_detection(file_path):
    display.clear_output()
    SOURCE_VIDEO_PATH = file_path
    model1 = "yolov8x.pt"
    VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    model = YOLO(model1)
    model.fuse()
    CLASS_NAMES_DICT = model.model.names
    print(model.model.names)
    TARGET_VIDEO_PATH="processing/processed.mp4"
    l1 = []
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    max1 = sys.maxsize
    max2 = sys.maxsize
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):
            # print(model(frame))
            results = model(frame)[0]
            # print(results)
            detections = Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int),
            )
            new_detections = []
            for _, confidence, class_id, tracker_id in detections:
                if class_id == 2 or class_id == 5 or class_id == 7:
                    l1 = []
                    l1.append(_)
                    new_detections.append(l1)
                    break
            for i in new_detections:
                for j in i:
                    x1 = int(j[0])
                    y1 = int(j[1])
                    x3 = int(j[2])
                    y3 = int(j[3])
                    roi_vertices = [
                        ((int((x1 + x3) / 2) - 500), y3 + 150),  # Bottom-left
                        ((int((x1 + x3) / 2) - 500), y1),  # Top-left
                        (int((x1 + x3) / 2), y1),  # Top-right
                        (int((x1 + x3) / 2), (y3 + 150)),  # Bottom-right
                    ]
                    roi_vertices1 = [
                        ((int((x1 + x3) / 2) + 500), y3 + 150),  # Bottom-right
                        ((int((x1 + x3) / 2) + 500), y1),  # Top-right
                        (int((x1 + x3) / 2), y1),  # Top-left
                        (int((x1 + x3) / 2), (y3 + 150)),  # Bottom-left
                    ]
                    cv2.rectangle(frame, (x1, y1), (x3, y3), (0, 255, 0), 4)
                    mask = np.zeros_like(frame)
                    cv2.fillPoly(mask, [np.array(roi_vertices)], (255, 255, 255))
                    cv2.fillPoly(mask, [np.array(roi_vertices1)], (255, 255, 255))
                    roi_image = cv2.bitwise_and(frame, mask)
                    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                    edges = cv2.Canny(blurred, 100, 150)
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
                    if lines is not None:
                        for line in lines:
                            x, y, x2, y2 = line[0]
                            if ((x > (((x1 + x3) / 2) - 500) and x < x1 and y < (y3 + 150) and y > y1) and
                                (x2 > (((x1 + x3) / 2) - 500) and x2 < x1 and y2 < (y3 + 150) and y2 > y1)):
                                a1, b1 = (x + x2) / 2, (y + y2) / 2
                                a2, b2 = (x1 + x3) / 2, (y1 + y3) / 2
                                if math.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2) < 325:
                                    cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 5)
                                    sink.write_frame(frame)
                            if ((x < (((x1 + x3) / 2) + 500) and x > x3 and y < (y3 + 150) and y > y3) and
                                (x2 < (((x1 + x3) / 2) + 500) and x2 > x3 and y2 < (y3 + 150) and y2 > y1)):
                                a1, b1 = (x + x2) / 2, (y + y2) / 2
                                a2, b2 = (x1 + x3) / 2, (y1 + y3) / 2
                                if math.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2) < 325:
                                    cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 5)
                                    sink.write_frame(frame)

def process_video(file_path, processed_folder):
    processed_file_path = os.path.join(processed_folder, 'processed_' + os.path.basename(file_path))
    object_and_lane_detection(file_path)
    shutil.copy("processing/processed.mp4", processed_file_path)
    print(f"Processed file saved to: {processed_file_path}")
    return processed_file_path
