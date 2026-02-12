import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SRC_PATH = "./mall.mp4"
DST_PATH = "./mall-result2.mp4"

# 원본(4K) 기준 polygon
polygon_4k = np.array([
    [1310, 2142],
    [1906, 1270],
    [2374, 1258],
    [3494, 2150]
], dtype=np.int64)

# 입력 비디오 정보(4K)
src_info = sv.VideoInfo.from_video_path(SRC_PATH)
src_w, src_h = src_info.resolution_wh  # (3840, 2160)

# 출력 해상도(1080p)
out_w, out_h = 1920, 1080

# polygon 스케일링
sx = out_w / src_w
sy = out_h / src_h
polygon = polygon_4k.astype(np.float32)
polygon[:, 0] *= sx
polygon[:, 1] *= sy
polygon = polygon.astype(np.int64)

# PolygonZone (0.27.0 정확 시그니처)
zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator()   # ← 텍스트 전용
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=4)

model = YOLO("yolov8s.pt")

# OpenCV VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(DST_PATH, fourcc, src_info.fps, (out_w, out_h))
if not writer.isOpened():
    raise RuntimeError("VideoWriter open 실패")

frames = sv.get_video_frames_generator(source_path=SRC_PATH)

for idx, frame in enumerate(frames):
    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)

    # person만
    detections = detections[detections.class_id == 0]

    # zone 필터링
    print("before zone:", len(detections))
    zone_mask = zone.trigger(detections)
    detections = detections[zone_mask]
    print("after zone:", len(detections))

    # ✅ class + confidence 라벨
    labels = [
        f"{model.names[int(c)]} {conf:.2f}"
        for c, conf in zip(detections.class_id, detections.confidence)
    ]

    # 1️⃣ 박스
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    # 2️⃣ 텍스트 (class + confidence)
    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    # 3️⃣ zone
    frame = zone_annotator.annotate(scene=frame)

    writer.write(frame)

writer.release()
print("saved:", DST_PATH)
