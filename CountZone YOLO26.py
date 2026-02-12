import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SRC_PATH = "./mall.mp4"
DST_PATH = "./mall-result-YOLO26.mp4"

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

# 4K polygon -> 1080p polygon 스케일링
sx = out_w / src_w
sy = out_h / src_h
polygon = polygon_4k.astype(np.float32)
polygon[:, 0] *= sx
polygon[:, 1] *= sy
polygon = polygon.astype(np.int64)

# ✅ supervision 0.27.0: PolygonZone는 해상도 인자를 받지 않음
zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

box_annotator = sv.BoxAnnotator(thickness=3)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=4)

model = YOLO("yolo26s.pt")

# OpenCV VideoWriter로 저장(supervision writer 우회)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(DST_PATH, fourcc, src_info.fps, (out_w, out_h))
if not writer.isOpened():
    raise RuntimeError("VideoWriter open 실패: mp4 코덱/FFmpeg/OpenCV 빌드 문제 가능")

frames = sv.get_video_frames_generator(source_path=SRC_PATH)

for idx, frame in enumerate(frames):
    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # person만

    # ✅ 0.27.0 올바른 호출: trigger(detections)
    zone_mask = zone.trigger(detections)

    # (선택) zone 안에 들어온 detection만 남기기
    detections_in_zone = detections[zone_mask]

    labels = [
        f"{model.names[int(c)]} {float(conf):.2f}"
        for c, conf in zip(detections_in_zone.class_id, detections_in_zone.confidence)
    ]

    # annotate (0.27.0 기준 positional이 안전)
    try:
        frame = box_annotator.annotate(frame, detections_in_zone, labels=labels)
    except TypeError:
        frame = box_annotator.annotate(frame, detections_in_zone)

    frame = zone_annotator.annotate(frame)

    writer.write(frame)

writer.release()
print("saved:", DST_PATH)
