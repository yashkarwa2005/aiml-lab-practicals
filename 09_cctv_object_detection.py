"""
AIML Lab Practical - Assignment 09
Aim: Implement Object Detection on CCTV Surveillance Footage / Video Feed
     using YOLO (You Only Look Once - YOLOv8) Architecture.
Author: Semester 5 AI & ML Lab
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def create_synthetic_surveillance_frame():
    """
    Generates a synthetic surveillance test frame with realistic street/pedestrian
    geometry if no external CCTV video/image is provided.
    """
    frame = np.full((480, 640, 3), (70, 75, 80), dtype=np.uint8)
    # Road / sidewalk
    cv2.rectangle(frame, (0, 240), (640, 480), (50, 50, 50), -1)
    cv2.line(frame, (0, 360), (640, 360), (255, 255, 255), 2)
    # Background building outlines
    cv2.rectangle(frame, (40, 60), (220, 240), (120, 110, 100), -1)
    cv2.rectangle(frame, (260, 40), (580, 240), (90, 100, 110), -1)
    # CCTV timestamp and camera ID overlay
    cv2.putText(frame, "CAM-01 [ENTRANCE-SURVEILLANCE]", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, "REC [LIVE 1080P/30FPS]", (440, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
    return frame

def run_cctv_detection(source=None, conf_threshold=0.25):
    print("=" * 65)
    print("      ASSIGNMENT 09: CCTV OBJECT DETECTION WITH YOLOV8")
    print("=" * 65)

    # 1. Load YOLOv8 Model (Pre-trained weights)
    print("[INFO] Loading YOLOv8 nano model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8 model initialized successfully.")

    # 2. Source handling: Video file, Webcam, or Test Frame
    is_video = False
    cap = None

    if source is not None and str(source).isdigit():
        print(f"[INFO] Opening Webcam device: {source}")
        cap = cv2.VideoCapture(int(source))
        is_video = True
    elif source is not None and os.path.exists(str(source)):
        ext = os.path.splitext(str(source))[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"[INFO] Opening CCTV Video file: {source}")
            cap = cv2.VideoCapture(str(source))
            is_video = True
        else:
            print(f"[INFO] Processing image file: {source}")
            frame = cv2.imread(str(source))
    else:
        # Default fallback test frame if no external file or camera is specified
        print("[INFO] No external video/image specified. Initializing test surveillance feed...")
        frame = create_synthetic_surveillance_frame()

    # 3. Object Detection Loop or Single Frame Processing
    if is_video and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[WARN] Unable to read from video source. Using synthetic surveillance frame.")
            frame = create_synthetic_surveillance_frame()

    # 4. Perform Inference
    print(f"[INFO] Running object detection (confidence threshold: {conf_threshold})...")
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    result = results[0]

    # 5. Parse Detections and Class Statistics
    boxes = result.boxes
    detected_counts = {}

    print("\n" + "-" * 65)
    print(f"{'Index':<6} | {'Class':<18} | {'Confidence':<12} | {'Bounding Box (xyxy)':<20}")
    print("-" * 65)

    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            confidence = float(box.conf[0].item())
            xyxy = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]

            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
            print(f"{idx+1:<6} | {class_name:<18} | {confidence:<12.2%} | {str(xyxy):<20}")
    else:
        print("No high-confidence objects detected in the current test frame.")

    print("-" * 65)
    print("Summary of Detected Entities in CCTV Feed:")
    if detected_counts:
        for cls_name, count in detected_counts.items():
            print(f"  - {cls_name.capitalize()}: {count}")
    else:
        print("  - Zero objects detected (try passing a live video or street scene image)")

    # 6. Render Detection Frame with Bounding Boxes
    annotated_frame = result.plot()  # Draws YOLO bounding boxes & labels

    # Overlay Surveillance HUD statistics
    hud_text = f"Objects Detected: {len(boxes) if boxes is not None else 0}"
    cv2.rectangle(annotated_frame, (10, 10), (280, 50), (0, 0, 0), -1)
    cv2.putText(annotated_frame, hud_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 7. Save and Display
    output_path = "output_assignment9.png"
    cv2.imwrite(output_path, annotated_frame)
    print(f"\n[SUCCESS] Annotated surveillance frame saved to '{output_path}'")

    # Display plot using matplotlib for compatibility across IDEs
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title("Assignment 09: CCTV Surveillance Object Detection (YOLOv8)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 CCTV Object Detection")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file (.mp4), image, or camera index (e.g., 0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    run_cctv_detection(source=args.source, conf_threshold=args.conf)

if __name__ == "__main__":
    main()
