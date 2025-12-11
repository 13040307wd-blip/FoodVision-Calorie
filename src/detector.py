# src/detector.py
from ultralytics import YOLO
import cv2
import json
import os

class FoodDetector:
    def __init__(self, model_path="yolov8s.pt", calorie_db_path="../data/calorie_db.json"):
        self.model = YOLO(model_path)

        # 칼로리 DB 로드
        db_full_path = os.path.join(os.path.dirname(__file__), calorie_db_path)
        with open(db_full_path, "r", encoding="utf-8") as f:
            self.calorie_db = json.load(f)

    def detect_and_annotate(self, img_path, out_path="result.jpg"):
        img = cv2.imread(img_path)
        results = self.model(img)[0]

        total_cal = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            # 칼로리 불러오기
            cal = self.calorie_db.get(label, None)
            if isinstance(cal, (int, float)):
                total_cal += cal

            # bounding box 표시
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} ({cal if cal else 'N/A'} kcal)"
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 총 칼로리 표시
        cv2.putText(img, f"Total: {total_cal} kcal", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(out_path, img)
        return out_path, total_cal
