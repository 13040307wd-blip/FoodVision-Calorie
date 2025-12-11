from ultralytics import YOLO
import cv2
import json
import os

MODEL_PATH = r"runs/classify/train/weights/best.pt"
CAL_DB_PATH = r"data/calorie_db.json"


def load_calorie_db():
    # calorie_db.json이 비어 있거나 없으면 기본 값 사용
    default_db = {
        "Bread": 250,
        "Dairy product": 150,
        "Dessert": 400,
        "Egg": 80,
        "Fried food": 500,
        "Meat": 300,
        "Noodles-Pasta": 350,
        "Rice": 300,
        "Seafood": 250,
        "Soup": 150,
        "Vegetable-Fruit": 80,
    }

    if not os.path.exists(CAL_DB_PATH):
        return default_db

    try:
        with open(CAL_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or not data:
                return default_db
            return data
    except Exception as e:
        print("[경고] calorie_db.json 읽기 오류:", e)
        return default_db


def main():
    # 1) 모델 & 칼로리 DB 로드
    print("[INFO] 모델 로딩 중...")
    model = YOLO(MODEL_PATH)

    calorie_db = load_calorie_db()

    # 2) 이미지 경로 입력
    img_path = input("분석할 이미지 경로를 입력하세요 (예: test_images/my_food.jpg): ").strip()
    if not img_path:
        img_path = "test_images/my_food.jpg"

    if not os.path.exists(img_path):
        print("이미지 파일을 찾을 수 없습니다:", img_path)
        return

    # 3) 1인분 배수 입력 (0.5, 1, 2 등)
    try:
        portion = float(input("몇 인분으로 볼까요? (예: 0.5, 1, 2): ").strip() or "1")
    except ValueError:
        portion = 1.0

    # 4) 예측
    print("[INFO] 예측 중...")
    results = model(img_path)

    # 5) 결과 해석
    for r in results:
        top1_idx = r.probs.top1
        top1_conf = float(r.probs.top1conf)
        class_name = r.names[top1_idx]

        # 칼로리 계산
        base_cal = calorie_db.get(class_name, 0)
        total_cal = base_cal * portion

        print(f"\n=== 예측 결과 ===")
        print(f"클래스: {class_name}")
        print(f"신뢰도: {top1_conf * 100:.2f}%")
        print(f"기본 1인분 칼로리: {base_cal} kcal")
        print(f"{portion} 인분 기준 예상 칼로리: {total_cal:.1f} kcal")

        # 6) 이미지에 텍스트 그려서 보여주기
        img = r.orig_img.copy()
        text = f"{class_name} ({top1_conf*100:.1f}%), {total_cal:.0f} kcal"

        cv2.putText(
            img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("FoodVision-Calorie", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
