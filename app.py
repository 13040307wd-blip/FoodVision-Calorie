from ultralytics import YOLO
import cv2
import json
import os

MODEL_PATH = r"runs/classify/train/weights/best.pt"
CAL_DB_PATH = r"data/calorie_db.json"


def load_calorie_db():
    with open(CAL_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_one_image(model, calorie_db):
    img_path = input("\n[1] 분석할 이미지 경로 (예: test_images/my_food.jpg, 그냥 엔터=기본): ").strip()
    if not img_path:
        img_path = "test_images/my_food.jpg"

    if not os.path.exists(img_path):
        print("[!] 이미지 파일을 찾을 수 없습니다:", img_path)
        return

    try:
        portion = float(input("[2] 몇 인분으로 볼까요? (예: 0.5, 1, 2, 엔터=1): ").strip() or "1")
    except ValueError:
        portion = 1.0

    print("[INFO] 예측 중...")
    results = model(img_path)

    for r in results:
        top1_idx = r.probs.top1
        top1_conf = float(r.probs.top1conf)
        class_name = r.names[top1_idx]

        base_cal = calorie_db.get(class_name, 0)
        total_cal = base_cal * portion

        print("\n=== 예측 결과 ===")
        print(f"클래스: {class_name}")
        print(f"신뢰도: {top1_conf * 100:.2f}%")
        print(f"기본 1인분 칼로리: {base_cal} kcal")
        print(f"{portion} 인분 기준 예상 칼로리: {total_cal:.1f} kcal")

        # 이미지에 텍스트 표시
        img = r.orig_img.copy()
        text = f"{class_name} ({top1_conf*100:.1f}%), {total_cal:.0f} kcal"

        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("FoodVision-Calorie", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    print("=== FoodVision-Calorie ===")

    if not os.path.exists(MODEL_PATH):
        print("[!] 모델 파일을 찾을 수 없습니다:", MODEL_PATH)
        return

    if not os.path.exists(CAL_DB_PATH):
        print("[!] 칼로리 DB를 찾을 수 없습니다:", CAL_DB_PATH)
        return

    print("[INFO] 모델 로딩 중...")
    model = YOLO(MODEL_PATH)
    calorie_db = load_calorie_db()

    while True:
        print("\n메뉴")
        print("1. 음식 이미지 분석하기")
        print("2. 종료")
        choice = input("선택: ").strip()

        if choice == "1":
            predict_one_image(model, calorie_db)
        elif choice == "2":
            print("프로그램을 종료합니다.")
            break
        else:
            print("올바른 번호를 입력하세요.")


if __name__ == "__main__":
    main()
