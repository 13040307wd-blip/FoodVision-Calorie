from ultralytics import YOLO

def main():
    # 1) 분류용 사전학습 모델 불러오기
    #    설치된 ultralytics 버전에 따라 둘 중 하나를 써봐:
    #    - "yolov8n-cls.pt"
    #    - "yolo11n-cls.pt"
    model = YOLO("yolov8n-cls.pt")  # 안 되면 "yolo11n-cls.pt"로 바꿔보기

    # 2) 학습 실행
    model.train(
        data="datasets/food11",  # 방금 만든 train/val/test 구조 루트
        epochs=20,               # 처음에는 20 정도로 테스트
        imgsz=224,               # 분류 기본 이미지 크기
        batch=32,                # 메모리 부족하면 16, 8로 줄이기
    )

    print("학습 끝!")


if __name__ == "__main__":
    main()
