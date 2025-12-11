import os
import shutil
import random

# 랜덤 시드 고정 (매번 같은 비율로 나누고 싶으면 유지)
random.seed(42)

# 현재 구조:
# Term project/data/<클래스 폴더들>
SRC_ROOT = "./data"

# YOLOv8용으로 만들 구조:
# Term project/datasets/food11/train|val|test/<클래스 폴더들>
DST_ROOT = "./datasets/food11"

# train/val/test 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # 나머지는 test

# 이미지 확장자들
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    # data 안의 폴더들 중에서 "폴더"만 클래스 폴더로 취급
    class_names = [
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d))
    ]

    print("[클래스 목록]")
    for name in class_names:
        print(" -", name)

    for cls in class_names:
        src_dir = os.path.join(SRC_ROOT, cls)
        # 해당 클래스의 모든 이미지 파일 찾기
        files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith(IMG_EXTS)
        ]

        if not files:
            print(f"[주의] {cls} 폴더에 이미지가 없습니다.")
            continue

        random.shuffle(files)
        n = len(files)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        print(f"\n[클래스: {cls}] 총 {n}장 -> "
              f"train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            dst_class_dir = os.path.join(DST_ROOT, split_name, cls)
            ensure_dir(dst_class_dir)

            for fname in split_files:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_class_dir, fname)
                shutil.copy2(src_path, dst_path)

    print("\n=== train/val/test 분할 완료! ===")
    print(f"결과 경로: {DST_ROOT}")


if __name__ == "__main__":
    main()
