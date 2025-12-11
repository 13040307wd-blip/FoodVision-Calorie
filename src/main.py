# src/main.py
import argparse
from detector import FoodDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", default="result.jpg", help="output file name")
    args = parser.parse_args()

    detector = FoodDetector()
    out_path, total_cal = detector.detect_and_annotate(args.image, args.out)

    print(f"Detection completed!")
    print(f"Output saved at: {out_path}")
    print(f"Total calories: {total_cal}")

if __name__ == "__main__":
    main()
