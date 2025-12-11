# FoodVision-Calorie

가천대학교 오픈소스SW 텀 프로젝트  
**"음식 사진으로 음식 종류를 분류하고, 예상 칼로리를 계산해 주는 서비스"**

---

## 1. 프로젝트 소개

이 프로젝트는 음식 사진 한 장을 입력하면:

1. 딥러닝 분류 모델(YOLOv8 classify)을 이용해  
   사진 속 음식이 **Food-11 데이터셋의 11개 카테고리** 중 무엇인지 예측하고,
2. 예측된 음식 종류에 대해 **1인분 기준 칼로리**를 조회한 뒤,
3. 사용자가 입력한 **인분(0.5배, 1배, 2배 …)** 에 맞춰 칼로리를 다시 계산해 주는 프로그램입니다.

> Food-11: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit  
> (Kaggle Food-11 데이터셋 기반)

---

## 2. 주요 기능

- 이미지 파일 입력 (로컬 이미지 경로)
- YOLOv8 분류 모델을 이용한 음식 카테고리 예측
- 음식 카테고리별 기본 1인분 칼로리 제공 (`data/calorie_db.json`)
- 인분 수(배수)를 입력하면 칼로리 재계산
- OpenCV 창으로 **이미지 + 예측 결과 + 칼로리**를 시각적으로 출력
- GitHub를 통한 팀 협업 (여러 명이 같은 코드베이스 사용)

---

## 3. 폴더 구조

> **주의**: `data/`, `datasets/`, `runs/` 등은 용량 문제로 `.gitignore`에 포함되어 있어  
> GitHub에는 올라가지 않습니다. 각자 로컬에서 따로 준비해야 합니다.

~~~text
FoodVision-Calorie/
├─ app.py                 # 메뉴 기반 메인 실행 스크립트
├─ train_food11.py        # Food-11로 YOLO 분류 모델 학습
├─ predict_food11.py      # 단일 이미지 예측 + 칼로리 계산 예제
├─ scripts/
│   └─ split_food11.py    # 원본 Food-11 → train/val/test로 나누는 스크립트
├─ data/
│   └─ calorie_db.json    # 음식 카테고리별 기본 칼로리 정보 (Git에 포함됨)
├─ datasets/              # (로컬) YOLO 학습용 데이터셋 위치 (Git에는 안 올라감)
├─ runs/                  # (로컬) YOLO 학습 결과(모델, 로그) (Git에는 안 올라감)
└─ test_images/           # (로컬) 테스트용 이미지(선택)
~~~

---

## 4. 개발 환경

- Python 3.10
- Conda (Anaconda / Miniconda)
- 주요 라이브러리
  - [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8 분류 모델)
  - opencv-python
  - matplotlib

---

## 5. 설치 방법 (팀원용 빠른 시작)

### 5.1 레포지토리 클론

~~~bash
git clone https://github.com/사용자명/FoodVision-Calorie.git
cd FoodVision-Calorie
~~~

> 위 주소에서 `사용자명` 부분은 실제 GitHub 아이디로 변경하세요.

### 5.2 Conda 가상환경 생성

~~~bash
conda create -n food11 python=3.10
conda activate food11
~~~

### 5.3 패키지 설치

~~~bash
pip install ultralytics opencv-python matplotlib
~~~

---

## 6. 데이터셋 준비 (Food-11)

1. Kaggle에서 Food-11 데이터셋 다운로드  
   - 예: https://www.kaggle.com/datasets/vermaavi/food11

2. 압축을 풀고, 이미지 폴더들을 `data/` 아래에 복사  
   최종 형태 예시:

   ~~~text
   FoodVision-Calorie/
     data/
       Bread/
       Dairy product/
       Dessert/
       Egg/
       Fried food/
       Meat/
       Noodles-Pasta/
       Rice/
       Seafood/
       Soup/
       Vegetable-Fruit/
       calorie_db.json   # 이 파일은 레포지토리에 이미 포함되어 있음
   ~~~

3. `split_food11.py` 실행으로 train/val/test 분할

   ~~~bash
   conda activate food11
   cd FoodVision-Calorie

   python scripts/split_food11.py
   ~~~

   실행 후 폴더 구조:

   ~~~text
   datasets/
     food11/
       train/
         Bread/ ...
         ...
       val/
         ...
       test/
         ...
   ~~~

---

## 7. 모델 학습

~~~bash
conda activate food11
cd FoodVision-Calorie

python train_food11.py
~~~

- YOLOv8 분류용 사전학습 모델(`yolov8n-cls.pt`)을 불러와서  
  `datasets/food11`에 대해 파인튜닝합니다.
- 학습이 끝나면 모델 가중치는 보통 아래 경로에 저장됩니다.

~~~text
runs/classify/train/weights/best.pt
~~~

> **주의:** `runs/` 폴더와 `*.pt` 파일은 `.gitignore` 처리되어 있어  
> GitHub에 올라가지 않습니다. 각자 로컬에서 학습해야 합니다.

---

## 8. 실행 방법 (FoodVision-Calorie 앱)

### 8.1 테스트용 이미지 준비

예시:

~~~text
FoodVision-Calorie/
  test_images/
    my_food.jpg
~~~

### 8.2 메뉴 기반 앱 실행

~~~bash
conda activate food11
cd FoodVision-Calorie

python app.py
~~~

- **메뉴 1**: 이미지 경로와 인분 수를 입력해 분석  
  - 예: `test_images/my_food.jpg`, 인분: `1`  
- 예측 결과:
  - 예측된 **클래스 이름** (Dessert, Meat, …)
  - **신뢰도(%)**
  - **1인분 칼로리**, **N 인분 기준 칼로리**
  - OpenCV 창으로 이미지 + 텍스트 출력

---

## 9. 칼로리 DB 수정 방법

`data/calorie_db.json` 파일에는 각 Food-11 클래스의 기본 1인분 칼로리가 들어 있습니다.

~~~json
{
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
  "Vegetable-Fruit": 80
}
~~~

- 숫자는 예시이며, 보고서/발표용으로는 팀에서 조사한 실제 평균값으로 수정 가능합니다.
- **키 이름은 폴더/클래스 이름과 정확히 일치해야 합니다.**

---

## 10. 팀 협업 가이드 (간단)

- 각자 로컬에서:
  - `git clone` → `conda 환경 + 패키지 설치` → `데이터셋 준비` → `학습`
- 코드 수정 후:

  ~~~bash
  git status            # 변경 확인
  git add <파일>...
  git commit -m "메시지"
  git push
  ~~~

- 가능한 역할 분배 예:
  - A: 모델 학습 / 코드 구조 정리
  - B: 칼로리 DB 조사 및 정리
  - C: README·발표 자료 작성
  - D: 예제 이미지/테스트 케이스 정리

---

## 11. 참고 자료

- Food-11 데이터셋 (Kaggle): https://www.kaggle.com/datasets/vermaavi/food11  
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics  

