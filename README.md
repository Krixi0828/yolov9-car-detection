# YOLOv9 Car Detection

## Overview

這是一個使用 **YOLOv9** 進行單一類別車輛偵測（Object Detection）的練習專案，目標是從零完成一套完整的 custom object detection pipeline，並透過四次實驗逐步找出模型問題、提出改善方法，再用量化結果驗證改善是否有效。

本專案的偵測類別只有：

```text
0: car
```

整體流程如下：

```text
資料蒐集與標註
        ↓
建立 YOLO dataset
        ↓
Experiment 1：From Scratch Baseline
        ↓
Experiment 2：Pretrained Fine-tuning
        ↓
Experiment 3：增加 Training Data
        ↓
Independent / Diagnostic Evaluation
        ↓
Error Analysis
        ↓
Experiment 4：提高 Input Resolution
        ↓
Final Model & External Inference
```

四次實驗的研究研究問題分別是：

| Experiment | 研究問題 |
|---|---|
| Experiment 1 | 少量資料、random initialization 能不能學會 car detection？ |
| Experiment 2 | 使用 pretrained weights 是否能改善 small-data training？ |
| Experiment 3 | 增加 traffic-scene training data 是否能提升 generalization？ |
| Experiment 4 | 提高 input resolution 是否能改善 small / distant car detection？ |

---

## Environment

本專案主要在 Apple Silicon MacBook Air 上進行訓練、evaluation 與 inference，使用 Conda 建立獨立 Python environment。

| Item | Version / Setting |
|---|---|
| OS | macOS |
| Hardware | Apple Silicon MacBook Air |
| Python | 3.10.21 |
| Conda Environment | `yolov9` |
| PyTorch | 2.14.0 |
| Compute Device | Apple MPS |
| YOLOv9 | WongKinYiu/yolov9 |
| Models | YOLOv9-T / YOLOv9-S |

由於執行環境為 Apple Silicon，因此本專案不使用 CUDA，而是透過 PyTorch MPS backend 進行運算：

```text
--device mps
```

完整 Python dependencies 與相容版本統一記錄於：

```text
requirements.txt
```

---

## Setup

### 1. 建立 Python Environment

建議使用 Python 3.10 建立獨立 Conda environment：

```bash
conda create -n yolov9 python=3.10
conda activate yolov9
```

接著安裝本專案需要的 Python dependencies：

```bash
pip install -r requirements.txt
```

---

### 2. Clone YOLOv9

本 repository 不重新包含完整 YOLOv9 source code，請另外 clone 官方 repository：

```bash
git clone https://github.com/WongKinYiu/yolov9.git
```

完成後，專案根目錄應包含：

```text
YOLOCOCO/
├── yolov9/
├── data/
├── results/
├── README.md
├── requirements.txt
└── ...
```

---

### 3. PyTorch Compatibility

本專案使用的 PyTorch environment 與原始 YOLOv9 repository 在 checkpoint loading 上存在相容性差異。

對可信任的官方 pretrained weights 與本地自行訓練的 checkpoints，需要將相關：

```python
torch.load(...)
```

調整為：

```python
torch.load(..., weights_only=False)
```

本專案實際修改的位置包含：

```text
yolov9/utils/general.py
yolov9/models/experimental.py
yolov9/train_dual.py
```

> `weights_only=False` 僅應使用於可信任來源的 checkpoint。

---

### 4. 準備 Dataset

完整 datasets 不直接包含在本 GitHub repository。

若要重現本專案實驗，需要先準備原始 custom dataset，並依照 [Data Sources](#data-sources) 下載官方 UA-DETRAC dataset。

建議資料結構：

```text
data/
├── car_dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
└── UA-DETRAC_raw/
    ├── DETRAC-Images.zip
    ├── DETRAC-Train-Annotations-XML/
    └── DETRAC-Test-Annotations-XML/
```

其中 `car_dataset/` 為原始 custom dataset：

```text
23 training images
6 validation images
```

UA-DETRAC 則用於 Experiment 3、Experiment 4 與 diagnostic benchmark。

---

### 5. 建立 Experiment Dataset

建立 Experiment 3 / 4 使用的 expanded training dataset：

```bash
python prepare_detrac_exp3.py
```

完成後可使用：

```bash
python check_exp3_labels.py
```

確認 images 與 YOLO annotations 是否正確。

建立 diagnostic benchmark：

```bash
python prepare_detrac_test.py
```

並使用：

```bash
python check_test_labels.py
```

檢查 test dataset。

完成 dataset preparation 後，即可依照各 Experiment section 中記錄的 training command 重現實驗。

---

## Dataset

### 1. Original Custom Dataset

最初自行蒐集並使用 LabelImg 手動標註 29 張圖片：

| Split | Images |
|---|---:|
| Train | 23 |
| Validation | 6 |
| Total | 29 |

Validation set 中共有 18 個 car instances。

Bounding box 使用 YOLO format：

```text
<class_id> <x_center> <y_center> <width> <height>
```

例如：

```text
0 0.512 0.483 0.428 0.301
```

其中 `0` 代表 `car`。

### 2. Expanded Training Dataset

Experiment 3 開始加入 UA-DETRAC traffic scenes：

```text
Original custom training data：23 images
UA-DETRAC subset：500 images
```

最後 training dataset 為：

```text
523 training images
3242 car bounding boxes
```

UA-DETRAC 原始 annotation 為 XML format，因此先將：

```text
vehicle_type="car"
```

轉換成 YOLO annotation：

```text
0 x_center y_center width height
```

500 張 UA-DETRAC images 由不同 video sequences 分散抽樣，避免大量選到高度相似的相鄰 frames。

Validation set 仍維持原本：

```text
6 images
18 car instances
```

### 3. Diagnostic Benchmark

因為原本 validation set 只有 6 張圖片，樣本太少，Experiment 3 後另外從 UA-DETRAC 官方 Test split 建立較大的 evaluation benchmark：

```text
200 images
1844 car instances
40 unseen UA-DETRAC Test sequences
```

這組 benchmark 最初用於測試 Experiment 3，但後續 error analysis 的結果也被用來設計 Experiment 4，因此後續更適合稱為：

```text
Diagnostic Benchmark
```

而不是完全 untouched 的 final test set。

---

## Data Sources

本專案主要使用 YOLOv9 官方 repository、YOLOv9 pretrained weights，以及 UA-DETRAC traffic dataset。

### YOLOv9

模型程式碼基於官方 YOLOv9 repository：

- **Repository:** WongKinYiu / YOLOv9
- **GitHub:** https://github.com/WongKinYiu/yolov9

Experiment 2–4 使用的：

```text
yolov9-s.pt
```

pretrained weights 亦來自 YOLOv9 官方專案。

本專案沒有修改 YOLOv9 的 model architecture，主要工作集中在：

- Custom dataset preparation
- Pretrained fine-tuning
- Dataset expansion
- Model evaluation
- Error analysis
- Experiment comparison

### UA-DETRAC

Experiment 3 與 Experiment 4 使用的 traffic-scene images 與 annotations 來自官方 **UA-DETRAC** dataset：

- **Dataset:** UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking
- **Official Page:** https://sites.google.com/view/daweidu/projects/ua-detrac

UA-DETRAC 原始 annotations 使用 XML format。

本專案只取：

```text
vehicle_type="car"
```

並轉換成 YOLO annotation format：

```text
<class_id> <x_center> <y_center> <width> <height>
```

所有 car 都對應為：

```text
0: car
```

Experiment 3 從 UA-DETRAC 官方 Train sequences 中分散抽取 500 frames，並與原本 23 張 custom training images 合併，形成：

```text
523 training images
3242 car instances
```

另外從官方 Test sequences 中建立：

```text
200 images
1844 car instances
```

作為 Experiment 3 與 Experiment 4 的 diagnostic benchmark。

### Dataset Redistribution

本 GitHub repository **不直接重新散布完整 UA-DETRAC dataset**。

以下大型或可重新產生的資料不會上傳至 GitHub：

```text
UA-DETRAC raw images
UA-DETRAC image archives
Generated training images
Generated test images
```

Repository 只保留 dataset preparation、annotation conversion、sampling 與 evaluation 所需要的程式碼與設定檔。

---

## Folder Structure

GitHub repository 只保留本專案自行撰寫的 preprocessing、evaluation、analysis scripts，以及實驗結果與設定檔。

大型 datasets、外部 repositories、model checkpoints 與 YOLO generated runs 不會上傳至 GitHub。

```text
YOLOCOCO/
├── README.md
├── .gitignore
│
├── data/
│   ├── car.yaml
│   ├── car_exp3.yaml
│   └── car_exp3_eval.yaml
│
├── results/
│   ├── experiment1/
│   │   ├── results.png
│   │   ├── results.csv
│   │   ├── PR_curve.png
│   │   ├── F1_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── opt.yaml
│   │   └── val_batch0_pred.jpg
│   │
│   ├── experiment2/
│   ├── experiment3/
│   ├── experiment4/
│   │
│   ├── diagnostic/
│   │   ├── exp3/
│   │   │   ├── object_size_analysis.csv
│   │   │   ├── sequence_analysis.csv
│   │   │   ├── PR_curve.png
│   │   │   ├── F1_curve.png
│   │   │   └── confusion_matrix.png
│   │   │
│   │   └── exp4/
│   │       ├── object_size_analysis.csv
│   │       ├── sequence_analysis.csv
│   │       ├── PR_curve.png
│   │       ├── F1_curve.png
│   │       └── confusion_matrix.png
│   │
│   └── inference/
│       └── final_inference.jpeg
│
├── split_car_dataset.py
├── prepare_detrac_exp3.py
├── check_exp3_labels.py
├── prepare_detrac_test.py
├── check_test_labels.py
└── analyze_test_errors.py
```

以下內容只保留在 local environment，不會直接上傳至 GitHub：

```text
yolov9/
labelImg/
data/UA-DETRAC_raw/
data/car_raw/
data/car_dataset/
data/car_dataset_exp3/
data/car_test/
data/exp3_sanity_check/
data/test_sanity_check/
data/test_error_analysis/
data/test_error_analysis_exp4_640/
model checkpoints (*.pt)
```

其中：

- `yolov9/` 與 `labelImg/` 為 external repositories
- UA-DETRAC raw data 與 generated datasets 體積較大，可由官方資料與 preprocessing scripts 重建
- YOLO training / test runs 的重要結果已整理至 `results/`

---

## Method

### 1. Manual Labeling

先使用 LabelImg 將 custom images 標註為 YOLO bounding-box format，並確認：

```text
Images: 29
Labels: 29
Missing labels: 0
Extra labels: 0
```

### 2. Train / Validation Split

將 29 張 custom images 固定切成：

```text
23 train
6 validation
```

Validation set 在四次實驗中保持不變，方便比較不同 training strategy。

### 3. Dataset Expansion

Experiment 3 使用官方 UA-DETRAC Train sequences，選出 500 個包含 car 的 frames，並轉成單一類別 `car` 的 YOLO labels。

最後：

```text
523 training images
3242 car instances
```

### 4. Diagnostic Evaluation

Experiment 3 後，從 UA-DETRAC 官方 Test sequences 建立 200-image benchmark。

YOLO 官方 evaluation 主要觀察：

```text
Precision
Recall
mAP@0.5
mAP@0.5:0.95
```

另外使用 `analyze_test_errors.py` 進行 diagnostic analysis：

```text
Confidence threshold = 0.217
IoU threshold = 0.5
```

分析：

```text
TP / FP / FN
Small / Medium object Recall
Sequence-level performance
```

為了公平比較 Experiment 3 與 Experiment 4，object-size categories 都使用固定的 `320×320` reference definition。

### 5. Final External Inference

最後使用 Experiment 4 的 `best.pt` 對一張額外道路圖片進行 inference，確認模型能被實際載入與使用。

---

# Training

## Experiment 1 — From Scratch Baseline

### Motivation

第一個實驗先建立 baseline，確認 custom dataset、YOLO annotation、Apple MPS 與 YOLOv9 training pipeline 都能正常運作。

這次不使用 pretrained weights，而是讓模型從 random initialization 開始學習。

### Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv9-T |
| Training Images | 23 |
| Validation Images | 6 |
| Epochs | 30 |
| Batch Size | 2 |
| Image Size | 320 × 320 |
| Device | Apple MPS |
| Pretrained Weights | No |

### Training Command

```bash
python train_dual.py \
  --data ../data/car.yaml \
  --cfg models/detect/yolov9-t.yaml \
  --weights '' \
  --epochs 30 \
  --batch-size 2 \
  --img 320 \
  --device mps \
  --name car_yolov9_t
```

`--weights ''` 代表從 random initialization 開始訓練。

### Results

使用 `best.pt` 在 validation set 上：

| Metric | Result |
|---|---:|
| Precision | 0.00743 |
| Recall | 0.222 |
| mAP@0.5 | 0.0552 |
| mAP@0.5:0.95 | 0.0265 |

### Observations

- Training pipeline 可以正常完成，也能產生 `best.pt`。
- 模型可以正常讀取 custom `car` dataset。
- Apple MPS 可以正常進行 training。
- 模型偵測能力很低，大量 ground-truth cars 被判成 background。
- Recall 只有 `0.222`，False Negative 很嚴重。
- mAP@0.5 只有 `0.0552`。
- Loss 波動較大，沒有形成穩定的 convergence。
- Dataset 中包含近距離大車、遠距離小車與多車場景，但只有 23 張 training images。

主要問題：

```text
Training data 太少
+
Random initialization
+
Scene / object-size variation 大
```

因此 Experiment 1 作為 baseline，不作為最終可用模型。

---

## Experiment 2 — Pretrained Fine-tuning

### Why Change?

Experiment 1 只有 23 張 training images，卻要求模型從 random weights 重新學習完整的 car visual representation，因此幾乎無法形成有效 detector。

所以 Experiment 2 改為使用 pretrained weights，測試 pretrained visual features 是否能改善 small-data training。

另外，這次使用 YOLOv9-S，因此相較 Experiment 1 同時改變了：

```text
Random initialization → Pretrained weights
YOLOv9-T → YOLOv9-S
```

因此結果可以證明 Experiment 2 整體策略較好，但不能把所有 improvement 都單獨歸因於 pretrained weights。

### Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv9-S |
| Training Images | 23 |
| Validation Images | 6 |
| Epochs | 30 |
| Batch Size | 2 |
| Image Size | 320 × 320 |
| Device | Apple MPS |
| Pretrained Weights | `yolov9-s.pt` |

### Training Command

```bash
python train_dual.py \
  --data ../data/car.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights yolov9-s.pt \
  --epochs 30 \
  --batch-size 2 \
  --img 320 \
  --device mps \
  --name car_yolov9_s_pretrained
```

Training 開始時：

```text
Transferred 1760/1772 items from yolov9-s.pt
```

### Results

| Metric | Experiment 1 | Experiment 2 |
|---|---:|---:|
| Precision | 0.00743 | **0.423** |
| Recall | 0.222 | **0.447** |
| mAP@0.5 | 0.0552 | **0.450** |
| mAP@0.5:0.95 | 0.0265 | **0.304** |

Best F1：

```text
~0.45 at confidence ~0.095
```

### Improvement over Experiment 1

```text
mAP@0.5
0.0552 → 0.450
```

模型已能正確偵測部分尺寸較大、外觀清楚的 cars，而不再像 Experiment 1 幾乎無法產生有效 prediction。

### Observations

- Pretrained weights 成功載入並完成 fine-tuning。
- Precision、Recall、mAP 與 F1 都大幅提升。
- 大型、清楚、近距離 car 已可被正確偵測。
- Small / distant cars 仍然容易漏檢。
- Highway multi-car scenes 的 False Negative 仍然很明顯。
- Validation set 仍只有 6 張，因此 metric 波動較大。

Experiment 2 證明 small-data training 可以透過 pretrained model 大幅改善，但 23 張 training images 仍不足以建立穩定的 traffic-scene detector。

---

## Experiment 3 — Dataset Expansion

### Why Change?

Experiment 2 已能辨識明顯的 car，但在：

```text
Small cars
Distant cars
Multi-car scenes
Complex traffic scenes
```

仍然有大量 False Negative。

因此 Experiment 3 不再只調模型，而是增加 training data 與 scene diversity。

主要變化：

```text
23 training images
        ↓
523 training images
```

並維持：

```text
YOLOv9-S
Pretrained weights
30 epochs
Batch size 2
320 × 320
```

### Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv9-S |
| Pretrained | Yes |
| Training Images | 523 |
| Training Boxes | 3242 |
| Validation Images | 6 |
| Epochs | 30 |
| Batch Size | 2 |
| Image Size | 320 × 320 |
| Device | Apple MPS |

### Training Command

```bash
python train_dual.py \
  --data ../data/car_exp3.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights yolov9-s.pt \
  --epochs 30 \
  --batch-size 2 \
  --img 320 \
  --device mps \
  --name car_yolov9_s_pretrained_exp3
```

### Validation Results

| Metric | Experiment 2 | Experiment 3 |
|---|---:|---:|
| Precision | 0.423 | **0.865** |
| Recall | 0.447 | **0.889** |
| mAP@0.5 | 0.450 | **0.923** |
| mAP@0.5:0.95 | 0.304 | **0.514** |

Best F1：

```text
~0.88 at confidence ~0.232
```

### Improvement over Experiment 2

```text
Recall
0.447 → 0.889

mAP@0.5
0.450 → 0.923
```

增加 traffic-camera、small-car 與 distant-car examples 後，模型在固定 validation set 上的 detection performance 大幅改善。

### Observations

- Precision、Recall、mAP 與 F1 都明顯優於 Experiment 2。
- Training loss 呈現較穩定下降：
  - `box_loss` 約下降至 `1.48`
  - `cls_loss` 約下降至 `0.98`
  - `dfl_loss` 約下降至 `1.21`
- Highway multi-car scenes 中可以偵測更多遠距離車輛。
- False Negative 明顯下降。
- 少數 close-up car images 仍可能出現 localization / vehicle-part confusion。
- 6-image validation set 太小，因此需要更大的 benchmark 來確認 generalization。

### Diagnostic Benchmark

Experiment 3 使用 200-image UA-DETRAC benchmark：

| Metric | Result |
|---|---:|
| Precision | 0.709 |
| Recall | 0.579 |
| mAP@0.5 | 0.646 |
| mAP@0.5:0.95 | 0.431 |

結果明顯低於 6-image validation，表示原本 validation set 對 generalization 的估計過於樂觀。

進一步 diagnostic analysis：

```text
TP = 1073
FP = 448
FN = 771
Precision = 0.705
Recall = 0.582
F1 = 0.638
```

Object-size analysis：

| Object Size | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| Small | 1524 | 784 | 740 | **0.514** |
| Medium | 320 | 289 | 31 | **0.903** |

771 個 False Negatives 中有 740 個來自 small cars。

因此 Experiment 3 最主要的 bottleneck 被定位為：

```text
Small / distant car detection
```

這個 error analysis 直接成為 Experiment 4 的設計依據。

---

## Experiment 4 — Higher Input Resolution

### Why Change?

Experiment 3 的 error analysis 顯示：

```text
Small Recall = 0.514
Medium Recall = 0.903
```

而且：

```text
771 total FN
740 small-car FN
```

代表模型不是不會辨識 car，而是對 small / distant cars 的 detection ability 明顯不足。

因此 Experiment 4 保持 model、dataset、epochs 與 batch size 不變，只將 input resolution：

```text
320 × 320
        ↓
640 × 640
```

Hypothesis：

> Higher input resolution 可以保留更多 small-object visual information，進而降低 small-car False Negative。

### Configuration

| Parameter | Experiment 3 | Experiment 4 |
|---|---:|---:|
| Model | YOLOv9-S | YOLOv9-S |
| Pretrained | Yes | Yes |
| Training Images | 523 | 523 |
| Epochs | 30 | 30 |
| Batch Size | 2 | 2 |
| Image Size | 320 × 320 | **640 × 640** |
| Device | Apple MPS | Apple MPS |

### Training Command

```bash
python train_dual.py \
  --data ../data/car_exp3.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights yolov9-s.pt \
  --epochs 30 \
  --batch-size 2 \
  --img 640 \
  --device mps \
  --name car_yolov9_s_pretrained_exp4_640
```

Training time：

```text
~2.88 hours
```

### Validation Results

| Metric | Experiment 3 | Experiment 4 |
|---|---:|---:|
| Precision | 0.865 | **0.979** |
| Recall | **0.889** | 0.833 |
| mAP@0.5 | 0.923 | **0.972** |
| mAP@0.5:0.95 | 0.514 | **0.592** |
| Best F1 | ~0.88 | **~0.91** |

Validation 上 Precision、mAP 與 F1 都提高，但 Recall 沒有同步提升，因此不能只靠 6-image validation 判斷 small-object 問題是否解決。

### Diagnostic Benchmark Results

使用與 Experiment 3 相同的 200 images / 1844 car instances：

| Metric | Experiment 3 — 320 | Experiment 4 — 640 |
|---|---:|---:|
| Precision | 0.709 | **0.736** |
| Recall | 0.579 | **0.717** |
| mAP@0.5 | 0.646 | **0.774** |
| mAP@0.5:0.95 | 0.431 | **0.571** |

### Small-object Analysis

固定使用相同的 `320×320` reference definition：

| Object Size | Experiment 3 Recall | Experiment 4 Recall |
|---|---:|---:|
| Small | 0.514 | **0.720** |
| Medium | 0.903 | **0.950** |

Small-car False Negatives：

```text
740 → 426
```

總 False Negatives：

```text
771 → 442
```

### Improvement over Experiment 3

```text
Diagnostic Recall
0.579 → 0.717

Diagnostic mAP@0.5
0.646 → 0.774

Diagnostic mAP@0.5:0.95
0.431 → 0.571

Small Recall
0.514 → 0.720
```

Small Recall 提升 `20.6` percentage points，small-car FN 減少 314 個。

因此 Experiment 4 的結果支持原本 hypothesis：

> Increasing input resolution from 320×320 to 640×640 improves small and distant car detection.

### Observations

- Small and distant car detection 明顯改善。
- 多個原本低 Recall sequences 也有提升，例如：
  - `MVI_40761`: `0.250 → 0.525`
  - `MVI_40855`: `0.354 → 0.630`
  - `MVI_40864`: `0.298 → 0.512`
- mAP@0.5:0.95 提升，表示較高 resolution 也改善 bounding-box localization。
- 在固定 diagnostic confidence `0.217` 下，FP 從 `448 → 668`，表示 higher resolution 雖然找回更多 cars，也產生更多 candidate detections。
- 部分 close-up images 仍會把 vehicle parts 誤判成完整 car。
- 部分 complex traffic scenes 仍同時存在 FN 與 FP。
- 640×640 的 training cost 明顯提高：Experiment 3 約 `0.595 h`，Experiment 4 約 `2.88 h`。

因此 Experiment 4 改善了主要的 small-object bottleneck，但仍存在 accuracy / computational cost 與 False Positive 的 trade-off。

---

## Final Model & External Inference

最終選擇 Experiment 4：

```text
YOLOv9-S
Pretrained: yolov9-s.pt
Training images: 523
Input resolution: 640 × 640
Epochs: 30
Batch size: 2
```

Weights：

```text
runs/train/car_yolov9_s_pretrained_exp4_640/weights/best.pt
```

External inference command：

```bash
python detect_dual.py \
  --weights runs/train/car_yolov9_s_pretrained_exp4_640/weights/best.pt \
  --source ../inference_images/road_test.jpeg \
  --img 640 \
  --device mps \
  --conf-thres 0.25 \
  --name final_inference
```

結果：

```text
Detected objects: 26 cars
Inference time: 163.2 ms
NMS time: 9.2 ms
Device: Apple MPS
```

Qualitative observations：

- Near / medium-distance cars 大多能以約 `0.8–0.9` confidence 被偵測。
- 許多 distant cars 也能被偵測，但 confidence 通常較低。
- Larger cars 的 bounding boxes 整體定位合理。
- Dense distant traffic 中 labels 容易互相重疊，主要是 visualization 問題。
- 這張 external image 沒有 ground-truth annotation，因此只作 qualitative demonstration，不能由此計算 Precision、Recall 或 mAP。

---

## Conclusion

四次實驗形成一條明確的改善流程：

| Experiment | Main Change | Key Result |
|---|---|---|
| Experiment 1 | YOLOv9-T, scratch, 23 images | mAP@0.5 = 0.0552 |
| Experiment 2 | YOLOv9-S + pretrained | mAP@0.5 = 0.450 |
| Experiment 3 | Increase training data to 523 | Diagnostic mAP@0.5 = 0.646 |
| Experiment 4 | 320 → 640 input resolution | Diagnostic mAP@0.5 = **0.774** |

主要發現：

```text
Pretrained weights
        ↓
改善 small-data training

More training data
        ↓
改善 generalization

Higher input resolution
        ↓
改善 small / distant car detection
```

最終 Experiment 4 在 diagnostic benchmark 上：

```text
Precision = 0.736
Recall = 0.717
mAP@0.5 = 0.774
mAP@0.5:0.95 = 0.571
Small Recall = 0.720
```

相較 Experiment 3：

```text
Small Recall
0.514 → 0.720
```

顯示提高 input resolution 確實有效降低原本最嚴重的 small-car False Negative 問題。

這個專案最後完成了：

```text
Manual labeling
        ↓
Custom dataset preparation
        ↓
Scratch baseline
        ↓
Pretrained fine-tuning
        ↓
Dataset expansion
        ↓
Independent evaluation
        ↓
Error analysis
        ↓
Hypothesis-driven experiment
        ↓
External inference
```

---

## Future Work

目前不再繼續增加 experiment，後續若要進一步改善，優先考慮：

- **Hard-negative training**：加入 bus、van、truck、vehicle parts 等 non-car examples，降低 False Positive。
- **More diverse traffic scenes**：加入夜間、雨天、遮擋與更複雜道路場景。
- **Deployment-specific data**：使用實際目標攝影機畫面進行 fine-tuning 與 evaluation。
- **Threshold tuning**：根據實際需求在 Precision 與 Recall 間選擇合適 operating point。
- **Efficiency optimization**：研究 640×640 帶來的 accuracy improvement 是否值得額外 training / inference cost。
- **Final untouched test set**：若未來繼續調整模型，應建立新的完全未參與 model decision 的 holdout dataset，避免 diagnostic benchmark 持續影響 model selection。
