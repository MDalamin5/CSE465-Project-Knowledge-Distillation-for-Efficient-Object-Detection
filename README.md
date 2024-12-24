
# Fine-Grained Feature Imitation for Efficient Object Detection

## Overview
This project demonstrates an approach to object detection that combines **Knowledge Distillation** with a novel **Fine-Grained Feature Imitation** method. By training a lightweight student model to mimic a computationally intensive teacher model, the technique improves object detection performance on resource-constrained devices.

---

## Features
- **Lightweight Model:** Achieves state-of-the-art accuracy with fewer parameters (1.78M for the student model).
- **Enhanced Performance:** mAP@50 of **0.707** and mAP@[0.5:0.95] of **0.435** on Pascal VOC datasets.
- **Generalization:** Successfully tested on diverse datasets like Pascal VOC, BCCD, Lemon Disease, and Incorrect-Mask-2.
- **Simplified Deployment:** Real-time inference on devices without requiring specialized hardware.

---

## Methodology
- **Teacher Model:** YOLOv5x.
- **Student Model:** YOLOv5n.
- **Core Approach:**
  - Introduced a **Combined Imitation Loss Function (CILF)** that integrates:
    - **KL Divergence Loss** for output alignment.
    - **Mean Squared Error (MSE) Loss** for feature alignment.
  - Used fine-grained imitation masks to focus learning on crucial regions in the teacher's feature maps.

---

## Results
### Comparison with YOLOv7-tiny (SOTA Lightweight Model):
| Model            | Parameters (M) | mAP@50 | mAP@[0.5:0.95] |
|-------------------|----------------|--------|----------------|
| YOLOv7-tiny      | 6.2            | 0.644  | 0.385          |
| **Our Model**     | **1.78**       | **0.707** | **0.435**      |

### Generalization Across Datasets:
| Dataset            | Precision | Recall | mAP@50 | mAP@[0.5:0.95] |
|---------------------|-----------|--------|--------|----------------|
| **Pascal VOC**      | 0.722     | 0.644  | 0.707  | 0.433          |
| **BCCD**            | 0.873     | 0.881  | 0.916  | 0.628          |
| **Lemon Disease**   | 0.928     | 0.888  | 0.925  | 0.693          |

---

## Key Files
- **`model.py`:** Implementation of YOLOv5x and YOLOv5n models.
- **`loss.py`:** Definition of the Combined Imitation Loss Function (CILF).
- **`train.py`:** Training pipeline for knowledge distillation.
- **`datasets/`:** Scripts for loading Pascal VOC, BCCD, and other datasets.

---

## Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MDalamin5/FGFI-Object-Detection-Using-Knowledge-Distillation.git
   cd SRC-COde
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training
Train the student model using the teacher model:
```bash
python train.py --dataset pascal_voc --teacher_model yolov5x --student_model yolov5n
```

### Evaluation
Evaluate the trained model:
```bash
python evaluate.py --model yolov5n --dataset pascal_voc
```

---

## Future Work
- Extending the approach to transformer-based object detection models.
- Testing on additional datasets to validate generalizability.
- Exploring advanced imitation strategies for improved feature alignment.

---

## Citation
If you use this work, please cite:
```
Md Al Amin, "Fine-Grained Feature Imitation for Efficient Object Detection Using Knowledge Distillation," North South University, 2024.
```

---

## Acknowledgments
Special thanks to Dr. Nabeel Mohammed for supervision and guidance.

---

## License
This project is licensed under the MIT License.
