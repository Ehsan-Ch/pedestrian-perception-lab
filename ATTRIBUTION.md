# Sources and licenses

## KITTI driving images and labels

The original PNG images and labels come from [KITTI Object Detection Evaluation 2012](https://www.cvlibs.net/datasets/kitti/eval_object.php), a project of Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago. Source archives are recorded in `reports/results.json`.

Citation: Andreas Geiger, Philip Lenz and Raquel Urtasun, **Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite**, CVPR 2012.

[KITTI's license](https://www.cvlibs.net/datasets/kitti/) is [Creative Commons Attribution–NonCommercial–ShareAlike 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). KITTI describes the data as intended for academic/noncommercial use. The JPEG previews in `reports/images/`, embedded annotations and dataset-derived portions of `reports/results.json`, and screenshots containing KITTI imagery are distributed under those terms, not the software MIT license. Changes: PNG-to-JPEG conversion for previews; brightness multiplication for the stress experiment; detection/annotation graphics overlaid in the interface. No endorsement by KITTI or its authors is implied.

The sample was selected before inference, with no hand-picking based on detector performance. Every selected image remains in the report. These metrics are not a KITTI leaderboard submission.

## Pretrained detector

[OpenCV Zoo YOLOX-S](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_yolox), model `object_detection_yolox_2022nov.onnx`. Revision: `47534e27c9851bb1128ccc0102f1145e27f23f98`. SHA-256: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.

The model and the OpenCV reference adapter are Apache-2.0 licensed; see [models/LICENSE-YOLOX.txt](models/LICENSE-YOLOX.txt). Our detector adapter implements the reference RGB, top-left letterbox, raw 0–255 input and YOLOX grid/stride decoding, with person-only filtering, clipped boxes, validation and NMS. The adapter is also provided under Apache-2.0. The 35.9 MB weights are fetched separately by `scripts/setup.py`, not committed.

Original YOLOX paper: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li and Jian Sun, **YOLOX: Exceeding YOLO Series in 2021**, [arXiv:2107.08430](https://arxiv.org/abs/2107.08430). Pretraining uses COCO. No model training or fine-tuning was performed in this project, and upstream training-data provenance is not independently audited.

## Software

OpenCV: Apache-2.0, including its bundled HOG people detector. NumPy: BSD-3-Clause. Their distributions retain their own notices. The HOG baseline uses `HOGDescriptor_getDefaultPeopleDetector()` with default hit threshold, 8-pixel stride/padding and scale factor 1.05.

Other original software in this project: MIT, copyright 2026 Ehsan Cheraghi. Development was AI-assisted. No people are identified, recognized by face, or assigned personal attributes.
