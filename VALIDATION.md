# Validation · 18 October 2026

## Executed evaluation

All 64 frozen KITTI sample IDs were processed by YOLOX-S on original images, YOLOX-S on brightness-scaled images, and HOG + SVM on original images. No failed scene was removed. The 80 pedestrian annotations include six boxes under 40 pixels tall and 29 with occlusion level at least one.

YOLOX original: 46 true positives, 10 false positives, 34 false negatives, 12 ignored predictions. The committed JSON includes every individual outcome. Model weights were checked against the published Git LFS SHA-256 before inference; original image checksums are retained. Reported latency excludes one model warm-up and image I/O; CPU, Python, NumPy and OpenCV details are recorded in the JSON. No GPU was used.

## Automated checks

`python -m unittest discover -s tests -v`: **36 tests passed**, zero failures, zero skips, on Python 3.12 / OpenCV 4.13.0 / NumPy 2.3.5, Windows.

Coverage includes letterbox/channel order, known synthetic YOLOX decoding, duplicate suppression, padding-only rejection, model hash enforcement, one-to-one matching, duplicate false positives, ignore-region precedence, empty metric handling, subgroup identity, annotation parsing, image checksum validation, attention-zone geometry, result reconstruction, sample strata, local-server origin/host checks, upload limits and private-path rejection. Server unit tests use a fake detector; the 64-image evaluation uses the real ONNX model.

These tests require neither a model download nor dataset downloads; they inspect the committed results. The GitHub workflow runs Python 3.11 and 3.12. A workflow file does not establish successful GitHub execution; account billing restrictions may prevent runners from starting.

## Browser checks

The dashboard rendered real driving images, measured summary values, detector boxes, missed-pedestrian annotations, and image-space warning zones. Scene 000384 showed 7 matched / 5 missed on original input and 6 matched / 6 missed under dimming. Original, dimmed and HOG results remain separately accessible.

Browser testing also exercised HOG selection (1 matched / 11 missed on scene 000384), a real local PNG upload (7 YOLOX person boxes, no ground-truth metrics), return to the study, and the JSON export. The exported original scene retained 7 true positives, 0 false positives and 5 false negatives. The preview is an actual dashboard screenshot. A stale detector label after switching from HOG to local YOLOX inference was corrected. The server's image route was tightened to prevent parent-path access within the project; regression coverage and all 36 tests passed after the fix.

The evaluation source hashes describe the code at measurement time. Subsequent changes affected server path containment, browser presentation and license headers, not model inference or metric calculation.

## Boundaries

This project detects the COCO person class and evaluates KITTI Pedestrian labels. It does not estimate identity, age, intent or gait. It has no tracking, depth, learned lane detection, vehicle interface, control policy or collision labels. Static-image attention signals are not validated safety warnings. No physical car, live driving stream, night-driving dataset, population-wide generalization, new training or independent official KITTI evaluation was exercised.
