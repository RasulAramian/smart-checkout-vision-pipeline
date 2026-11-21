# Smart Checkout Vision Pipeline

Modular computer vision pipeline for smart checkout and basket product tracking using **YOLOv8**, **OpenCV**, **ByteTrack**, and **FastAPI**.

## Model Performance
- **Backbone:** Fine-tuned `YOLOv8n` (`models/best.pt`) on 15 supermarket product classes
- **Validation Accuracy:** `88.5%` mAP@50 (`0.583` mAP@50-95)
- **Inference Speed:** `~2.2 ms` per frame (on validation benchmark)

## Project Structure
```text
smart-checkout-vision-pipeline/
├── app.py                     # FastAPI service for frame detection & basket state
├── src/
│   ├── __init__.py
│   └── basket_tracker.py      # YOLOv8 detection + ByteTrack tracking logic
├── models/
│   └── best.pt                # Fine-tuned YOLOv8 weights
├── notebooks/
│   └── 01_train_yolov8_detector.ipynb
├── assets/                    # Training & evaluation plots
├── tests/
│   └── test_basket_tracker.py # Unit & pipeline smoke tests
├── Dockerfile
└── requirements.txt