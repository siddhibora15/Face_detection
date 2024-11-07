# Advanced Detection System 🎯

A modern desktop application that combines real-time face detection and object detection using OpenCV and YOLOv8. Built with CustomTkinter for a sleek, dark-themed UI, this application offers both camera-based and image-based detection capabilities.

![Preview Image](/screenshots/p.png)

## ✨ Key Features

- Real-time face detection using Haar Cascades
- Object detection using YOLOv8
- Live camera feed processing
- Image file processing
- Screenshot capture functionality
- Dark-themed modern UI
- Real-time detection statistics
- Toggle controls for face/object detection

## 🚀 Prerequisites

```plaintext
Python 3.7+
OpenCV (cv2)
CustomTkinter
PIL (Python Imaging Library)
Ultralytics YOLO
NumPy
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/PIYUSH-JOSHI1/Face-detection-codesoft-.git
cd advanced-detection-system
```

2. Install required packages:
```bash
pip install opencv-python
pip install customtkinter
pip install Pillow
pip install ultralytics
pip install numpy
```

3. Download YOLOv8 model (automatic on first run)

## 💻 Usage

Run the application:
```bash
python face_detection.py
```

### Controls:
- Start/Stop Camera: Toggle camera feed
- Load Image: Process a single image file
- Save Screenshot: Capture current frame
- Face Detection Toggle: Enable/disable face detection
- Object Detection Toggle: Enable/disable object detection

## 🛠️ Technical Details

- Face Detection: Uses Haar Cascade Classifier
- Object Detection: Uses YOLOv8 neural network
- UI Framework: CustomTkinter
- Image Processing: OpenCV & PIL
- Screenshot Storage: Local 'screenshots' directory

## 📊 Features Breakdown

1. Detection Capabilities:
   - Face detection with bounding boxes
   - Multi-object detection with labels
   - Confidence scores for objects
   
2. UI Elements:
   - Real-time statistics display
   - Toggle switches for features
   - Status bar updates
   - Error handling with popup messages

3. Image Handling:
   - Camera feed processing
   - Image file import
   - Screenshot export
   - Automatic image resizing

## ⚙️ Configuration

The application uses default configurations:
- Camera: Default system camera (index 0)
- YOLOv8: Smallest model (yolov8n.pt)
- Max image size: 800px (auto-resizing)
- Screenshot format: PNG

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License

## 🚨 Troubleshooting

Common issues:
- Camera not detected: Check system permissions
- YOLO model download: Ensure internet connectivity
- Image loading fails: Check supported formats
- Performance issues: Reduce frame size or disable features

## 💡 Tips

1. For best performance:
   - Use good lighting for face detection
   - Keep camera stable
   - Ensure subjects are within frame
   
2. For best results:
   - Position faces clearly for detection
   - Avoid overcrowded scenes
   - Maintain good lighting conditions
