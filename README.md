# AI-Based Smart Traffic Light System Using YOLOv8

An intelligent traffic control system that dynamically adjusts signal durations based on **real-time pedestrian detection** using **YOLOv8** and computer vision.  
This project demonstrates how **AI and IoT** can be integrated into smart city infrastructure to improve road safety and optimize traffic flow — even on **low-cost hardware** such as the Raspberry Pi.

---

## Overview

Conventional traffic signal systems operate on static timers that cannot adapt to varying pedestrian density.  
This system introduces **deep learning-based adaptive control**, allowing the signal cycle time to automatically adjust depending on the number of detected pedestrians.

### Core Features:
- Real-time pedestrian detection using **fine-tuned YOLOv8**
- Adaptive signal cycle duration (dynamic red-yellow-green control)
- On-screen visualization of current light, pedestrian count, and FPS
- Compatible with **CPU, GPU**, and **Raspberry Pi (3, 4, 5)** boards
- Adjustable detection thresholds for different environments
