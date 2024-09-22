### Football Analysis Project

## Introduction
The Football Analysis Project aims to analyze football match videos by detecting and tracking key objects, including players, referees, and the football itself. The project leverages YOLO (You Only Look Once), one of the most advanced AI models for object detection, to accurately track the movement of these objects throughout the game. Additionally, we enhance YOLO's performance by fine-tuning the model through training.

The project also assigns players to teams based on their t-shirt colors using KMeans clustering for pixel segmentation. This enables the system to differentiate between teams and helps measure important statistics, such as ball acquisition percentage for each team.

By incorporating optical flow, the system can also account for camera movement between frames, ensuring that measurements of players' movement are more precise, even when the camera pans or zooms.

### Requirements:

Python 3.x
ultralytics
supervision
OpenCV
NumPy
Matplotlib
Pandas

## Modules Used
The following modules are used in this project:

YOLO: AI object detection model
Kmeans: Pixel segmentation and clustering to detect t-shirt color
Optical Flow: Measure camera movement
Perspective Transformation: Represent scene depth and perspective
Speed and distance calculation per player


# Output
https://drive.google.com/file/d/1SlaVlTHWqQRG8Z-M4tpJIpQVvIM8Zcf0/view?usp=sharing