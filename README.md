# Object Detection & YOLOs
Object detection is a technique of training computers to detect objects from images or videos. Over the years, there are many object detection architectures and algorithms created by multiple companies and researchers.

YOLO refers to “You Only Look Once” is one of the most versatile and famous object detection models. For every real-time object detection work, YOLO is the first choice by Data Scientist and Machine learning engineers. YOLO algorithms divide all the given input images into the SxS grid system. Each grid is responsible for object detection. Now those Grid cells predict the boundary boxes for the detected object. For every box, we have five main attributes: x and y for coordinates, w and h for width and height of the object, and a confidence score for the probability that the box containing the object.

The YOLO network consists of three main pieces-
1. Backbone - A convolutional neural network that aggregates and forms image features at different granularities.
2. Neck - A series of layers to mix and combine image features to pass them forward to prediction.
3. Head - Consumes features from the neck and takes box and class prediction steps.

## OpenVINO
The Intel distribution of the OpenVINO toolkit is a free to download set of Python and C++ scripts that are used to optimize, tune, and improve the inference of AI models. The toolkit also includes the Intel Open Model Zoo, which contains numerous pre-trained and pre-optimized models ready to be used. Here we will see how latest YOLOs (Yolov5, v6 and v7) models can be inferenced using Intel OpenVINO toolkit.

# Disclaimer
This is not officially tested by Intel OpenVINO. For experimental/research purposes only.
