# YOLOv7
Just a few days ago, [YOLOv7](https://github.com/WongKinYiu/yolov7) came into the limelight by beating all the existing object detection models to date. Anyone who has worked in Object detection has heard about YOLOs. It’s been here for a while now, and to date, we have seen a lot of YOLO versions. YOLO is not a single architecture but a flexible research framework written in low-level languages.
YOLOv7 outperforms all existing state of art object detection models in terms of speed and highest accuracy of 56.8% AP on COCO dataset. Yolo

## Benchmarks
In terms of speed and accuracy, YOLOv7 has now surpassed all of the known object detectors like [YOLOR](https://github.com/WongKinYiu/yolor), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [Scaled-YOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4), [YOLOv5](https://github.com/ultralytics/yolov5), [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [DINO-5scale-R50](https://github.com/IDEACVR/DINO), [ViT-Adapter-B](https://github.com/czczup/ViT-Adapter). Its speed varies from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP. Also runs on 30FPS + on V100 GPU.

YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector [SWIN](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) & [Cascade-Mask R-CNN](https://arxiv.org/abs/1906.09756) by 509% in speed and 2% in accuracy, and convolutional based detector [ConvNeXt-XL](https://github.com/facebookresearch/ConvNeXt) Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy.

## Features
- Several trainable bag-of-freebies methods have greatly improve the detection accuracy without increasing the inference cost.
- Proposed “extend” and “compound scaling” methods for the real-time object detector that can effectively utilize parameters and computation.
- Reduced about 40% parameters and 50% computation of state-of-the-art real-time object detector, made faster inference speed and higher detection accuracy.
- A simple and standard training framework for any detection & instance segmentation tasks, based on [𝐝𝐞𝐭𝐞𝐜𝐭𝐫𝐨𝐧𝟐](https://github.com/facebookresearch/detectron2)
- Supports [𝐃𝐄𝐓𝐑](https://arxiv.org/pdf/2005.12872.pdf) and many transformer-based detection frameworks out-of-box
- Supports easy-to-deploy pipeline thought onnx
- This is the only framework that supports 𝐘𝐎𝐋𝐎𝐯𝟒 + 𝐈𝐧𝐬𝐭𝐚𝐧𝐜𝐞𝐒𝐞𝐠𝐦𝐞𝐧𝐭𝐚𝐭𝐢𝐨𝐧 in single-stage style
- Easily plugin into transformers-based detector
- YOLOv7 added instance segmentation to the YOLO arch. Also many transformer backbones, arches included.
- YOLOv7 will have other versions that will deal with image segmentation and pose estimation. In the github repo of YOLOv7 they show a teaser about YOLOv7-mask and YOLOv7-pose, which would be image segmentation and pose estimation models respectively.
- YOLOv7 achieves mAP 43, AP-s exceed MaskRCNN by 10 with a convex-tiny backbone while the similar speed with YOLOX-s, more models listed below, it's more accurate and even lighter.

# YOLOv7 Inferencing using OpenVINO toolkit
Now we will see how YOLOv7 model inferencing can be done using Intel OpenVINO toolkit.

The following components are required-

- OpenVINO toolkit
- Model Optimizer - For Openvino toolkit (version < 2022.1) , Model optimizer comes included in the toolkit. But for 2022.1 versions onwards, OpenVINO development tools (like model optimizer) need to be installed seperately.
- System – Intel CPU/ GPU/ VPU
- Python
- ONNX
- Pytorch
- Netron model visualizer

## OpenVINO toolkit Installation
### For OpenVINO version < 2022.1
1. [Install](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download-previous-versions.html) OpenVINO toolkit 2021.4 or 2021.3 or any other older version of your choice. Download and install suitable toolkit depending on your operating system. Follow all the required install instructions.
2. Here Model optimizer comes included in the toolkit.

### For OpenVINO version >= 2022.1
1. [Install](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) OpenVINO toolkit 2022.1 or any other recent version of your choice. Download and install suitable toolkit depending on your operating system. Follow all the required install instructions. Here the OpenVINO development tools are not included in the toolkit, they need to be installed seperately.
2. From the 2022.1 release, the [OpenVINO Development Tools](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_install_dev_tools.html#doxid-openvino-docs-install-guides-install-dev-tools) (like Model Optimizer, Benchmark Tool, Accuracy Checker & Post-Training Optimization Tool, etc) can only be installed via [PyPI](https://pypi.org/project/openvino-dev/). Download and install the Development Tools Package from PyPI where we use Model optimizer for our YOLOv5 optimization.

## YOLOv5 inferencing using OpenVINO toolkit
### Windows/Linux :
#### Clone the YOLOv5 repository from Github
1. Clone the latest YOLOv5 [repository](https://github.com/ultralytics/yolov5) and install requirements.txt in Python>=3.7.0 environment, including PyTorch>=1.7. Models and datasets download automatically from the latest YOLOv5 release.
Running the following commands(one after the other) in the terminal of Windows/Linux-
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

2. For Windows OS - Download latest Wget exe file from [here](https://eternallybored.org/misc/wget) and then copy wget.exe to your C:\Windows\System32 folder location. Skip this step incase of Linux.

#### Downloading Pytorch weights and Converting them to ONNX weights
1. There are various pretrained models to start training from. Here we select YOLOv5s, the smallest and fastest model available.
![YOLOv5 pretrained models](https://user-images.githubusercontent.com/37048080/179829729-3eb55365-fbee-40e1-b4e7-48c22206d2b7.png)

2. Convert Pytorch Weights to ONNX Weights - The YOLOv5 repository provides a script export.py to export Pytorch weights with extensions *.pt to ONNX weights with extensions *.onnx. Run the following command to download & convert the latest version of YOLOv5s Pytorch Weights(yolov5s.pt) to ONNX weights:

```
python export.py  --weights yolov5-v6.1/yolov5s.pt  --img 640 --batch 1
```
Then we can get yolov5s.onnx in yolov5-v6.1 folder containing ONNX version of YOLOv5s.

#### Convert ONNX weights file to OpenVINO IR(Intermediate Representation)
1. After we get ONNX weights file from the last section, we can convert it to IR file with model optimizer. We need to specify the output node of the IR when we use model optimizer to convert the YOLOv5 model. There are 3 output nodes in YOLOv5.
2. Download & Install [Netron](https://github.com/lutzroeder/netron)  or use Netron [web app](https://netron.app/) to visualize the YOLOv5 ONNX weights. Then we find the output nodes by searching the keyword “Transpose” in Netron. After that, we can find the convolution node marked as oval shown in following Figure. After double clicking the convolution node, we can read its name “Conv_198” for stride 8 on the properties panel marked as rectangle shown in following Figure. We apply this name “Conv_198” of convolution node to specify the model optimizer parameters. Similarly, we can find the other two output nodes “Conv_217” for stride 16 and “Conv_236” for stride 32. 
![YOLOv5_Output_node](https://user-images.githubusercontent.com/37048080/179829580-f2edd2bc-189c-4d70-9e5f-08819a92e1f8.jpg)

3. Run the following command to generate the IR of YOLOv5 model(if OpenVINO version >= 2022.1):

```
Python C:/Users/"Bethu Sai Sampath"/openvino_env/Lib/site-packages/openvino/tools/mo/mo.py --input_model yolov5-v6.1/yolov5s.onnx --model_name yolov5-v6.1/yolov5s -s 255 --reverse_input_channels --output Conv_198,Conv_217,Conv_236
```

If OpenVINO version < 2022.1, run the following command:
```
Python “C:/Program Files (x86)”/Intel/openvino_2021.4.752/deployment_tools/model_optimizer/mo.py --input_model yolov5-v6.1/yolov5s.onnx --model_name yolov5-v6.1/yolov5s -s 255 --reverse_input_channels --output Conv_198,Conv_217,Conv_236
```

Where --input_model defines the pre-trained model, the parameter --model_name is name of the network in generated IR and output .xml/.bin files, -s represents that all input values coming from original network inputs will be divided by this value, --reverse_input_channels is used to switch the input channels order from RGB to BGR (or vice versa), --output represents the name of the output operation of the model. 
After this command execution, we get IR of YOLOv5s in FP32 in folder yolov5-v6.1.

4. The result of the optimization process is an IR model. The model is split into two files - model.xml(XML file containing the network architecture) & 
model.bin (binary file contains the weights and biases)
#### YOLOv5 Inference Demo

1. After we generate the IR of YOLOv5 model, use the Python demo(yolov5_openvino_demo.py script) for inferencing process of YOLOv5 model.
2. Download some images/videos and object classes for inferencing. Run the following commands one after the other-
```
wget -O face-demographics-walking.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4
wget -O bottle-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4
wget -O head-pose-face-detection-female.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4
wget https://github.com/bethusaisampath/YOLOv5_Openvino/blob/main/yolo_80classes.txt
wget https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/traffic1.jpg
wget https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/traffic2.jpg
wget https://raw.githubusercontent.com/bethusaisampath/YOLOs_OpenVINO/main/YOLOv5/yolov5_openvino_demo.py
```
The inference Python script can be found at [yolov5_openvino_demo.py](https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/yolov5_openvino_demo.py)

3. Run the following commands for Inferencing-
```
python yolov5_openvino_demo.py -i data/images/bus.jpg -m yolov5-v6.1/yolov5s.xml
```
![Demo_1](https://user-images.githubusercontent.com/37048080/180057938-5f60a73a-e0fc-432d-9841-a2e9bd9e5966.JPG)

```
python yolov5_openvino_demo.py -i data/images/zidane.jpg -m yolov5-v6.1/yolov5s.xml
```
![Demo_2](https://user-images.githubusercontent.com/37048080/180057962-fc23db43-a90f-4500-a855-5b4e8514fce7.JPG)

```
python yolov5_openvino_demo.py -i traffic2.jpg -m yolov5-v6.1/yolov5s.xml
```
![Demo_4](https://user-images.githubusercontent.com/37048080/180059994-e562f53f-6b2f-45b3-9c8f-3aa02f082c57.JPG)

```
python yolov5_openvino_demo.py -i face-demographics-walking.mp4 -m yolov5-v6.1/yolov5s.xml
```
On the start-up, the application reads command-line parameters and loads a network to the Inference Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

https://user-images.githubusercontent.com/37048080/179828046-78eed3dc-00ed-456f-aa80-debe9a9965de.mp4


