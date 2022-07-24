# YOLOv6
[YOLOv6](https://github.com/meituan/YOLOv6) was recently introduced by Chinese company Meituan. It is not part of the official YOLO series but was named so since the authors of this architecture were heavily inspired by the original one-stage YOLO.  Meituan is a large e-commerce company in China, and their technical team is akin to what we might think of in the United States when we think about Amazon Research.

YOLOv6 is a target detection framework dedicated to industrial applications. As per the company’s release, the most used YOLO detection frameworks – YOLOv5, YOLOX, and PP-YOLOE – leave a lot of room for improvement in terms of speed and accuracy. Recognising these ‘flaws,’ Meituan has introduced MT-YOLOv6 by studying and drawing further on the existing technologies in the industry. The MT-YOLOv6 framework supports the entire chain of industrial applications requirements like model training, inference, and multiplatform deployment. According to the team, MT-YOLOv6 has carried out improvements and optimisations at the algorithmic level, like training strategies and network structure, and has displayed impressive results in terms of accuracy and speed when tested on COCO datasets.
YOLOv6 is a single-stage object detection framework dedicated to industrial applications, with hardware-friendly efficient design and high performance. It outperforms YOLOv5 in detection accuracy and inference speed, making it the best OS version of YOLO architecture for production applications.

Unlike YOLOv5/YOLOX, which are based on CSPNet and use a multi-branch approach and residual structure, Meituan redesigned the Backbone and Neck according to the idea of hardware-aware neural network design. As per the team, this helps in overcoming the challenges of latency and bandwidth utilisation. The idea is based on the characteristics of hardware and that of inference/compilation framework. Meituan introduced two redesigned detection components – EfficientRep Backbone and Rep-PAN Neck.

Further, the researchers at Meituan adopted the decoupled head structure, taking into account the balance between the representation ability of the operators and the computing overhead on the hardware. They used a hybrid strategy to redesign a more efficient decoupling head structure. The team observed that with this strategy, they were able to increase the accuracy by 0.2 per cent and speed by 6.8 per cent.

In terms of training, Meituan adopted three strategies:
- Anchor-free paradigm: This strategy has been widely used in recent years due to its strong generalisation ability and simple code logic. Compared to other methods, the team found that the Anchor-free detector had a 51 per cent improvement in speed.
- SimOTA Tag Assignment Policy: To obtain high-quality positive samples, the team used the SimOTA algorithm that dynamically allocates positive samples to improve detection accuracy.
- SIoU bounding box regression loss: YOLOv6 adopts the SIoU bounding box regression loss function to supervise the learning of the network. The SIoU loss function redefines the distance loss by introducing a vector angle between required regression. This improves the regression accuracy, resulting in improved detection accuracy.

# YOLOv6 Inferencing using OpenVINO
Now we will see how YOLOv6 model inferencing can be done using Intel OpenVINO toolkit.

The following components are required-

- OpenVINO toolkit
- Model Optimizer - For Openvino toolkit (version < 2022.1) , Model optimizer comes included in the toolkit. But for 2022.1 versions onwards, OpenVINO development tools (like model optimizer) need to be installed seperately.
- System – Intel CPU/ GPU/ VPU
- Python
- ONNX
- Pytorch
- Netron model visualizer

## OpenVINO toolkit Installation
### For OpenVINO version >= 2022.1
1. [Install](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) OpenVINO toolkit 2022.1 or any other recent version of your choice. Download and install suitable toolkit depending on your operating system. Follow all the required install instructions. Here the OpenVINO development tools are not included in the toolkit, they need to be installed seperately.
2. From the 2022.1 release, the [OpenVINO Development Tools](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_install_dev_tools.html#doxid-openvino-docs-install-guides-install-dev-tools) (like Model Optimizer, Benchmark Tool, Accuracy Checker & Post-Training Optimization Tool, etc) can only be installed via [PyPI](https://pypi.org/project/openvino-dev/). Download and install the Development Tools Package from PyPI where we use Model optimizer for our YOLOv5 optimization.


### For OpenVINO version < 2022.1
1. [Install](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download-previous-versions.html) OpenVINO toolkit 2021.4 or 2021.3 or any other older version of your choice. Download and install suitable toolkit depending on your operating system. Follow all the required install instructions.
2. Here Model optimizer comes included in the toolkit.

## YOLOv6 inferencing
### Windows/Linux :
#### Clone the YOLOv6 repository from Github
1. Clone the latest YOLOv6 [repository](https://github.com/meituan/YOLOv6) and install requirements.txt in Python>=3.7.0 environment, including PyTorch>=1.7. Models and datasets download automatically from the latest YOLOv6 release.
Running the following commands(one after the other) in the terminal of Windows/Linux-
```
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```

2. For Windows OS - Download latest Wget exe file from [here](https://eternallybored.org/misc/wget) and then copy wget.exe to your C:\Windows\System32 folder location. Skip this step incase of Linux.

#### Downloading Pytorch weights and Converting them to ONNX weights
1. There are various pretrained models to start training from. 
![Yolov6](https://user-images.githubusercontent.com/37048080/180662039-95fda16f-82b6-4ff1-89c9-b4ce0e32ed42.JPG)
Here we use YOLOv6-tiny model for inferencing on OpenVINO. Run the following command to download the YOLOv6t pytorch weights-
```
wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.pt
```

2. Convert Pytorch Weights to ONNX Weights and to Intermediate Representation - 

The speciality with YOLOV6 is that Meituan team provided python script for directly converting Pytorch weights to OpenVINO IR using Model Optimizer.
The YOLOv6 repository provides a script export_openvino.py in deploy/OpenVINO to export Pytorch weights with extensions *.pt to ONNX weights with extensions *.onnx and also to generate OpenVINO IR(.xml, .bin and .mapping files). Run the following command to download & convert the latest version of YOLOv6t Pytorch Weights(yolov6t.pt) to ONNX weights & IR:

```
python deploy/OpenVINO/export_openvino.py --weights yolov6s.pt --img 640 --batch 1
```
Then we can get yolov6t.onnx in the original location and IR files in yolov6t_openvino folder.

#### YOLOv6 Inference Demo

1. After we download the pretrained YOLOv6t model, use the foloowing command for inferencing images on YOLOv6t model.
2. Run the following commands -
```
wget -O test.jpg https://i.imgur.com/1IWZX69.jpg
wget -O face-demographics-walking.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4
wget https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/traffic1.jpg
wget https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/traffic2.jpgpy
```
```
python tools/infer.py --weights yolov6t.pt --source <path to image/directory>
```
Output will be saved at runs/inference/exp by default

--weights    : Model path for inference

--source     : Image file/Image path

--yaml       : Yaml file for data

--img-size   : Image size (h,w) for inference size

--conf-thres : confidence threshold for inference

--iou-thres  : NMS iou thresold for inference

--max-det    : maximum inference per image

--device     : device to run model like 0,1,2,3 or cpu

--save-txt   : save results to *.txt

--save-img   : save visualized inference results

--classes    : filter by classes 

--project    : save inference results to project/name

3. Demos-
```
python tools/infer.py --weights yolov6t.pt --source data/images
```
![image1](https://user-images.githubusercontent.com/37048080/180662698-c4541bec-b98c-4e7d-b141-576733c8ef03.jpg)

![image2](https://user-images.githubusercontent.com/37048080/180662701-bb7e7d2c-e012-4670-ac1b-a53b29f47193.jpg)

![image3](https://user-images.githubusercontent.com/37048080/180662704-e8339170-9e3b-4597-89ba-552d4aff1251.jpg)

```
python tools/infer.py --weights yolov6t.pt --source test.jpg
```
![test](https://user-images.githubusercontent.com/37048080/180662767-87998b57-d439-4e3b-a480-415a034080c3.jpg)


# Performance
## Speed Test using Intel [Benchmark Tool](https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html#benchmark-python-tool)
This tool estimates deep learning inference performance on supported devices. Performance can be measured for two inference modes: latency and throughput-oriented.
This tool is already installed when we download & install the OpenVINO Development tools via PyPI.
All the development tools are located here- C:\Users\xxxxx\openvino_env\Lib\site-packages\openvino\tools

Run the following command to run the Benchmark Python tool for estimating the YOLOv6t performance on test images-
```
benchmark_app -m yolov6t_openvino/yolov6t.xml -i data/images -d CPU -niter 100 -progress
```
![Yolov6_Benchmark](https://user-images.githubusercontent.com/37048080/180663564-af862067-c0c6-4393-bcc6-a008906f135b.JPG)


## [Accuracy Checker Tool](https://docs.openvino.ai/latest/omz_tools_accuracy_checker.html#deep-learning-accuracy-validation-framework)
The Accuracy Checker is an extensible, flexible and configurable Deep Learning accuracy validation framework. The tool has a modular structure and allows to reproduce validation pipeline and collect aggregated quality indicators for popular datasets both for networks in source frameworks and in the OpenVINO™ supported formats.
All the development tools are located here- C:\Users\xxxxx\openvino_env\Lib\site-packages\openvino\tools

# Conclusion
Though it provides outstanding results, it’s important to note that MT-YOLOv6 is not part of the official YOLO series. As of now, This name is not official and still in progress about the naming of YOLOv6. 
As a new repository, YOLOv6 is slightly harder to wield in practice than YOLOv5 and does not have as many established pathways and articles around using the network in practice for training, deployment and debugging - something that may change with time.
