# YOLOv5
After a few days of the release of the YOLOv4 model on 27 May 2020, YOLOv5 got released by [Glenn Jocher](https://www.linkedin.com/in/glenn-jocher/)(Founder & CEO of Utralytics). It was publicly released on Github [here](https://github.com/ultralytics/yolov5). Glenn introduced the YOLOv5 Pytorch based approach, and Yes! YOLOv5 is written in the Pytorch framework. With the continuous effort and 58 open source contributors, YOLOv5 set the benchmark for object detection models very high, it already beats the [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) and its other previous YOLO versions.

The largest contribution of YOLOv5 is to translate the Darknet research framework to the PyTorch framework. The Darknet framework is written primarily in C and offers fine grained control over the operations encoded into the network. In many ways the control of the lower level language is a boon to research, but it can make it slower to port in new research insights, as one writes custom gradient calculations with each new addition. Pytotch inferences are very fast that before releasing YOLOv5, many other AI practitioners often translate the YOLOv3 and YOLOv4 weights into Ultralytics Pytorch weight. There is no official YOLOV5 paper released yet and also many controversies are happening about its name.

# YOLOv5 Inferencing using OpenVINO toolkit
Now we will see how YOLOv5 model inferencing can be done using Intel OpenVINO toolkit.

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

## YOLOv5 inferencing
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

1. After we generate the IR of YOLOv5 model, use the Python demo(yolo_openvino_demo.py script) for inferencing process of YOLOv5 model.
2. Download some images/videos and object classes for inferencing. Run the following commands one after the other-
```
wget -O face-demographics-walking.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4
wget -O bottle-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4
wget -O head-pose-face-detection-female.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4
wget https://github.com/bethusaisampath/YOLOv5_Openvino/blob/main/yolo_80classes.txt
wget https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/traffic1.jpg
wget https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/traffic2.jpg
wget https://raw.githubusercontent.com/bethusaisampath/YOLOs_OpenVINO/main/YOLOv5/yolo_openvino_demo.py
```
The inference Python script can be found at [yolo_openvino_demo.py](https://github.com/bethusaisampath/YOLOs_OpenVINO/blob/main/YOLOv5/yolo_openvino_demo.py)

3. Run the following commands for Inferencing-
```
python yolo_openvino_demo.py -i data/images/bus.jpg -m yolov5-v6.1/yolov5s.xml
```
![Demo_1](https://user-images.githubusercontent.com/37048080/180057938-5f60a73a-e0fc-432d-9841-a2e9bd9e5966.JPG)

```
python yolo_openvino_demo.py -i data/images/zidane.jpg -m yolov5-v6.1/yolov5s.xml
```
![Demo_2](https://user-images.githubusercontent.com/37048080/180057962-fc23db43-a90f-4500-a855-5b4e8514fce7.JPG)

```
python yolo_openvino_demo.py -i traffic2.jpg -m yolov5-v6.1/yolov5s.xml
```
![Demo_4](https://user-images.githubusercontent.com/37048080/180059994-e562f53f-6b2f-45b3-9c8f-3aa02f082c57.JPG)

```
python yolo_openvino_demo.py -i face-demographics-walking.mp4 -m yolov5-v6.1/yolov5s.xml
```
On the start-up, the application reads command-line parameters and loads a network to the Inference Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

https://user-images.githubusercontent.com/37048080/179828046-78eed3dc-00ed-456f-aa80-debe9a9965de.mp4


If you have Intel [GPU](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html#supported-devices) supported by OpenVINO, run the following command and then compare the inference times-
```
python yolo_openvino_demo.py -i traffic2.jpg -m yolov5-v6.1/yolov5s.xml -d GPU
```

# Performance
## [Benchmark Tool](https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html#benchmark-python-tool)
This tool estimates deep learning inference performance on supported devices. Performance can be measured for two inference modes: latency and throughput-oriented.
This tool is already installed when we download & install the OpenVINO Development tools via PyPI.
All the development tools are located here- C:\Users\xxxxx\openvino_env\Lib\site-packages\openvino\tools

Run the following command to run the Benchmark Python tool for estimating the YOLOv5s performance on test images-
```
benchmark_app -m yolov5-v6.1/yolov5s.xml -i data/images -d CPU -niter 100 -progress
```
![Benchmark_tool](https://user-images.githubusercontent.com/37048080/180296321-0da20329-b6c0-46f5-ae19-97a1d05d9796.JPG)


## [Accuracy Checker Tool](https://docs.openvino.ai/latest/omz_tools_accuracy_checker.html#deep-learning-accuracy-validation-framework)
The Accuracy Checker is an extensible, flexible and configurable Deep Learning accuracy validation framework. The tool has a modular structure and allows to reproduce validation pipeline and collect aggregated quality indicators for popular datasets both for networks in source frameworks and in the OpenVINO™ supported formats.
All the development tools are located here- C:\Users\xxxxx\openvino_env\Lib\site-packages\openvino\tools
