# Jetson Nano的YOLO配置教程

作者：姚雪健

日期：2021.11.11

[TOC]

# ==**！！！注意版本对应！！！**==

本教程所用的Nano系统版本为4.5.1，TensorRT版本7.x，Deepstream版本5.1，torch版本1.9.0，torchvision版本0.10.0，yolov5版本 5.0

TensorRT在Nano系统中预装且好像无法更改版本，系统版本看JetPack，注意与Deepstream对应，否则无法成功运行

 ==相关github库源代码、whl、权重等文件U盘中均有，可本地拷贝至Nano==

torch和torchvision版本也需要对应，具体见相应章节

[torch和torchvision](# 4. YOLOV5S环境配置)

[Deepstream和JetPack](# 6. Deepstream加速)

ctrl+左键点击跳转



# 1. 系统烧录与安装



## 1.1 镜像下载

官方教程：[Getting Started With Jetson Nano Developer Kit | NVIDIA Developer](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup)

Jetson nano官网镜像下载地址: [Jetson Download Center | NVIDIA Developer](https://developer.nvidia.com/embedded/downloads)

SD卡烧录软件Etcher下载地址：[balenaEtcher - Flash OS images to SD cards & USB drives](https://www.balena.io/etcher/)

下载完官方镜像后使用Etcher进行烧录，可跳过校验过程，插入Nano核心模块下的SD卡槽，接上显示器通电开机



## 1.2 系统安装

安装系统时为避免后续不必要的麻烦，建议选择语言为英文，键盘布局为美式，时区选择上海，APP Partition Size默认，Select Nvpmodel Mode选择MAXN

==账号和密码请务必记住==





# 2. Nano基本环境配置



## 2.1 apt换源

### 	2.1.1 备份原始source.list

```shell
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
```

### 2.1.2 修改source.list

```shell
sudo vim /etc/apt/sources.list
```

vim删除或备注所有内容，添加清华大学源（aarch64版）

```shell
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
```

### 2.1.3 更新软件源

```shell
sudo apt-get update
```

### 2.1.4 更新软件（选做）

```shell
sudo apt-get upgrade
```



## 2.2 安装pip3并换源

### 2.2.1 pip3安装并检查最新

```shell
sudo apt-get install python3-pip
pip3 install --upgrade pip
```

### 2.2.2 pip3换源

```shell
mkdir ~/.pip
vim ~/.pip/pip.conf
```

vim写入清华pip3源

```shell
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```



## 2.3 Python基本库安装

### 2.3.1 pillow

```shell
pip3 install pillow
```

### 2.3.2 Cython

```shell
pip3 install Cython
```

### 2.3.3 matplotlib（重点）

安装下述包，特别是第一个==libfreetype6-dev==，其他可能系统已经自带了。如果未安装libfreetype6-dev，matplotlib将会安装报错

```shell
sudo apt-get install libfreetype6-dev
sudo apt-get install pkg-config
sudo apt-get install libpng12-dev
sudo apt-get install pkg-config
```

安装matplotlib3.2.2（YOLOV5最低要求版本也是对Nanp兼容性最好的一个版本）

```shell
pip3 install matplotlib==3.2.2
```

### 2.3.4 numpy（后期出现报错首先考虑版本问题）

numpy目前最新为1.19.5

```shell
pip3 install numpy
```

默认安装会自动安装最新版本。但是YOLOV5最低版本要求为1.18.5。

在后续tensorrtx模型转译过程中，新版本可能会报==一行==什么core错误这种代码，那就是numpy版本过高，需要安装numpy==1.18.5

```shell
pip3 install numpy==1.18.5
```

安装过程需要转译，时间较长，耐心等待



## 2.4 交换分区内存设置

后期编译torchvision，生成TensorRT推理引擎文件不够用，这里通过增加swap内存的方法来解决这个问题

```shell
sudo vim /etc/systemd/nvzramconfig.sh
```

修改

```shell
mem=$((("${totalmem}" / 2 / "${NRDEVICES}") * 1024))
```

为

```shell
mem=$((("${totalmem}" * 2 / "${NRDEVICES}") * 1024))
```

即修改==/==为==*==

重启系统

```shell
reboot
```



## 2.5 CUDA添加系统PATH

```shell
vim .bashrc
```

在文档最后添加下述命令
```shell
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

重启或等待一段时间后输入

```shell
nvcc -V
```

如果显示CUDA版本信息则配置PATH成功





# 3. YOLOV4-TINY环境配置



## 3.1 Clone源码

下载YOLOV4源码

```shell
git clone https://github.com/AlexeyAB/darknet.git
```

有时可能会出现无法连接库

可将https改为git

```shell
git clone git://github.com/AlexeyAB/darknet.git
```

也可直接PC上clone之后复制文件夹到Nano



## 3.2 编译源码并导入权重

### 3.2.1 编译源码

在darknet文件夹右键opem in terminal或者cd darknet，修改Makefile文件中下述行，讲0改为1，即启用功能

```shell
GPU=1
CUDNN=1
OPENCV=1
LIBSO=1
```

然后make编译

```shell
make -j
```

编译结束后输入

```shell
./darknet
```

如果输出

```shell
usage: ./darknet <funcyion>
```

则编译成功

### 3.2.2 导入权重

```shell
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

有时网络不稳或者提示库不存在，可从YOLOV5 github库寻找权重，也可直接U盘复制

YOLOV5 github地址：[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)



## 3.3 调用CSI摄像头

在YOLOV4文件夹下运行

```shell
./darknet detector demo cfg/coco.data cfg/yolov4-tiny.cfg yolov4-tiny.weights "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

其中flip-method为输出画面方面（0-4），此处默认正向，可根据需要自行调整

为了快速启动，可在/home下创建一个快速启动文件v4.sh

```shell
vim v4.sh
```

输入

```shell
cd darknet
./darknet detector demo cfg/coco.data cfg/yolov4-tiny.cfg yolov4-tiny.weights "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

退出后即完成.sh文件创建

运行时只需ctrl+alt+t打开terminal，默认打开文件为/home输入

```shell
sh v4.sh
```

即可实现快速运行YOLOV4-TINY的CSI检测





# 4. YOLOV5S环境配置



## 4.1 torch安装

### 4.1.1 whl文件下载

官方教程：[PyTorch for Jetson - version 1.10 now available - Jetson & Embedded Systems / Jetson Nano - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)

本教程版本为Pytorch v1.9.0，U盘中包含该whl，具体下载地址为https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl

各版本Pytorch具体地址见官方文档

### 4.1.2 依赖包安装

```shell
sudo apt-get install libopenblas-base libopenmpi-dev
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```

### 4.1.3 torch安装

将下载好的torch的whl安装包放在/home下，呼出terminal，输入

```shell
pip3 install torch-1.x.0-cp36-cp36m-linux_aarch64.whl  # where 0.x.0 is the torch version
```

代码中install的名称为该whl的文件名，基本就改1.x.0即可



## 4.2 torchvision安装

### 4.2.1 torchvision下载

版本对应表

```text
PyTorch v1.0 - torchvision v0.2.2
PyTorch v1.1 - torchvision v0.3.0
PyTorch v1.2 - torchvision v0.4.0
PyTorch v1.3 - torchvision v0.4.2
PyTorch v1.4 - torchvision v0.5.0
PyTorch v1.5 - torchvision v0.6.0
PyTorch v1.6 - torchvision v0.7.0
PyTorch v1.7 - torchvision v0.8.1
PyTorch v1.8 - torchvision v0.9.0
PyTorch v1.9 - torchvision v0.10.0
PyTorch v1.10 - torchvision v0.11.1
```

下载对应torchvision

```shell
git clone --branch v0.x.0 https://github.com/pytorch/vision torchvision  # where 0.x.0 is the torchvision version
```

修改branch后为对应torchvision版本号，即v0.x.0

### 4.2.2 配置文件

进入torchvision文件夹并修改配置文件

```shell
cd torchvision
export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version 
```

修改=后为对应torchvision版本号，即0.x.0

### 4.2.3 编译

```shell
python3 setup.py install --user
```

时间较长，请耐心等待

编译完成后显示

```shell
Finished processing dependencies for torchvision==0.x.0  # where 0.x.0 is the torchvision version 
```



## 4.3 YOLOV5安装

### 4.3.1 YOLOV5下载

```shell
git clone https://github.com/ultralytics/yolov5.git
```

有时可能会出现无法连接库

可将https改为git

也可直接PC上clone之后复制文件夹到Nano

### 4.3.2 YOLOV5依赖库安装

```shell
cd yolov5
pip3 install -r requirements.txt
```

==注意==

运行完该命令后，numpy会被升级到最新版本以运行yolov5，但这会导致后面tensorrtx转译时报错，因此后期还要将numpy卸载重新下载==1.18.5==版本

（==现在不需要，但需要记得这件事==)



### 4.4 权重下载

完成上述步骤后运行yolov5

```shell
python3 detect.py
```

此时会自动下载yolov5s.pt权重，但是可能会出现需要安装字体库Arial.ttf的情况，且安装会一直卡住

这时可直接点击terminal上网址前往下载ttf也可直接复制U盘中ttf文件

之后在/home下创建./nano/ultralytics文件夹，将ttf放入其中

```shell
mkdir /home/nano/ultralytics
```

之后再次运行detect.py，即可下载yolov5s.pt。

若速度过慢则可以直接复制U盘中文件至yolov5

运行完成推理



# 5. TensorRT加速



## 5.1 Tensorrtx下载

```shell
git clone https://github.com/wang-xinyu/tensorrtx.git
```



## 5.2 .pt模型转译

官方转译文档：[tensorrtx/README.md at master · wang-xinyu/tensorrtx (github.com)](https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/README.md)

进入/home/tensorrtx/yolov5文件夹复制gen_wts.py至/home/yolov5，并运行该py，即可得到yolov5s.wts文件

```shell
cd tensorrtx
cp yolov5/gen_wts.py ~/yolov5
cd ~/yolov5
python3 gen_wts.py -w yolov5s.pt -o yolov5s.wts
```

==如果报错一行包括core什么的，就是前面所说的numpy版本过高，需要重新安装numpy=1.18.5==

==如果使用自己训练的权重，将-w和-o后名称进行修改，也可将自训练权重改为yolov5.pt，以免出现不必要的麻烦==



## 5.3 .engine文件和.so动态库编译

回到/home/tensorrtx/yolov5文件夹，在其下新建build文件夹并将生成的.wts文件移动至此，然后cmake生成makeFile

```shell
cd ~/tensorrtx/yolov5
mkdir build && cd build
mv ~/yolov5/yolov5s.wts ./
cmake ..
```

==若使用自训练权重，需要将yololayer.h里的CLASS_NUM修改成自己的。因为官方用的是coco数据集，所以默认是80。==

然后执行makefile，==每次修改为CLASS_NUM都要make一次==

```shell
make -j
```

生成.engine和.so文件

```shell
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
```

在../samples下放入几张照片进行测试

```shell
sudo ./yolov5 -d yolov5s.engine ../samples
```

输出结果在build文件夹





# 6. Deepstream加速

## 6.1 Deepstream下载

官方下载地址（注意版本对应）：[DeepStream Getting Started | NVIDIA Developer](https://developer.nvidia.com/deepstream-getting-started)

Deepstream5.1下载地址：https://developer.nvidia.com/deepstream-sdk-v510-jetsontbz2

好像今天刚更新了6.0版，与JetPack4.6对应

==下载.tar格式安装包==



## 6.2 Deepstream安装

官方安装指南：[Quickstart Guide — DeepStream 6.0 Release documentation (nvidia.com)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#jetson-setup)

### 6.2.1 安装依赖环境

```shell
sudo apt install \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1
```

### 6.2.2 安装tar压缩包

将下载好的deepstream_sdk_v5.1.0_jetson.tbz2放在/home下，输入

```shell
sudo tar -xvf deepstream_sdk_v5.1.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.0
sudo ./install.sh
sudo ldconfig
```

不同的版本对应修改相应代码即可

安装完成后输入

```shell
deepstream-app --version-all
```

如果显示deeptream信息则安装成功

### 6.2.3 运行官方案例

进入官方案例文件夹

```shell
cd /opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app/
```

运行

```shell
deepstream-app -c source8_1080p_dec_infer-resnet_tracker_tiled_display_fp16_nano.txt
```

出现多开视频



## 6.3 YOLOV5的Deepstream加速

### 6.3.1 库下载

下载开源大佬的库

```shell
git clone https://github.com/DanaHan/Yolov5-in-Deepstream-5.0.git
```

### 6.3.2 编译

==若使用自训练权重==

则进入Yolov5-in-Deepstream-5.0/Deepstream 5.0/nvdsinfer_custom_impl_Yolo修改nvdsparsebbox_Yolo.cpp文件中的类型数量

```shell
cd nvdsinfer_custom_impl_Yolo
vim nvdsparsebbox_Yolo.cpp
```

将nvdsparsebbox_Yolo.cpp中的

```shell
static const int NUM_CLASS_YOLO = 80;
```

修改为自训练权重中的标签数

```shell
static const int NUM_CLASS_YOLO = x;  # where x is your number 
```

==若使用yolov5s.pt则不需要进行上述步骤==

在nvdsinfer_custom_impl_Yolo目录下进行make编译

```shell
make -j
```

编译成功后将之前tensorrtx生成的yolov5s.engine文件和libmyplugins.so放到Yolov5-in-Deepstream-5.0/Deepstream 5.0

```shell
cp ~/tensorrtx/yolov5/build/yolov5s.engine ./
cp ~/tensorrtx/yolov5/build/libmyplugins.so ./
```

若为自训练权重则与自己名称对应

### 6.3.3 修改配置文件

#### 6.3.3.1 labels.txt

将YOLOV4中的coco数据集的标签复制到该文件夹下

```shell
cd Yolov5-in-Deepstream-5.0/Deepstream 5.0
cp ~/darknet/data/coco.names ./labels.txt
```

==若使用自训练权重，则在Yolov5-in-Deepstream-5.0/Deepstream 5.0下新建labels.txt==

```shell
vim labels.txt
```

==输入训练的标签，一行一个==，如

```shell
person
dog
cat
```

#### 6.3.3.2 deepstream_app_config_yoloV5.txt

返回Yolov5-in-Deepstream-5.0/Deepstream 5.0目录

```shell
cd ..
```

修改deepstream_app_config_yoloV5.txt的[source 0]，[primary-gie]和[tracker]下部分代码

```shell
vim deepstream_app_config_yoloV5.txt
```

==[source 0]：第24行==

```shell
uri=file:/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_1080p_h264.mp4
```

修改deepstream的版本号于自己对应

==[primary-gie]：第84，85行==

```shell
model-engine-file=yolov5s.engine
labelfile-path=labels.txt
```

修改.engine和.txt文件与自己对应

==[tracker]：第101行==

```shell
ll-lib-file=/opt/nvidia/deepstream/deepstream-5.0/lib/libnvds_mot_klt.so
```

修改deepstream的版本号于自己对应

#### 6.3.3.3 config_infer_primary_yoloV5.txt

修改config_infer_primary_yoloV5.txt的[property]下部分代码

==[property]：==

==第45，46行==

```shell
model-engine-file=yolov5s.engine
labelfile-path=labels.txt
```

修改.engine和.txt文件与自己对应

==第50行==

```shell
num-detected-classes=80
```

修改识别类型数量与自训权重对应

==启用第60行，备注第59行==

```shell
#custom-lib-path=objectDetector_Yolo_V5/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

### 6.3.4 测试

在Yolov5-in-Deepstream-5.0/Deepstream 5.0目录下运行

```shell
LD_PRELOAD=./libmyplugins.so deepstream-app -c deepstream_app_config_yoloV5.txt
```



## 6.4 使用CSI摄像头检测

修改deepstream_app_config_yoloV5.txt

将[source0]下的

```shell
[source0]
enable=1
```

修改为

```shell
[source0]
enable=0
```

在[source0]下添加[source1]并启用，即调用CSI摄像头

```shell
[source1]
enable=1
#Type - 1=CameraV4L2 2=URI 3=MultiURI 4=RTSP 5=CSI
type=5
camrea-width=1280
camrea-height=720
camrea-fps-n=30
camrea-fps-d=1
```

在Yolov5-in-Deepstream-5.0/Deepstream 5.0目录下运行

```shell
LD_PRELOAD=./libmyplugins.so deepstream-app -c deepstream_app_config_yoloV5.txt
```

即可实现CSI摄像头检测
