name: inverse
class: center, middle, inverse
layout: true
title: Python-basic

---
class: titlepage, no-number

# Python Basic
## .gray.author[Youngjae Yu]

### .x-small[https://github.com/yj-yu/tensorflow-basic]
### .x-small[https://yj-yu.github.io/tensorflow-basic]

.bottom.img-66[ ![](images/lablogo.png) ]


---
layout: false

## About

- Overview
- Python Basic - Python tutorial (book : Jump to python) 
- Environment settings (Anaconda 3) and GPU (optional)
- Advanced tutorial
---

template: inverse

# Recent advance of deep learning applications
4차 산업혁명시대의 자원개발을 중심으로

---

template: inverse
# 앞으로 저희가 나눌 내용은..

---

template: inverse
# 딥 러닝 구현을 위한
가장 빠른 방법을 추구,
코딩 고수가 되지 않아도 됩니다.
---
## Industry 4.0


.bottom.center.img-50[ ![](images/industry-4-0.png) ]

---
## With the advent of Deep Learning..

.top.center.img[ ![](images/deep_history.png) ]
.bottom.center.img[ ![](images/nn.png) ]

---
## Deep Learning makes Latent Representation

.bottom.center.img[ ![](images/1--N5_FtLQFRCcpN2ykq0idQ.png) ]

---
## Deep Learning makes Latent Representation

.bottom.center.img[ ![](images/1--N5_FtLQFRCcpN2ykq0idQ.png) ]

### 중간 단계의 representation을 data로부터 생성

---
## Deep learning application


.bottom.center.img[ ![](images/3d.png) ]

### Deep Generative model

.bottom.center.img-70[ ![](images/3d2.png) ]


---

## Feature generation and detection


.bottom.center.img[ ![](images/3d3.png) ]

---

## Recognize 3D environment - PointNet


.bottom.center.img[ ![](images/pointnet.jpg) ]

---

## Recognize 3D environment - PointNet

.bottom.center.img[ ![](images/semantic.jpg) ]


---

## Another applications

.bottom.center.img[ ![](images/satelite.png) ]


---

template: inverse
## Python is
Platform 
independent
interpreter


---

template: inverse
## Python is enough
70 % of Google code is made of python

얼마든지 효율적이고 빠른 코드를 작성할 수 있습니다.

---

template: inverse
## Python is easy
가독성이 좋습니다. 

코드 리딩 훈련을 파이썬으로 하면, 

최신 work을 빠르게 follow up 할 수 있습니다

---

## Example - Objective

.bottom.center.img[ ![](images/challenge.png) ]

---

## Example - Find open source

.bottom.center.img[ ![](images/challenge2.png) ]

git에는 필요한 거의 모든 모델과 data processing code가 있습니다.

---

## Example - git clone ~~
.bottom.center.img[ ![](images/challenge3.png) ]

git clone ~~

---

## Example - Make your graph!

.bottom.center.img-70[ ![](images/challenge4.png) ]

tensorflow 이틀 배우면 할 수 있습니다.


---

## Make your own work!

1. 해결하고자 하는 문제와 가장 닮은 데이터와 그래프를 찾습니다. 가급적이면 google 직원들이 짠 코드로!
2. 우리가 배운 python, tensorflow 지식으로 살짝 바꿔봅시다.
3. 이것 저것 실험해 봅니다. 빵빵한 GPU가 있다면 금상 첨화입니다.
4. 좋은 결과가 나올 때까지 여러 graph를 실험해 봅니다.

---

template: inverse
## Let's start python!


---


## Environment setting


## Install anaconda

Install instruction for windows OS
https://www.tensorflow.org/install/install_windows

Anaconda에 있는 배포판에 numpy 등 기본 라이브러리를 기본적으로 포함

아래 링크에서 Python 3.x 버전으로 설치를 해주세요.
https://www.continuum.io/downloads#windows

.bottom.center.img-50[ ![](images/anaconda.png) ]

---
## Install anaconda

Windows 키를 누른뒤에 anaconda prompt 입력하면 console이 뜹니다.

conda 라는 명령어로 여러 개의 가상환경을 만들 수 있습니다.

```python
conda create --name tf python=3.6
```


```python
#tf라는 환경이 만들어졌는지 확인
conda info --envs
```
---
## Install anaconda

자 이제 기본 실습환경 세팅을 위해 tf라는 가상 환경으로 들어갑니다.

```python
activate tf
#만약 비활성화하고 싶다면 deactivate tf를 치세요.

#기본 라이브러리 확인
conda list

```

pip가 보이시죠? 이제 pip를 통해 tensorflow 및 실습 환경을 위한 라이브러리를 추가할 것입니다.
다음 명령어들을 입력하여 자동으로 tensorflow 최신 배포판을 설치합니다.

```python
pip install ipython
pip install jupyter
pip install tensorflow
```



---
## (Optional) Configuration - CUDA, graphic driver

우분투 설치를 마친 직후 부팅해보면 운영체제에서 그래픽카드를 아직 인식하지 못한 상태이기 때문에 해상도가 매우 낮을 수 있습니다. 이 때, 그래픽 드라이버를 설치하면 고해상도가 됩니다.

NVIDIA 그래픽 드라이버를 배포하는 PPA를 설치하고 업데이트를 합니다. (367.4x 버전 이상의 최신 버전이어야 함)

```bash
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
$ sudo apt-get install nvidia-375
```
설치가 끝나면 재부팅합니다.
```bash
$ sudo reboot
```

---
## Configuration - CUDA, graphic driver

재부팅 후 고해상도 화면이 나오면 성공이라고 생각하면 됩니다. 
터미널에 nvidia-smi를 입력하면 아래와 같이 드라이버 버전과 시스템에 인식된 GPU를 확인할 수 있습니다.
```bash
$ nvidia-smi
Mon Mar  6 01:01:51 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.39                 Driver Version: 375.39                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 970     Off  | 0000:05:00.0      On |                  N/A |
|  0%   29C    P8    12W / 180W |    292MiB /  4034MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1128    G   /usr/lib/xorg/Xorg                             169MiB |
|    0      1887    G   compiz                                         121MiB |
+-----------------------------------------------------------------------------+
```
---
## Configuration - CUDA, graphic driver

만약 그래픽 드라이버 설치 도중 바이오스 화면이 뜨거나, 
설치를 완료하고 부팅했는데 무한 로그인 loop에 빠진다면 
바이오스 설정에서 secure boot 옵션을 disabled 상태로 바꿔주세요.


---
## Configuration - CUDA Toolkit 8.0 설치

공식 다운로드 페이지
https://developer.nvidia.com/cuda-downloads
에서 우분투 16.04의 runfile(local)을 다운로드한다. 모두 받았다면 아래와 같이 실행합니다.
```bash
$ sudo sh cuda_8.0.61_375.26_linux.run
```

장문의 라이센스 문구가 나오는데, 
Enter를 입력하며 넘기기 귀찮다면 Ctrl+C를 입력. 
한 번에 아래 질문으로 넘어갑니다. 이후의 질문에 아래와 같이 답하세요.

```bash
Do you accept the previously read EULA?
accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Toolkit?  
(y)es/(n)o/(q)uit: y
```
---
## Configuration - CUDA Toolkit 8.0 설치

```bash
Enter Toolkit Location  
 [ default is /usr/local/cuda-8.0 ]: 

Do you want to install a symbolic link at /usr/local/cuda?  
(y)es/(n)o/(q)uit: y

Install the CUDA 8.0 Samples?  
(y)es/(n)o/(q)uit: n

Enter CUDA Samples Location  
 [ default is /home/your_id ]: 

```

---
## Configuration - CUDA Toolkit 8.0 설치

설치를 마친 뒤 환경변수 설정을 합니다. 터미널에 아래와 같이 입력합시다.

```bash
$ echo -e "\n## CUDA and cuDNN paths"  >> ~/.bashrc
$ echo 'export PATH=/usr/local/cuda-8.0/bin:${PATH}' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
```
위와 같이 실행하면 ~/.bashrc에 마지막 부분에 아래 내용이 추가됩니다.

```bash
## CUDA and cuDNN paths 
export PATH = /usr/local/cuda-8.0/bin : $ { PATH } 
export LD_LIBRARY_PATH = /usr/local/cuda-8.0/lib64 : $ { LD_LIBRARY_PATH }
```

---
## Configuration - CUDA Toolkit 8.0 설치

변경된 환경변수를 적용하고 cuda 설치여부를 확인합시다.

```bash
$ source ~/.bashrc
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```

다음 단계로 넘어가기 전에 cuda가 어느 위치에 설치되어 있는지 확인하고 넘어갑시다. 
CuDNN 파일을 붙여넣을 경로를 보여주므로 중요합니다. 
기본으로 /usr/local/cuda/인 경우가 많은데, 기본적으로 /usr/local/cuda-8.0/ 입니다.

```bash
$ which nvcc
/usr/local/cuda-8.0/bin/nvcc
```

---
## Configuration -CuDNN v5.1 설치

https://developer.nvidia.com/rdp/cudnn-download


에서 CuDNN을 다운로드 (회원가입이 필요). 
여러 파일 목록 중 cuDNN v5.1 Library for Linux(파일명: cudnn-8.0-linux-x64-v5.1.tgz)를 받습니다.

아래와 같이 압축을 풀고 그 안의 파일을 cuda 폴더(주의: which nvcc 출력값 확인)에 붙여넣고 권한설정을 합니다. which nvcc 실행 결과 cuda 폴더가 /usr/local/cuda-8.0이 아니라 /usr/local/cuda일 수도 있으니 꼼꼼히 확인합시다.

```bash
$ tar xzvf cudnn-8.0-linux-x64-v5.1.tgz
$ which nvcc
/usr/local/cuda-8.0/bin/nvcc
$ sudo cp cuda/lib64/* /usr/local/cuda-8.0/lib64/
$ sudo cp cuda/include/* /usr/local/cuda-8.0/include/
$ sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*
$ sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h
```

---
## Configuration -CuDNN v5.1 설치

아래와 같은 명령어를 입력하여 비슷한 출력값이 나오면 설치 성공입니다.

```bash
$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2  
#define CUDNN_MAJOR      5
#define CUDNN_MINOR      1
#define CUDNN_PATCHLEVEL 10
--
#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#include "driver_types.h"
```

---
## Configuration - NVIDIA CUDA Profiler Tools Interface 설치

NVIDIA CUDA Profiler Tools Interface를 터미널에 아래와 같이 입력하여 설치합니다.
공식 문서에서 필요하다고 하니 설치합시다.

```bash
sudo apt-get install libcupti-dev
```

---
## Configuration


실습에 앞서
pip를 통해 tensorflow 및 실습 환경을 위한 라이브러리를 추가합니다.
다음 명령어들을 입력하여 자동으로 tensorflow 최신 배포판을 설치합니다. 

```python
sudo pip install tensorflow-gpu
# or CPU version
sudo pip install tensorflow
pip install jupyter
pip install matplotlib
...

```


---

template: inverse

# Python Basic


---
## Install configuration

```python
git clone https://github.com/yj-yu/python-basic.git
cd python-basic
ls
```

code(https://github.com/yj-yu/python-basic)

```bash
./code
├── Jump-to-python
├── Python-Lectures
├── PythonZeroToAll
└── cs228-material

```

- Jump-to-python : https://github.com/LegendaryKim/Jump-to-Python
- Python-Lectures : https://github.com/rajathkmp/Python-Lectures
- PythonZeroToAll : https://github.com/hunkim/PythonZeroToAll
- cs228-material : https://github.com/kuleshov/cs228-material.git
  한글 버전 (AI korea) http://aikorea.org/cs231n/python-numpy-tutorial/
---
```python
jupyter notebook

```

Do it!
---

name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
  <b>Special Thanks to</b>: 변훈 연구원님, 송재준 교수님</p>
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
