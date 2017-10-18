name: inverse
class: center, middle, inverse
layout: true
title: Python-basic

---
class: titlepage, no-number

# Python Basic
## .gray.author[Youngjae Yu]

### .x-small[https://github.com/yj-yu/python-basic]
### .x-small[https://yj-yu.github.io/python-basic]

.bottom.img-66[ ![](images/lablogo.png) ]


---
layout: false

## About

- Overview
- Python Basic - Python tutorial (book : Jump to python)
- Environment settings Anaconda 3
- Advanced tutorials
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

## Enjoy programming!

### "Life is too short, You need python." 

(인생은 너무 짧으니 파이썬이 필요해.)




---

## Be pythonic

The Official Python Tutorial
- https://docs.python.org/2/tutorial/

Wikibooks’ Non-Programmers Tutorial for Python
- https://en.wikibooks.org/wiki/Non-Programmer%27s_Tutorial_for_Python_3/Intro

Build dynamic web site (Nettuts+'s python, Django Book)
- https://code.tutsplus.com/series/python-from-scratch--net-20566
- http://www.djangobook.com/en/2.0/index.html

---
## Be pythonic

Make game with python (Invent with Python, Build a Python Bot)
- http://inventwithpython.com/chapters/
- https://code.tutsplus.com/tutorials/how-to-build-a-python-bot-that-can-play-web-games--active-11117

If you want to learn Computer Science, Algorithm from python
- Think Python: How to Think Like a Computer Scientist) http://www.greenteapress.com/thinkpython/thinkpython.html
- 번역글 : http://www.flowdas.com/thinkpython/

시간 있으실 때 무료 e-book 및 tutorial을 해보시기 바랍니다.
다른거 힘들게 배울 필요 없습니다.


---
## Be pythonic

Pycon KR에 참가도 해보세요
프로그래밍으로 상상한 거의 모든것이 가능합니다.

- https://www.pycon.kr/2017/program/schedule/


---

template: inverse

# Recent advance of deep learning applications
[예를 들면](https://github.com/devsisters/multi-speaker-tacotron-tensorflow)
[demo](http://carpedm20.github.io/tacotron/en.html)

---

template: inverse
# 앞으로 저희가 나눌 내용은..
Python과 

Machine Learning, 

Deep Learning,

저희가 제일 잘 할 수 있는 분야도 소개해 드리고 싶습니다.

---

template: inverse
# 딥 러닝, Machine learning? 맛보기
가장 빠른 방법을 추구,
코딩 고수가 되지 않아도 됩니다.
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

## Make your own work!

1. 해결하고자 하는 문제와 가장 닮은 데이터와 이미 구현된 모델을 찾습니다. 가급적이면 google 직원들이 짠 코드로!
2. 우리가 배운 python, (tensorflow) 지식으로 살짝 바꿔봅시다.
3. 이것 저것 실험해 봅니다. 빵빵한 GPU가 있다면 금상 첨화입니다.
4. 좋은 결과가 나올 때까지 여러 모델을 실험해 봅니다.

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

First, Jump to python!

Do it!
---

name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
