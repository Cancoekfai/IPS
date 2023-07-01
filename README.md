# IPS
Code to run Offline Signature Verification from our paper [A Scaling Factor Based Image Processing Strategy for Object Detection](https://www.warse.org/IJATCSE/static/pdf/file/ijatcse011232023.pdf)


# Data Preparation
Download [2021 Zhuhai Open Data Innovation Apps Contest Dataset](https://drive.google.com/file/d/1jH-ZfzvupFgkmzhNiqDNxasZV033Vwls/view?usp=drive_link) to a folder named data.


# Training
```shell
cd code/yolov5-master
python train.py
```

# Evaluation
```shell
cd code/yolov5-master
python test.py
```