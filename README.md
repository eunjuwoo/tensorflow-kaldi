# tensorflow-kaldi Toolkit
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"/>

The tensorflow-kaldi implements a neural network based Solid Voice Trigger system using Kaldi and Tensroflow. The main idea is that kaldi can be used to extract feature while Tensorflow is a better choice to build the neural network.    
In the future, this toolkit can be useful to make flexible Trigger System or EPD(End Point Detection) System.


## Requirement
* Python version > 3.x (Recommend Anaconda)
* [Kaldi](https://kaldi-asr.org, "Kaldi link") version > 5.5
* [Tensorflow](https://tensorflow.org, "Tensorflow link") version > 2.x 

## Dataset
* [SNIPS](https://github.com/snipsco/keyword-spotting-research-datasets, "Download the SNIPS dataset") dataset
  : The wake word is "Hey Snips" pronounced with no pause between the two words. It is provided by Snips, Paris, France (https://snips.ai)
* [TEDLIUM](http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz, "Download the TEDLIUM dataset") dataset
  : English speech data

## Methodology
### Extract feature and Align.     
 Run the Kaldi s5_r3 baseline of TEDLIUM. This step is necessary to compute features and labels later used to train the Tensorflow neural network. We recommend running the tedlium s5 recipe (not including the DNN training):
  ```
  egs/tedlium/s5/run.sh
  ```
 We run the code below to use i-vector(100-th) and MFCC(40-th) as features and alignment(phones) as labels. 
  ```
  ./main.sh --stage 3
  ```
### Create TFRecords file. 
  ```
  data/mk_tfrecord.py --OPTION [train/test/validation]
  ```

### Train Neural Network
  ```
  train.py \
    --feat.train_tfr [str, train TFRecord file name] \
    --feat.test_tfr [str, test TFRecord file name] \
    --feat.validation_tfr [str, validation TFRecord file name] \
    --trainer.continue-learning [str, true(transfer learning) / false(flat start learning)] \
    --trainer.activation-function [str, activation function] \
    --trainer.batch-size [int, batch size] \
    --trainer.learning-rate [float, learning rate] \
    --trainer.epoch [int, epoch] \
    --dir [str, output directory name]
  ```
----------------------------------------------------------------------------------------------------------------------------
<!--
- markdown reference : https://gist.github.com/ihoneymon/652be052a0727ad59601
- create badge : [![태그이름](https://img.shields.io/badge/태그에 적히는 글씨-태그색?style=flat-square&logo=로고이름&logoColor=로고색)](관련된 내 링크)
               : <img src="https://img.shields.io/badge/[태그에적히는글씨]-[태그색]?style=flat-square&logo=[simpleicons에서찾은로고이름]&logoColor=white"/>
               : 상세 방법 참고 : https://soo-vely-dev.tistory.com/159
               : 아이콘 찾는 사이트 : https://simpleicons.org/?q=visual%20stu
-->
