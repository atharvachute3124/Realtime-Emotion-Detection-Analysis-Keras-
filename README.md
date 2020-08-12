# Realtime-Facial-Emotion-Detection-Analysis-Keras

## Introduction
One of the most researched topics in the modern-day machine learning arena is detecting emotions. The ability to accurately detect and identify an emotion opens up numerous doors for Advanced Human-Computer Interaction. Detection of emotions can be done through human speech, body posture and facial expression. The aim of this project is to detect up to seven distinct facial emotions in real time. This project runs on top of a Convolutional Neural Network (CNN) that is built with the help of Keras whose backend is TensorFlow in Python.

## Overview
* This model is trained with Convolutional Neural Network built with Keras using the Tensorflow backend.
* It uses OpenCV for Image Processing tasks where we identify a face with live webcam feed and feed it into the Neural network for Emotion Detection.
* The Objective of our project is to examine the Facial emotions in real time using a CNN.

## Dataset 
* The dataset used here is Fer2013 that consists of 32,298 images which is further categorized into training (28,709) and testing (3,589) images respectively.
* This dataset(~143MB zip) can be downloaded from [here.](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## Dependencies:
* pip install tensorflow
* pip install keras
* pip install opencv-python

## How to Execute
1. Run ED_execution.py
2. Run videoTester.py for realtime emotion detection </br>
   <b>OR</b> </br>
3. Run imageTester.py for emotion detection on individual images

## Acknowlegdements
* [Haar Cascade Object Detection Face & Eye(sentdex)](https://www.youtube.com/watch?v=88HdqNDQsEk&feature=youtu.be)
* [How to do Facial Emotion Recognition Using a CNN](https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f)
