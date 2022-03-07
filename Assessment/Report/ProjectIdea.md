# Project Idea

## Title: Flower Image Claasification

---

## Idea: 

**I want my project to be able to classify images of flowers using the camera on my laptop**


## Data:
It loads data using tf.keras.utils.image_dataset_from_directory(https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory) and capture images data from my laptop.



## Method Selection

For classifying images, convolutional neural network(CNN) has been used in our lecture to recognize dog and cat. However, modern convolutional neural networks have millions of parameters. Training them from scratch requires a lot of labeled training data and a lot of computing power (hundreds of GPU-hours or more). We only have about three thousand labeled photos and want to spend much less time, so we need to be more clever.
