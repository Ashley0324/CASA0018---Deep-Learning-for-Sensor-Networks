# 🌸Flower Images Classifiction With Transfor Learning

## ❓Definition of problem being solved 
I want my project to be able to classify images of flowers using the camera on my laptop and iPhone📱.

### Project Overview

This model is designed to classify different kinds of flowers. The inspiration for this project is when I move to London, I buy flowers every week, but some flowers I have never seen is really attractive and I wondered what kind of flower it is. So, I want to find a way to identify these beautiful flowers.

We learned how to recognise cat and dog in lecture part by training a CNN, but I found another tutorial to classify Flowers with transfer Learning. It brought back memories of our leture about transfer learning: it uses less time and is more clever by extracting useful features already learnt from millions of pictures. Higher accuracy and faster convergence are also advantages of this method.((Duncan, CASA0018_06))

I have a better choice, why not?

### 🚩Research Question
- How to classify image with transfer learning.
- How to Visualize the rea-time prediction. 
- How to improve final accuracy.

### 📊Data Available

I download the flower dataset provided by TensorFlow officail example(http://download.tensorflow.org/example_images/flower_photos.tgz). It has 2904 images with 5 labels(Daisy, Dandelions, sunflowers, rose, tulips). I split these as training data(80%) and test data(20%). Training data was used to train the model, while test data was used to evaluate how well the model perform on the new data.
Daisy|Dandelions|Sunflowers|Rose|Tulips
--|:--:|--:|--:|--:
493|701|560|509|641

Here are some labled images:
![image](https://user-images.githubusercontent.com/99146042/159248053-c2df3e4f-96f9-4fb8-973e-c7ffd2b7bbb1.png)
(https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model)

### 🧮Outcomes Anticipated

Improve the accuracy above 80%. (While the tutorial accuracy is 78% with CNN)

### 📱Application Design

I'd like to capture picture data from my laptop. So I use Edge Empulse to create the application.
Here is the link to my edge impulse page:https://studio.edgeimpulse.com/studio/87595/

And you can scan this QR code to enjoy the mobile application now!
<img width="464" alt="Screenshot 2022-03-20 at 13 10 31" src="https://user-images.githubusercontent.com/99146042/159163886-6d528306-15ce-4e3f-a3c5-67ff353b0603.png">


## ✨Documentation of experiments and results 
(model training results, description of training runs, model architecture choices, visual record of experiments) 
### 1. Define some parameters for the loader:
<img width="295" alt="Screenshot 2022-03-19 at 02 12 44" src="https://user-images.githubusercontent.com/99146042/159103828-4645091d-7100-40c1-8b9a-f582c03be10f.png">


### 2. Neural Network settings

<img width="588" alt="Screenshot 2022-03-19 at 02 24 22" src="https://user-images.githubusercontent.com/99146042/159103172-0b6f89c9-c527-47fc-bb08-615604bbc883.png">

### 3. Model Architecture Choices

I chose two maethods to classify flower images: Convolutional Neural Network(CNN) and transfer learning. To explore how they work and their performance. Finally, I used transfer learning to build my application for higher accuracy.

### 4. Model Training Results
I still remember the first model training result(accuracy:39.9%), it's a really stupid model. It even recognizes my face like a rose with 99% similarity. So I started the process to improve my model. This is my record of each test result.

<img width="900" alt="Screenshot 2022-03-20 at 13 35 06" src="https://user-images.githubusercontent.com/99146042/159164937-8e12c1bb-dd38-40ae-90fa-0d31d68156cd.png">

<img width="582" alt="Screenshot 2022-03-19 at 02 24 32" src="https://user-images.githubusercontent.com/99146042/159103308-89146791-136c-411d-ae30-f240fcf30d0a.png">


### 📝Critical reflection and learning from experiments 
We can conclude the factors influcing results from the results: Image Size,Color depth,Training cycles,and Data augmentation.

Actually, th elearning rate will impact the loss value.  Choosing the learning rate is challenging as a value too small may result in a long training process that could get stuck, whereas a value too large may result in learning a sub-optimal set of weights too fast or an unstable training process. According to the Leslie(2015), I find the best learning rate in this model is 0.0005.

In conclusion, I think the transfer learning model works better than the CNN model in flower images classification. Even it cost more time. But it deserves it, isn't it?


## 💡Improvement
- When I tried to **add the training cycles from 11 to 12 or higher**, the system always fail to do it. I still can't find the reason. I'll work it in the future.
<img width="495" alt="Screenshot 2022-03-21 at 12 05 58" src="https://user-images.githubusercontent.com/99146042/159273030-8fe6fee3-e113-4fbd-866a-d32342c9bf61.png">

- **Try a new model called VGG16**, according to Emine and Ahmat(2019), With transfer learning, frequently used pretrained deep learning models such as Alexnet, Googlenet, VGG16, DenseNet and ResNet are used for image classification. The results show that the models used achieve acceptable performance rates while the highest performance is achieved with the VGG16 model.

- Explore how the numbers of nuerons impacts the performance.

## 📘References:
- Duncan, W. (2022). CASA0018_06. London: UCL, 7-9. https://moodle.ucl.ac.uk/pluginfile.php/4514917/mod_resource/content/1/CASA0018_06.pdf
- Goodfellow, I., Bengio, Y. & Courville, A., 2016. Deep learning / Ian Goodfellow, Yoshua Bengio and Aaron Courville.
- https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
- https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model


