# Classify Flowers based on 3 models and anlysis
Qian Jin
- Repo link:https://github.com/Ashley0324/CASA0018---Deep-Learning-for-Sensor-Networks
- Edge impulse link: https://studio.edgeimpulse.com/studio/87306
- Date: 24 Mar 2022
 
## Introduction
Flowers play a vital role in the connected environment and urban life. When people walk in the park and encounter beautiful flowers, they will wonder what kind they are. But plant experts are rare, and this app can help users solve this problem easily. Users only need to open the mobile phone to take pictures to know the type of flowers. It's like carrying a botanist with you.

Usually, scholars use convolutional neural networks and transfer learning to classify and recognise flowers. According to Emine and Ahmet(2019) transfer learning performs better and gets high accuracy. This research aims to explore what affects the results in flower classification by testing two methods(CNN and transfer learning) based on the flutter flower dataset. The results show that image size, colour depth, training cycles, learning rate and data augmentation are factors influencing the accuracy and loss value. After training and testing the model, the accuracy is obviously improved. At the last step, Edge Impulse was used to create a web application and mobile QR code to capture and classify the new picture this system is never seen.

## Research Question
I want my project to be able to classify images of flowers using the camera on my laptop and iPhone.And here are subproblems that need to be solved:

- Collect flower images dataset and classify these images based on CNN and transfer learning.
- Comparing the differences between the two methods
- Explore the factors that affect the results of the experiment and improve results
- Deploy applications to run final impulse on the mobile phone and computer 

## Application Overview

This application allows users to use the camera on their laptop or camera to capture a picture then analyse the type of flowers. You can use it whenever and wherever even without the internet. Without spending a dime or buying new sensors, any user can test the flowers that appear in their lives.
<img width="900" alt="Screenshot 2022-03-24 at 15 06 06" src="https://user-images.githubusercontent.com/99146042/159947046-aa3c8f22-bbe2-48e4-b0ff-8968008c335c.png">

## Data
The flower dataset provided by TensorFlow officail example(http://download.tensorflow.org/example_images/flower_photos.tgz). It has 2904 images with 5 labels(Daisy, Dandelions, sunflowers, rose, tulips). I split these as training data(80%) and test data(20%). Training data was used to train the model, while test data was used to evaluate how well the model perform on the new data.
Daisy|Dandelions|Sunflowers|Rose|Tulips
--|:--:|--:|--:|--:
493|701|560|509|641

Here are some labled images:
![image](https://user-images.githubusercontent.com/99146042/159248053-c2df3e4f-96f9-4fb8-973e-c7ffd2b7bbb1.png)
(https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model)

## Model

Edge impulse provides three models for images classification: Transfer learning, Classification(Keras) and Regression(Keras). According to the research of Gulli, Kapoor and Pal(2019), regression(Keras) are great for predicting numeric continuous values. So this research used CNN(Keras) and transfer learning as test models and explore their differences and principles. Edge Impulse comes with Keras and MobileNetv2 by default, so we can use it without importing.

### CNN(Keras) model
The convolutional Neural Network is a kind of deep learning method which has become dominant in various computer vision tasks(Yamashita ***et al***., 2018)). A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer.
<img width="700" alt="image" src="https://user-images.githubusercontent.com/99146042/159966131-d37392c7-3c4d-4b14-989e-847100647acd.png">

The convolutional layer plays a key role in the CNN model. This layer performs a dot product between two matrices, where one matrix is the set of learnable parameters otherwise known as a kernel, and the other matrix is the restricted portion of the receptive field. The pooling layer replaces the output of the network at certain locations by deriving a summary statistic of the nearby outputs. Neurons in this layer have full connectivity with all neurons in the preceding and succeeding layers. This is why it can be computed as usual by a matrix multiplication followed by a bias effect.

In conclusion, from images’ raw pixel data, the CNN model extracts the features automatically and train the model for classification.

### Transfer learning model

While a convolutional neural network is used to solve machine learning problems, the network can be started with random values for training and can be done by transfer learning(Emine & Ahmet, 2019). Transfer learning and domain adaptation refer to the situation where what has been learned in one setting … is exploited to improve generalization in another setting(Goodfellow, Bengio and Courville,  2016) It is just like “standing on the shoulder of giants”. But transfer learning only works in deep learning if the model features learned from the first task are general.

There are two methods to use transfer learning:
- Develop model
- Pre-trained model (the common method)

Using the pre-trained model to use transfer learning, these models are used most in image classification: VGG, ResNet, AlexNet, and GoogleNet(Cengıl and Çinar, 2019).  And it showed that VGG performed best.

### MobileNetv2

MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. This is the transfer learning model included in Edge impulse(Cengıl and Çinar, 2019). It is also a common model for image processing in computer vision. This model is often used in image classification, image segmentation, and object detection. But in the study of identifying flowers, the model did not perform well.

### VGG model

VGG is a pre-training model proposed by Simonyan and Zisserman(2014) in in the ILSVRC-2014 competition. Here is the [source code](https://github.com/Ashley0324/CASA0018---Deep-Learning-for-Sensor-Networks/blob/main/Flower_Images_Classification.ipynb) and training recording. VGG16 adopts a very simple structure. The entire network uses the same size of convolution kernel size (3×3) and max-pooling layer (2×2). There are two structures of VGG, which are 16-layer structures (13 convolutional layers). layer and 3 fully connected layers) and 19-layer structure (16 convolutional layers and 3 fully connected layers). Compared to VGG19, VGG16 has almost the same accuracy, but is faster. Here is the structure of VGG16:
<img width="482" alt="image" src="https://user-images.githubusercontent.com/99146042/160887554-81fbf641-25a4-41d8-ac90-0c8e99d4a8d4.png">

## Experiments

![image](https://user-images.githubusercontent.com/99146042/161139738-910fa090-8c0d-4b1f-b054-03e4e6ad40a4.png)

I followed this process to build this project.
- **Collect data:** In this step, I collected 2904 images with 5 labels. These images were split into two parts: training data(80%) and validation data(20%).
- **Build the model:** I used three models in different tests. The CNN and MobileNetV2 are included in the Edge Impulse. It is convinient to use them by adding a learning block. As a external model, I added VGG16 model to edge impulse fllowing this tutorial: https://docs.edgeimpulse.com/docs
- **Train the model:** After building or importing the model, I changed different parameters in the model to obtain better results. These parameters include: image size, color depth, training cycles, learning rate and data augmentation.
- **Test the model:** It used the model test function in edge impulese to test the model.
- **Deploy the application**: Laptops and mobile phones are devices people use most. So I develop the application on these two types device by Edge Impluse. Users can open this link(https://studio.edgeimpulse.com/studio/87306/deployment) to run the laptop application. While users can scan this QR code to classify flowers in their mobile phones.
![image](https://user-images.githubusercontent.com/99146042/161396001-bf07057b-b197-4ed2-b346-e268b12c84f5.png)

Here is the experience recording of changing models and parameters:
<img width="788" alt="Screenshot 2022-04-17 at 19 03 55" src="https://user-images.githubusercontent.com/99146042/163726797-a3eb5ce6-e390-40b5-b9c9-c2a1103be85e.png">


## Results and Observations

### Evaluate the performance
The evaluation of the results mainly comes from the following three indicators: accuracy（Traning data and validation data）, loss（Traning data and validation data）and running time per epoch.

### The parameters what impacts the results

1.Image size
We resized all images to equal dimetions before training, this is the width and height all images are resized to. Bigger image size means more image pixel values.Then the more features are generated. So bigger image size will perfomance better.

2.Color depth
RGB image performance better than grayscale because it has colour imaformation and generate more features.

3.Training cycles
This is the number of times a training cycle is repeated. In a neural network, every time a training record is considered, the previous weights are quite different, and hence, it is necessary to repeat the cycle many times.  

4.Learning rate
This hyperparameter defines the size of each step in the gradient descent. Set smaller numbers to see ‘slower’ learning. The value of the learning rate is between 0 and 1. Choosing the learning rate is challenging as a value too small may result in a long training process that could get stuck, whereas a value too large may result in learning a sub-optimal set of weights too fast or an unstable training process. Smith(2017) proposes a method to find the optimal learning rate , setting a very accurate learning rate after each epoch. 20 times of learning rate, it is good to find out how much the learning rate of each period is tested, and finally compare the loss or accuracy of different learning rates. I find the best learning rate in this model is 0.001 in VGG16 model.

5.Data augmentatio
This will be introduced in next part.

### Overfitting and data augmentation

#### Overfitting 

Hawkins(2004) defined overfitting as "the resulting analysis that corresponds too closely or precisely to a particular set of data and therefore may fail to fit additional data or reliably predict future observations”. In the above experiments, the model training results of the training data are significantly better than those of the test data, which is a case of overfitting.

Overfitting is usually caused by:
- The amount of data in the centralized data is too small. The training data completely reflects the regularity of all data. Therefore, it is not possible for the model to extract some special features rather than realistic features.
- Stochastic noise can also lead to overfitting.
- The third reason is the high complexity of the objective function. Produces deterministic noise.

#### Data augmentation

In order to solve the problem of overfitting, the most common method is to increase the amount of data. However, obtaining sample data is difficult in most cases. Then a simple transformation can be performed on the existing samples to obtain more samples. For example, the image can be flipped, so that the features change, but the target remains the same so that it can be regarded as generating some new samples. This process is data augmentation.

### Results on devices

#### Overview of the results

In the end I got 99.84% accuracy in the traning data while 84.92% accuract in the test data on my computer. The results obtained by this method have higher accuracy than the TensorFlow case. Actually, the accuracy in the official case is 80.30%(https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model), which only use transfer learning and model from TF-Hub. Besides, in anonther example, the author used convolutional neural network and the Sequential model to classify the same dataset and the accuracy is 78%.(https://www.tensorflow.org/tutorials/images/classification)

It can be seen that the effect of VGG16 model is quite stable, and the calculation time in the whole process is only more than 30 minutes, which is the charm of transfer learning. It's less than CNN model, and similar with MobileNetv2 model. So, I think it works perfectly for high accuracy and less runtime. In the future I will continue to adjust the batch size, or the structure of the model to get a better result.

#### Advantages

- Model choice: Among the three models, VGG16 performs significantly better, and its impact on the results is significant.
- Parameter tuning: During multiple tests, tuning of different parameters and then realizing that they can have a positive or negative effect on the results. Reading through the literature, I understand why these parameters make effect.
- Device selection: I selected two devices, a laptop and a mobile phone, to run the program. They are very convenient to use.

#### Disadvantages
- Overfitting
However, there is still an overfitting problem. This is because the training dataset numbers are not enough. To solve this problem, I need to collect more flower images.

### Future Work

To improve the effect of apllication, I'd like to do this actions:
- Add the number of each kind of flower images.
- Build the dataset with more types flowers.
- Add the training cycle.
- When I tried to add the training cycles from 20 to 25 or higher, the system always fail to do it. I still can't find the reason. I'll work it in the future.
![image](https://user-images.githubusercontent.com/99146042/161394943-7ab9eb31-16b7-4006-af8a-8493a83bca22.png)


## Bibliography

1. Cengıl, E., & Çinar, A. (2019, September). Multiple classification of flower images using transfer learning. In 2019 International Artificial Intelligence and Data Processing Symposium (IDAP) (pp. 1-6). IEEE.
2. Gulli, A., Kapoor, A., & Pal, S. (2019). Deep learning with TensorFlow 2 and Keras: regression, ConvNets, GANs, RNNs, NLP, and more with TensorFlow 2 and the Keras API. Packt Publishing Ltd.
3. Yamashita, R., Nishio, M., Do, R. K. G., & Togashi, K. (2018). Convolutional neural networks: an overview and application in radiology. Insights into imaging, 9(4), 611-629.
4. Gupta, P. R., Sharma, D., & Goel, N. (2022). Image Forgery Detection by CNN and Pretrained VGG16 Model. In Proceedings of Academia-Industry Consortium for Data Science (pp. 141-152). Springer, Singapore.
5. Cengıl, E., & Çinar, A. (2019, September). Multiple classification of flower images using transfer learning. In 2019 International Artificial Intelligence and Data Processing Symposium (IDAP) (pp. 1-6). IEEE.
6. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
7. Hawkins, D. M. (2004). The problem of overfitting. Journal of chemical information and computer sciences, 44(1), 1-12.
8. Smith, L. N. (2017, March). Cyclical learning rates for training neural networks. In 2017 IEEE winter conference on applications of computer vision (WACV) (pp. 464-472). IEEE.

----

## Declaration of Authorship

I, AUTHORS NAME HERE, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

Qian Jin
28 Mar 2022
