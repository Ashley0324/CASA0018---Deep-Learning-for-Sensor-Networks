# Classify Flowers with Transfer Learning and VGG model
Qian Jin
- Repo link:https://github.com/Ashley0324/CASA0018---Deep-Learning-for-Sensor-Networks
- Edge impulse link: https://studio.edgeimpulse.com/studio/87595/
- Date: 24 Mar 2022

## Introduction
Flowers play a vital role in the connected environment and urban life. Flower classification in a high accuracy is helpful to biologists and people who like flowers. Usually, scholars use convolutional neural networks and transfer learning to classify and recognise flowers. According to Yong(2018) transfer learning performs better and gets high accuracy. This research aims to explore what affects the results in flower classification by testing two methods(CNN and transfer learning) based on the flutter flower dataset. The results show that image size, colour depth, training cycles, learning rate and data augmentation are factors influencing the accuracy and loss value. After training and testing the model, the accuracy is obviously improved. At the last step, Edge Impulse was used to create a web application and mobile QR code to capture and classify the new picture this system is never seen.

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

Edge impulse provides three models for images classification: Transfer learning, Classification(Keras) and Regression(Keras). According to the research of Gulli, Kapoor and Pal(2019), regression(Keras) are great for predicting numeric continuous values. So this research used CNN(Keras) and transfer learning as test models and explore their differences and principles.

### CNN(Keras) model
The convolutional Neural Network is a kind of deep learning method which has become dominant in various computer vision tasks(Rikiya,2018). A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer. (Sharmad,2021). 
<img width="700" alt="image" src="https://user-images.githubusercontent.com/99146042/159966131-d37392c7-3c4d-4b14-989e-847100647acd.png">

The convolutional layer plays a key role in the CNN model. This layer performs a dot product between two matrices, where one matrix is the set of learnable parameters otherwise known as a kernel, and the other matrix is the restricted portion of the receptive field. The pooling layer replaces the output of the network at certain locations by deriving a summary statistic of the nearby outputs. Neurons in this layer have full connectivity with all neurons in the preceding and succeeding layers. This is why it can be computed as usual by a matrix multiplication followed by a bias effect.

In conclusion, from images’ raw pixel data, the CNN model extracts the features automatically and train the model for classification.

### Transfer learning model

While a convolutional neural network is used to solve machine learning problems, the network can be started with random values for training and can be done by transfer learning(Emine & Ahmet, 2019). Transfer learning and domain adaptation refer to the situation where what has been learned in one setting … is exploited to improve generalization in another setting(Goodfellow, Bengio, & Courville,  2016) It is just like “standing on the shoulder of giants”. But transfer learning only works in deep learning if the model features learned from the first task are general.

There are two methods to use transfer learning:
- Develop model
- Pre-trained model (the common method)

Using the pre-trained model to use transfer learning, these models are used most in image classification: VGG, ResNet, AlexNet, and GoogleNet(Emine & Ahmet, 2019).  And it showed that VGG performed best.

### MobileNetv2

MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. This is the transfer learning model included in Edge impulse. It is also a common model for image processing in computer vision. This model is often used in image classification, image segmentation, and object detection. But in the study of identifying flowers, the model did not perform well.

### VGG model

VGG is a pre-training model proposed by Oxford's Visual Geometry Group, so the origin of the VGG name is to take the initials of these three words. VGG16 adopts a very simple structure. The entire network uses the same size of convolution kernel size (3×3) and max-pooling layer (2×2). There are two structures of VGG, which are 16-layer structures (13 convolutional layers). layer and 3 fully connected layers) and 19-layer structure (16 convolutional layers and 3 fully connected layers). Compared to VGG19, VGG16 has almost the same accuracy, but is faster. Here is the structure of VGG16:
<img width="482" alt="image" src="https://user-images.githubusercontent.com/99146042/160887554-81fbf641-25a4-41d8-ac90-0c8e99d4a8d4.png">

## Experiments

![image](https://user-images.githubusercontent.com/99146042/161139738-910fa090-8c0d-4b1f-b054-03e4e6ad40a4.png)

I followed this process to build this project.

<img width="816" alt="Screenshot 2022-03-31 at 23 56 55" src="https://user-images.githubusercontent.com/99146042/161162997-69ce3d4c-2d91-441f-add7-dcdeca8869af.png">


## Results and Observations

### Evaluate the performance

The evaluation of the results mainly comes from the following three indicators: accuracy（Traning data and validation data）, loss（Traning data and validation data）and running time per epoch.

### The parameters what impacts the results
- Image size: the biger image size is, the more features the data have, so bigger image size will perfomance better.
- Color depth: RGB image performance better than grayscale because it has colour imaformation and generate more features.
- Training cycles: 
- Learning rate
- Data augmentation
- Neurons

### Overfitting and data augmentation

#### Overfitting 

According to Hawkins(2004), overfitting is "the resulting analysis that corresponds too closely or precisely to a particular set of data and therefore may fail to fit additional data or reliably predict future observations”. In the above experiments, the model training results of the training data are significantly better than those of the test data, which is a case of overfitting.

Overfitting is usually caused by:
- The amount of data in the centralized data is too small. The training data completely reflects the regularity of all data. Therefore, it is not possible for the model to extract some special features rather than realistic features.
- Stochastic noise can also lead to overfitting.
- The third reason is the high complexity of the objective function. Produces deterministic noise.

#### Data augmentation

In order to solve the problem of overfitting, the most common method is to increase the amount of data. However, obtaining sample data is difficult in most cases. Then a simple transformation can be performed on the existing samples to obtain more samples. For example, the image can be flipped, so that the features change, but the target remains the same so that it can be regarded as generating some new samples. This process is data augmentation

### Results on devices

#### Overview of the results
Synthesis the main results and observations you made from building the project. Did it work perfectly? Why not? What worked and what didn't? Why? What would you do next if you had more time?  

In the end I got 88.83% accuracy on my computer. The results obtained by this method have higher accuracy than the TensorFlow case. Actually, the accuracy in the official case is 80.30%.(https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model), which only use transfer learning and model from TF-Hub. Besides, in anonther example, the author used convolutional neural network and the Sequential model to classify the same dataset and the accuracy is 78%.

It can be seen that the effect of the model is quite stable, and the calculation time in the whole process is only more than 30 minutes, which is the charm of transfer learning. So, I think it works perfectly. And this is because I used the VGG16 model. In the future I will continue to adjust the batch size, or the structure of the model to get a better result.(https://www.tensorflow.org/tutorials/images/classification)

#### Advantages

- Model choice: Among the three models, VGG16 performs significantly better, and its impact on the results is significant.
- Parameter tuning: During multiple tests, tuning of different parameters and then realizing that they can have a positive or negative effect on the results. Reading through the literature, I understand why these parameters make effect.
- Device selection: I selected two devices, a laptop and a mobile phone, to run the program. They are very convenient to use.

#### Disadvantages
- Overfitting
However, there is still an overfitting problem. This is because the training dataset numbers are not enough. To solve this problem, I need to collect more flower images.

- Runtime
Even now the runtime has been reduced. But it still has room for improvement due to my laptop configuration issues. Computation time can be further reduced if better commercial servers or higher-configured computers are used.

## Bibliography

Some references don't have City published
1. Duncan, W. (2022). CASA0018_06. London: UCL, 7-9. https://moodle.ucl.ac.uk/pluginfile.php/4514917/mod_resource/content/1/CASA0018_06.pdf
2. Last name, First initial. (Year published). Title. Edition. (Only include the edition if it is not the first edition) City published: Publisher, Page(s). http://google.com
3. Goodfellow, I., Bengio, Y. & Courville, A., 2016. Deep learning / Ian Goodfellow, Yoshua Bengio and Aaron Courville.
4. Gulli, A., Kapoor, A., & Pal, S. (2019). Deep learning with TensorFlow 2 and Keras: regression, ConvNets, GANs, RNNs, NLP, and more with TensorFlow 2 and the Keras API. Packt Publishing Ltd.
5. Smith, L. N. (2017, March). Cyclical learning rates for training neural networks. In 2017 IEEE winter conference on applications of computer vision (WACV) (pp. 464-472). IEEE.
6. https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
7. https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model
8. D. Grattarola and C. Alippi, "Graph Neural Networks in TensorFlow and Keras with Spektral [Application Notes]," in IEEE Computational Intelligence Magazine, vol. 16, no. 1, pp. 99-106, Feb. 2021, doi: 10.1109/MCI.2020.3039072.
9. E. Cengıl and A. Çinar, "Multiple Classification of Flower Images Using Transfer Learning," 2019 International Artificial Intelligence and Data Processing Symposium (IDAP), 2019, pp. 1-6, doi: 10.1109/IDAP.2019.8875953.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press
11. Hawkins, D. M. (2004). The problem of overfitting. Journal of chemical information and computer sciences, 44(1), 1-12.

----

## Declaration of Authorship

I, AUTHORS NAME HERE, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

Qian Jin
28 Mar 2022
