# Classify Flowers with Transfer Learning and VGG model
Qian Jin
Repo link:https://github.com/Ashley0324/CASA0018---Deep-Learning-for-Sensor-Networks

## Introduction
This model is designed to classify different kinds of flowers. The inspiration for this project is when I move to London, I buy flowers every week, but some flowers I have never seen is really attractive and I wondered what kind of flower it is. So, I want to find a way to identify these beautiful flowers.

We have two methods to choose based on machine learning to achieve this goal. They are convolutional neural networks (CNN, we used it to recognise cat and dog in lecture part) and transfer learning. I used transfer learning to achieve my goals because it uses less time and is more clever by extracting useful features already learnt from millions of pictures. Higher accuracy and faster convergence are also advantages of this method.((Duncan, CASA0018_06))

So, how do we do it? This process includes three parts:

- Find a pretrained model based on a dataset that represents the data you are working with. 
- Take the weights of that pretrained model by removing the last “classifier” layer 
- Freeze the pre-trained layers and add in a new dense layer and classifier which will be the trainable layers
(Duncan, CASA0018_06)

Ultimately, the recognition rate through this method can be as high as over 80%.
![image](https://user-images.githubusercontent.com/99146042/155976249-116852f7-f7ae-4250-98a0-c0246de7d439.png)

## Research Question
I am trying to solve the problem of classifying different kinds of flowers with high accuracy way.

## Application Overview
Thinking back to the various application diagrams you have seen through the module - how would you describe an overview of the building blocks of your project - how do they connect, what do the component parts include.

*probably ~200 words and a diagram is usually good to convey your design!*

## Data
In this project, I used the TensorFlow flower example dataset (http://download.tensorflow.org/example_images/flower_photos.tgz). This dataset consists of images of flowers with 5 labels.  The flowers dataset consists of examples which are labelled images of flowers. Each example contains a JPEG flower image and the class label: what type of flower it is. Let's display a few images together with their labels.
<img width="500" alt="Screenshot 2022-02-28 at 11 48 31" src="https://user-images.githubusercontent.com/99146042/155981071-ff8feabd-cc10-4736-ac81-0088386409a0.png">

When training a machine learning model, we split our data into training and test datasets. Generally, the training and validation data set is split into an 80:20 ratio. Thus, 20% of the data is set aside for validation purposes(Deep learning, 2016). Because the amount of data for different types of flowers is not exactly the same, and the data in the labels array has not been shuffled, the most appropriate method is to use the StratifiedShuffleSplit method to perform hierarchical random division.
![image](https://user-images.githubusercontent.com/99146042/155981144-0033e208-cf7a-4f97-9def-e220e83a4102.png)


## Model
VGG is the winning model of the ILSVRC competition in the visual field in 2014, with an error rate of 7.3% on the ImageNet dataset, which greatly broke the previous year's world record of 11.7%. VGG16 basically inherited the deep ideas of AlexNet, and carried it forward, making it deeper. AlexNet only uses an 8-layer network, while the two versions of VGG are a 16-layer network version and a 19-layer network version. In the following transfer learning practice, I use the slightly simpler VGG16, which has almost the same accuracy as VGG19, but is faster.

![image](https://user-images.githubusercontent.com/99146042/155983752-3a0fa4bb-8911-4364-a047-cbde8c68327d.png)

The input data format of VGG is 244 * 224 * 3 pixel data. After a series of convolutional neural network and pooling network processing, the output is a 4096-dimensional feature data, and then through a 3-layer fully connected neural network processing, and finally the classification results are obtained by softmax normalization.

First, we will hand over all the pictures to VGG16, and use the five-round convolutional network layer and pooling layer in the deep network structure of VGG16 to obtain a 4096-dimensional feature vector for each picture, and then we directly use this feature vector instead The original image, plus several layers of fully connected neural networks, are trained on the flower dataset.

Therefore, in essence, we use VGG16 as an image feature extractor and then perform an ordinary neural network learning on this basis, so that the original 244 * 224 * 3 dimension data is converted into 4096 dimensions, and each amount of one-dimensional information is greatly increased, which greatly reduces the consumption of computing resources, and realizes the application of knowledge obtained in learning object recognition to special flower classification problems.


## Experiments
What experiments did you run to test your project? What parameters did you change? How did you measure performance? Did you write any scripts to evaluate performance? Did you use any tools to evaluate performance? Do you have graphs of results? 

*probably ~300 words and graphs and tables are usually good to convey your results!*

## Results and Observations
Synthesis the main results and observations you made from building the project. Did it work perfectly? Why not? What worked and what didn't? Why? What would you do next if you had more time?  

In the end I got 88.83% accuracy on my computer, you can continue to adjust the batch size, or the structure of the model to get a better result. The results obtained by this method have higher accuracy than the TensorFlow case. Actually, the accuracy in the official case is 80.30%.(https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model), which only use transfer learning and model from TF-Hub. Besides, in anonther example, the author used convolutional neural network and the Sequential model to classify the same dataset and the accuracy is 78%.

It can be seen that the effect of the model is quite stable, and the calculation time in the whole process is only more than 30 minutes, which is the charm of transfer learning. So, I think it works perfectly. And this is because I used the VGG model. In the future I will continue to adjust the batch size, or the structure of the model to get a better result.(https://www.tensorflow.org/tutorials/images/classification)

*probably ~300 words and remember images and diagrams bring results to life!*

## Bibliography

1. Duncan, W. (2022). CASA0018_06. London: UCL, 7-9. https://moodle.ucl.ac.uk/pluginfile.php/4514917/mod_resource/content/1/CASA0018_06.pdf
2. Last name, First initial. (Year published). Title. Edition. (Only include the edition if it is not the first edition) City published: Publisher, Page(s). http://google.com
3. Goodfellow, I., Bengio, Y. & Courville, A., 2016. Deep learning / Ian Goodfellow, Yoshua Bengio and Aaron Courville.
4. https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
5. https://www.tensorflow.org/hub/tutorials/image_feature_vector#build_the_model

----

## Declaration of Authorship

I, AUTHORS NAME HERE, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

Qian Jin
28 Mar 2022
