# MOTION CLASSIFICATION (HAR)

![HAR](https://i.ibb.co/rQYXZtp/Screenshot-8.png)



**This Project is done with two approaches: -**
**1. Using SVM supervised machine learning architecture.**
**2. Using CNN deep learning architecture.**



## Overview

**Human Activity Recognition or Motion Classification** is performed on **PAMAP2** physical activity monitoring dataset. To classify the pose over time into **6** classes:- 
1. Ascending stairs
2. Descending stairs
3. Nordic walk
4. Rope Jumping
5.  Running
6. Walking

## Content
 1. Dataset.
 2. What is SVM?
 3. What is CNN?
 4. Working: -
          (a) SVM structue.
          (b) CNN structure.
  5. Results.
  6. Application.
  7. Enhancement Scope.     
  

## 1. Dataset

**Data Set Information:**

The PAMAP2 Physical Activity Monitoring dataset contains data of 18 different physical activities (such as walking, cycling, playing soccer, etc.), performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor. The dataset can be used for activity recognition and intensity estimation, while developing and applying algorithms of data processing, segmentation, feature extraction and classification.  
  
 **Sensors**  
3 Colibri wireless inertial measurement units (IMU):  
- Sampling frequency: 100Hz  
- Position of the sensors:  
- 1 IMU over the wrist on the dominant arm  
- 1 IMU on the chest  
- 1 IMU on the dominant side's ankle  
HR-monitor:  
- Sampling frequency: ~9Hz  
  
  
**Data files**  
Raw sensory data can be found in space-separated text-files (.dat), 1 data file per subject per session (protocol or optional). Missing values are indicated with NaN. One line in the data files correspond to one timestamped and labeled instance of sensory data. The data files contain 54 columns: each line consists of a timestamp, an activity label (the ground truth) and 52 attributes of raw sensory data.

**Attribute Information:**

The 54 columns in the data files are organized as follows:  
1. timestamp (s)  
2. activityID (see below for the mapping to the activities)  
3. heart rate (bpm)  
4-20. IMU hand  
21-37. IMU chest  
38-54. IMU ankle  
  
The IMU sensory data contains the following columns:  
1. temperature (Â°C)  
2-4. 3D-acceleration data (ms-2), scale: Â±16g, resolution: 13-bit  
5-7. 3D-acceleration data (ms-2), scale: Â±6g, resolution: 13-bit  
8-10. 3D-gyroscope data (rad/s)  
11-13. 3D-magnetometer data (Î¼T)  
14-17. orientation (invalid in this data collection)  
  
List of **activityIDs** and corresponding activities:   
4 Walking  
5 Running   
7 Nordic walking  
12 Ascending stairs  
13 Descending stairs    
24 Rope jumping  

          
 
## 2. SVM: -

**SVM Model**

SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.

SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.

Lets begin with a problem. Suppose you have a dataset as shown below and you need to classify the red rectangles from the blue ellipses(let’s say positives from the negatives). So your task is to find an ideal line that separates this dataset in two classes (say red and blue).
![SVM](https://i.ibb.co/ZWpK7FK/Screenshot-9.png)
Learn more about LSTM [here](https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989)

## 3. CNN: -

**CNN Network**

A convolutional neural network (CNN) is a type of  artificial neural network used in image recognitionand processing that is specifically designed to process pixel data.

CNNs are powerful image processing, artificial intelligence AI that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition, along with recommender systems and natural language processing NLP.

A neural network is a system of hardware and/or software patterned after the operation of neurons in the human brain. Traditional neural networks are not ideal for image processing and must be fed images in reduced-resolution pieces. CNN have their “neurons” arranged more like those of the frontal lobe, the area responsible for processing visual stimuli in humans and other animals. The layers of neurons are arranged in such a way as to cover the entire visual field avoiding the piecemeal image processing problem of traditional neural networks.

A CNN uses a system much like a multilayer perceptron that has been designed for reduced processing requirements. The layers of a CNN consist of an input layer, an output layer and a hidden layer that includes multiple convolutional layers, pooling layers, fully connected layers and normalization layers. The removal of limitations and increase in efficiency for image processing results in a system that is far more effective, simpler to trains limited for image processing and natural language processing.
![CNN](https://www.upgrad.com/blog/wp-content/uploads/2020/12/1-4.png)
Learn more about CNN [here](https://searchenterpriseai.techtarget.com/definition/convolutional-neural-network)


## 4. Working: -

## **(a)  Structure Of SVM: -**

The basics of Support Vector Machines and how it works are best understood with a simple example. Let’s imagine we have two tags: _red_ and _blue_, and our data has two [features](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/#feature-engineering): _x_ and _y_. We want a classifier that, given a pair of _(x,y)_ coordinates, outputs if it’s either _red_ or _blue_. We plot our already labeled training data on a plane:

![enter image description here](https://monkeylearn.com/static/52081a1b625e8ba22c00210d547b4f1a/e7065/plot_original.webp)

A support vector machine takes these data points and outputs the hyperplane (which in two dimensions it’s simply a line) that best separates the tags. This line is the **decision boundary**: anything that falls to one side of it we will classify as _blue_, and anything that falls to the other as _red_.

![enter image description here](https://monkeylearn.com/static/57fd2448dfb67cfff990f32191463e80/e7065/plot_hyperplanes_2.webp)



But, what exactly is _the best_ hyperplane? For SVM, it’s the one that maximizes the margins from both tags. In other words: the hyperplane (remember it's a line in this case) whose distance to the nearest element of each tag is the largest.

![enter image description here](https://monkeylearn.com/static/7002b9ebbacb0e878edbf30e8ff5b01c/e7065/plot_hyperplanes_annotated.webp)

### Nonlinear data

Now this example was easy, since clearly the data was linearly separable — we could draw a straight line to separate  _red_  and  _blue_. Sadly, usually things aren’t that simple. Take a look at this case:

![enter image description here](https://monkeylearn.com/static/2631f704a0b3f6e31246294578a7d777/8f244/plot_circle_01.webp)

It’s pretty clear that there’s not a linear decision boundary (a single straight line that separates both tags). However, the vectors are very clearly segregated and it looks as though it should be easy to separate them.

So here’s what we’ll do: we will add a third dimension. Up until now we had two dimensions:  _x_  and  _y_. We create a new  _z_  dimension, and we rule that it be calculated a certain way that is convenient for us:  _z = x² + y²_  (you’ll notice that’s the equation for a circle).

This will give us a three-dimensional space. Taking a slice of that space, it looks like this:
![enter image description here](https://monkeylearn.com/static/a4dc8a44f6a8adf55df920a602668a42/8f244/plot_circle_04.webp)


**SVM Applications**

SVM model have useful applications in the following areas:

-   **Face detection**
-   **Text and hypertext categorization**
-   **Classification of images**
-   **Bioinformatics**
-   **Handwriting recognition**
-   **Generalized predictive control(GPC)**
-   **Protein fold and remote homology detection**

## **(b)  Structure of CNN: -**

### **Convolution Layers**

There are three types of layers that make up the CNN which are the convolutional layers, pooling layers, and fully-connected (FC) layers. When these layers are stacked, a CNN architecture will be formed. In addition to these three layers, there are two more important parameters which are the dropout layer and the activation function which are defined below.

**Convolutional Layer**

This layer is the first layer that is used to extract the various features from the input images. In this layer, the mathematical operation of convolution is performed between the input image and a filter of a particular size MxM. By sliding the filter over the input image, the dot product is taken between the filter and the parts of the input image with respect to the size of the filter (MxM). The output is termed as the Feature map which gives us information about the image such as the corners and edges. Later, this feature map is fed to other layers to learn several other features of the input image.

**Pooling Layer**

In most cases, a Convolutional Layer is followed by a Pooling Layer. The primary aim of this layer is to decrease the size of the convolved feature map to reduce the computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map. Depending upon method used, there are several types of Pooling operations.

Here we used Max Pooling, the largest element is taken from feature map. Where the Average Pooling calculates the average of the elements in a predefined sized Image section. The total sum of the elements in the predefined section is computed in Sum Pooling. The Pooling Layer usually serves as a bridge between the Convolutional Layer and the FC Layer.

**Fully Connected Layer**

The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.
In this, the input image from the previous layers are flattened and fed to the FC layer. The flattened vector then undergoes few more FC layers where the mathematical functions operations usually take place. In this stage, the classification process begins to take place.

**Dropout**

Usually, when all the features are connected to the FC layer, it can cause overfitting in the training dataset. Overfitting occurs when a particular model works so well on the training data causing a negative impact in the model’s performance when used on a new data.
To overcome this problem, a dropout layer is utilised wherein a few neurons are dropped from the neural network during training process resulting in reduced size of the model. 

**Activation Functions**

Finally, one of the most important parameters of the CNN model is the activation function. They are used to learn and approximate any kind of continuous and complex relationship between variables of the network. In simple words, it decides which information of the model should fire in the forward direction and which ones should not at the end of the network.
It adds non-linearity to the network. Here we used activation functions such as the ReLU, Softmax. Each of these functions have a specific usage. For a binary classification CNN model, sigmoid and softmax functions are preferred an for a multi-class classification, generally softmax us used.

**SVM Applications**

SVM model have useful applications in the following areas:

-   **Decoding Facial Recognition**
-   **Analyzing Documents**
-   **Classification of images**
-   **Understanding Climate**
-   **Handwriting recognition**


## 5. Results

### SVM  Approch:-

> Output 1:  Dataset used
> ![Dataset used](https://i.ibb.co/DRGSKCq/Screenshot-10.png)
> Output 2:  Confusion Matrix
> ![enter image description here](https://i.ibb.co/hWHSSVG/Screenshot-12.png)
> Output 3:  Accuracy
> ![enter image description here](https://i.ibb.co/qpLHFXw/Screenshot-11.png)

### CNN Architecture
> Output 1: Confusion Matrix
> ![enter image description here](https://i.ibb.co/cwS22mj/Screenshot-15.png)
> Output 2 : Accuracy
> ![enter image description here](https://i.ibb.co/2qMyBKH/Screenshot-14.png)


## 6. Applications

Application of *Human Activity Recognition* are as follows: -

1. Aid first responder.
2. Energy conservation in buildings.
3. Home-based rehabilitation.
4. Indoor localization.
5. Logistics support to location-based services.
6. Security-related applications.
7. Wildlife observation.



## 7. Enhancement

In order to achieve more accurate results one can go for trying different deep learning algorithm or some modification in the existing method like trying: -
1. LSTM-CNN approach
2. SVM approach (using different kerenl or increase the training set).
3. Different datasets etc...


