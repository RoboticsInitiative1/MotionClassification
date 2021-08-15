# MOTION CLASSIFICATION (HAR)

![HAR](https://camo.githubusercontent.com/79793dddf3ce2ce2220bf50b3d8c1d2747cb2c60d7454ff8dcc30a92fa09ab6a/68747470733a2f2f7777772e616e64726f6964686976652e696e666f2f77702d636f6e74656e742f75706c6f6164732f323031372f31322f616e64726f69642d757365722d61637469766974792d7265636f676e6974696f6e2d7374696c6c2d77616c6b696e672d72756e6e696e672d64726976696e672e6a7067)

**This Project is done with two approaches: -**
**1. First is using LSTM deep learning architecture.**
**2. Second is with LSTM-CNN architecture.**



## Overview

**Human Activity Recognition** is performed using **PAMAP2** physical activity monitoring dataset. Classifying the pose over time into **6** classes:- 
1. Rope Jumping
2. Walking
3. Descending stairs
4. Ascending stairs
5. Nordic walk
6. Running

## Content
 1. Dataset.
 2. What is LSTM?
 3. What is CNN?
 4. Working: -
          (a) LSTM structue.
          (b) CNN structure.
  5. Results.
  6. Application.
  7. Enhancement Scope.     
  

## 1. Dataset

**Data Set Information:**

The PAMAP2 Physical Activity Monitoring dataset contains data of 18 different physical activities (such as walking, cycling, playing soccer, etc.), performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor. The dataset can be used for activity recognition and intensity estimation, while developing and applying algorithms of data processing, segmentation, feature extraction and classification.  
  
 **Sensors**  
3 Colibri wireless inertial measurement units (IMU):  
- sampling frequency: 100Hz  
- position of the sensors:  
- 1 IMU over the wrist on the dominant arm  
- 1 IMU on the chest  
- 1 IMU on the dominant side's ankle  
HR-monitor:  
- sampling frequency: ~9Hz  
  
**Data collection protocol**  
Each of the subjects had to follow a protocol, containing 12 different activities. The folder â€œProtocolâ€ contains these recordings by subject.  
Furthermore, some of the subjects also performed a few optional activities. The folder â€œOptionalâ€ contains these recordings by subject.  
  
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
4 walking  
5 running   
7 Nordic walking  
12 ascending stairs  
13 descending stairs    
24 rope jumping  
0 other (transient activities)
          
 
## 2. LSTM: -

**LSTM Network**

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced in  (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

Learn more about LSTM [here](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/)

## 3. CNN: -

**CNN Network**

A convolutional neural network (CNN) is a type of  artificial neural network used in image recognitionand processing that is specifically designed to process pixel data.

CNNs are powerful image processing, artificial intelligence AI that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition, along with recommender systems and natural language processing NLP.

A neural network is a system of hardware and/or software patterned after the operation of neurons in the human brain. Traditional neural networks are not ideal for image processing and must be fed images in reduced-resolution pieces. CNN have their “neurons” arranged more like those of the frontal lobe, the area responsible for processing visual stimuli in humans and other animals. The layers of neurons are arranged in such a way as to cover the entire visual field avoiding the piecemeal image processing problem of traditional neural networks.

A CNN uses a system much like a multilayer perceptron that has been designed for reduced processing requirements. The layers of a CNN consist of an input layer, an output layer and a hidden layer that includes multiple convolutional layers, pooling layers, fully connected layers and normalization layers. The removal of limitations and increase in efficiency for image processing results in a system that is far more effective, simpler to trains limited for image processing and natural language processing.

Learn more about CNN [here](https://searchenterpriseai.techtarget.com/definition/convolutional-neural-network)


## 4. Working: -

## **(a)  Structure Of LSTM: -**

LSTM has a chain structure that contains four neural networks and different memory blocks called  **cells**. Information is retained by the cells and the memory manipulations are done by the **gates.** 
There are three gates –

![enter image description here](https://media.geeksforgeeks.org/wp-content/uploads/newContent1.png)

**1. Forget Gate:** The information that no longer useful in the cell state is removed with the forget gate. Two inputs _x_t_ (input at the particular time) and _h_t-1_ (previous cell output) are fed to the gate and multiplied with weight matrices followed by the addition of bias. The resultant is passed through an activation function which gives a binary output. If for a particular cell state the output is 0, the piece of information is forgotten and for the output 1, the information is retained for the future use.

![enter image description here](https://media.geeksforgeeks.org/wp-content/uploads/newContent2.png)

**2. Input gate:** Addition of useful information to the cell state is done by input gate. First, the information is regulated using the sigmoid function and filter the values to be remembered similar to the forget gate using inputs _h_t-1_ and _x_t_. Then, a vector is created using _tanh_ function that gives output from -1 to +1, which contains all the possible values from h_t-1 and _x_t_. Atlast, the values of the vector and the regulated values are multiplied to obtain the useful information.

![enter image description here](https://media.geeksforgeeks.org/wp-content/uploads/newContent4.png)

**3. Output gate:** The task of extracting useful information from the current cell state to be presented as an output is done by output gate. First, a vector is generated by applying tanh function on the cell. Then, the information is regulated using the sigmoid function and filter the values to be remembered using inputs _h_t-1_ and _x_t_. Atlast, the values of the vector and the regulated values are multiplied to be sent as an output and input to the next cell.

![enter image description here](https://media.geeksforgeeks.org/wp-content/uploads/newContent3.png)



**LSTM Applications**

LSTM networks have useful applications in the following areas:

-   Language modeling
-   Machine translation
-   Handwriting recognition
-   Image captioning
-   Image generation using attention models
-   Question answering
-   Video-to-text conversion
-   Polymorphic music modeling
-   Speech synthesis
-   Protein secondary structure prediction

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

## 5. Results

### LSTM Architecture:-

> Output 1:  Dataset used
> ![Dataset used](https://i.ibb.co/m8TBsv4/1.jpg)
> Output 2:  Confusion Matrix
> ![enter image description here](https://i.ibb.co/vZY8B16/2.jpg)
> Output 3:  Accuracy
> ![enter image description here](https://i.ibb.co/NFJp6yk/3.jpg)

### LSTM-CNN Architecture
> Output 1: Confusion Matrix
> ![enter image description here](https://i.ibb.co/c1s5vdV/IMG-20210814-203509.jpg)
> Output 2 : Accuracy
> ![enter image description here](https://i.ibb.co/DLy5v3q/IMG-20210814-203424.jpg)


## 6. Applications

Application of *Human Activity Recognition* are as follows: -

1. Indoor localization.
2. Aid first responder.
3. Home-based rehabilitation.
4. Security-related applications.
5. Logistics support to location-based services.
6. Wildlife observation.
7. Energy conservation in buildings.



## 7. Enhancement

In order to achieve more accurate results one can go for trying different deep learning algorithm or some modification in the existing method like trying: -
1. LSTM-CNN approach
2. SVM approach
3. Different datasets etc...


