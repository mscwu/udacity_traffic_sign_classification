# **Traffic Sign Recognition** 

---

## **Build a Traffic Sign Recognition Project**

### The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image_output/sample_signs.jpg "Sample_signs"
[image2]: ./image_output/sample_normalization.jpg "Grayscaling"
[image3]: ./image_output/training_set_initial_sample_distribution.jpg "Initial_distribution"
[image4]: ./new_images/example_00005.jpg "Traffic Sign 1"
[image5]: ./new_images/example_00019.jpg "Traffic Sign 2"
[image6]: ./new_images/example_00021.jpg "Traffic Sign 3"
[image7]: ./new_images/example_00032.jpg "Traffic Sign 4"
[image8]: ./new_images/example_00034.jpg "Traffic Sign 5"
[image9]: ./image_output/image_aug_sample.jpg "Augmented Images"
[image10]: ./image_output/augmented_distribution.jpg "Augmented Sample Distribution"
[image11]: ./image_output/signs_from_web.jpg "Signs from Web"
[image12]: ./image_output/top_5.jpg "Top 5 Predictions"

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mscwu/udacity_traffic_sign_classification/blob/master/Traffic_Sign_Classifier.ipynb)

#### Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

      I used the numpy to calculate summary statistics of the traffic signs data set:

      * The size of training set is 34799
      * The size of the validation set is 4410
      * The size of test set is 12630
      * The shape of a traffic sign image is 32x32x3
      * The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.
  Here is an exploratory visualization of the data set.
  * Sample Distribution of Training Dataset

![alt text][image3]
  * Samples of Training Dataset

![alt text][image1]


#### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As a first step, I decided to convert the images to grayscale. The reason was that we didn't want color to be recognized as a feature by our model since it made the model more complex but did not provide increased accuracy. Just like human beings don't need to use the color to distinguish a sign from another, neither does our CNN model.  

* The next step is to implement histogram equalization on the images to reduce the effect of dynamic range and make the test set images easier for the model to train. I tried three methods here. The first was "equalizeHist" function in OpenCV library. This method provided basic equalizing but the result was not so desirable. The second was "equalize_adapthist" function in SKImage library. The results seemed better but the computation time was too long. The third one, which was also the one I adopted in the project, was the "clahe" function in OpenCV library. "CLAHE" stands for "Contrast Limited Adaptive Histogram Equalization" which is an algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image. I used 8x8 tiles for the picture and a cliplimit of 40. The higher the cliplimit, the higher the contrast. 
  Here is an example of a traffic sign image before and after grayscaling and histogram equalization.  
![alt text][image2]

* As a last step, we need to increase the amount of training images in the dataset. As is shown in the bar chart above, the dataset is very unbalanced. The most frequently observed class has 10 times more training data than the least ones. In order to balance the dataset, it is necessary to artificially augment the dataset.  

2. To add more data to the the data set, I used the following steps.  
* Determine whether images of a class needs augmentation. At first, I used a threshold of 1500 meaning that if the images for a certain class is less than 1500, image augmentation will be applied to that class. All images in that class will be used to generate new data first and then a few of them are selected randomly to fill the blanks and make the totoal to 1500. This turned out to be not sufficient for training. After a few trials, I used 7000 as a threshold for each class.
* Randomly shear the images to create more perspective view. +/- 6 degrees is used in my implementation.
* Randomly rotate the images. It is important to keep in mind that the rotation should be within reasonable range. For example, extremely speaking, a "Keep Left" sign flipped by 180 degrees becomes a "Keep Right" sign. Although, if we intentionly do this and change the label accordingly, it will contribute to the data augmentation. However, I did not include this as we have other ways to generate more data too. +/- 12 degrees is used in my implementation.
* Randomly transform the images. Again, there is a limit on how much one can move the image until features in the images are lost. For example, "Bumpy Road" sign has the bumpy road shape at the bottom of the sign. If the images is shifted down too much and the bumpy road it self is clipped, it renders the training pictures useless. +/- 3 pixels is used in my implementation.
* Randomly scale the images. The images are scaled between 100% and 110%.  

  Here is an example of an original image and an augmented image for each class that needs to be augmented:  

![alt text][image9]

  As a result of data augmentation, the training data set is increased by 266201 images, which almost doubles the amount of data avaiable.  
  New sample distribution looks like this.  
![alt text][image10]


2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

  My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32      									|
| RELU					|		
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x32 				|
| Flatten	      	| outputs 800 				|
| Fully connected		| outputs 256       									|
| RELU					|
| Dropout		| Keep Probability 0.4      									|
| Fully connected		| outputs 64       									|
| RELU					|	
| Dropout		| Keep Probability 0.4      									|
| Fully connected		| outputs 43       									|
| Softmax				|         									|
 


3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Optimizer: AdamOptimizer
* Number of epochs: 40
* Learning rate: 0.0006
* Dropout keep probability: 0.4

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

  My final model results were:
  * training set accuracy of 1.0
  * validation set accuracy of 0.974
  * test set accuracy of 0.952

  An iterative approach was chosen:
  * The first architecture that was tried and why was it chosen?
The model architecture was based on the LeNet model as a starting point. LeNet is a simple CNN that works well for MINIST dataset so it will be our baseline.
  * What were some problems with the initial architecture?
The problem with the initial architecture was the lack of dropout layer made it prone to overfitting the training dataset. The lack of features in the first convolution layer also made it less capcable of predicting traffic signs.
  * How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * Which parameters were tuned? How were they adjusted and why?
    * The depth of the first and second convolution layers were adjusted. They were expanded from 6 and 16 to 16 and 32. The reason was that 6 feature maps might be good enough for numbers with only 10 classes. However, for a 43-class traffic sign dataset and human, vehicles, shapes to be recognized, 6 and 16 features were far from enough. This was the most significant deviation we had from the original LeNet model we had in this project.
    * Since the depth of the convolution layers were changed, the width of the fully connected layers were also adjusted accordingly.
    * Learning rate was set to be 0.0006. At first, 0.001 was chosen as the rate by I found the model was overfitting as the result converged to a high validation loss and low traing loss. I lowered the learning rate and increased the number of epoches to make the model learn longer and slower.
    * Dropout rate of 0.6 was used. I started with 0.2 but the overfitting is obvious. I gradually lowered the keep probability until a dropout rate of 0.6 was reached and the model performed well.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Important design changes include adding more depth to the convolution layer and the addition of dropout layer. The former is discussed above. The addition of dropout layer is to prevent the model from overfitting. By not passing the full output of one fully connected layer to another, we reduced the amount of data the model could "see" during each iteration. Thus reducing the risk of overfitting.

  If a well known architecture was chosen:
  * What architecture was chosen?
    LeNet was chosen.
  * Why did you believe it would be relevant to the traffic sign application?
    LeNet performs well for numbers and some traffic signs contains numbers too. LeNet is also not too deep and takes less time to train.
  * How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The model has a training accuracy of 1.0, validation accuracy of 0.974 and test accuracy of 0.952. There is no significant under or overfitting present. 

#### Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

  Here are five German traffic signs that I found on the web:

![alt text][image11]



  * The first image might be difficult to classify because there is some kind of triangular shape behind the sign which might make the model confused.  
  * The second image might be difficult to classify because of the vertical pole that is behind the sign.  
  * The third image might be difficult to classify because the background color is very close to the edge of the sign.  
  * The fourth image might be difficult to classify because the sign is enclosed in a square boundary.  
  * The fifth image might be difficult to classify because part of the boundary shares the same color as the background and the feature of the sign-vertical narrowing road is very close to that of the general warning sign-an exclamation sign.  

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Keep Right     			| Keep Right 										|
| Right of Way at Next Intersection					| Right of Way at Next Intersection											|
| Roundabout Mandatory	      		| No Vehicles					 				|
| Road Narrows on the Right			| Road Narrows on the Right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is a good result considering that these images are chosen to delibrately confuse the network.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here is a graph showing how confident the model is at predicting each sign.

![alt text][image12]

It can be seen that for 4 out of 5 signs, the network is almost certain that it predicts them correctly. For the fourth sign, "Roundabout Mandatory", the model mixes that with a few other signs. However, judging by the sign that it predict, the networks recognizes the round shape in the sign. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


