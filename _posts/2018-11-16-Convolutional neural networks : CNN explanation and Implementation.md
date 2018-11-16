---
layout: post
title: Convolutional neural networks : CNN explanation and Implementation
published: False
---

![Imgur](https://i.imgur.com/Qd9JpWj.png)

Convolutional neural network ( CNN ) is a type of neural network architecture specially made to deal with visual data. 
In this article we will discuss the architecture of CNN and implement it on CIFAR-10 dataset. The main benefit of using a CNN over simple ANN on visual data is that CNN’s are constrained to deal with image data exclusively. 
Two main features of CNNs are
1. Weight sharing 
2. Feature extractors

![Imgur](https://i.imgur.com/x0DIK9L.jpg)

Several new layers are introduced in CNNs to extract the useful features from our image or reducing the size of image without using the
original representation. 

### Convolution Layer - 

Convolutional layer apply convolution operation on the input layer, passing the results to next layer. A convolution operation is basically computing a dot product between their weights and a small region 
they are connected(currently overlapping) to in the input volume. This will change the dimensions depending on the filter size used and number of filters used. 

![Imgur](https://i.imgur.com/tRdPkca.gifv)


### ReLU Layer

Rectifying Linear Unit (ReLU) layer applies the relu activation element-wise. It is a mathematical function, which returns a positive value or 0 in place of previous negative values :

{\displaystyle f(x)=x^{+}=\max(0,x)}

It does not change the dimensions of the previous layer.

### Pooling Layer

Pooling layer will perform a down-sampling operation along the width and resulting in the reduction of the dimensions. The sole purpose of pooling is to reduce spatial dimensions.
There are various types of pooling in which the most common is Max Pooling, i.e taking the maximum element from the window.

![Imgur](https://i.imgur.com/5Q54piT.png)


### Stride

Stride decides by how much we move our window ,when we have a stride of one we move across and down a single pixel. With higher stride values, we move large number of pixels at a time and hence produce smaller output volumes.

### Padding

Padding is used to preserve the boundary information , since without padding they are only traversed once.






