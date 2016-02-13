---
layout: post
title: Using convolution neural networks to read street signs
published: True
---

The [First Steps with Julia](https://www.kaggle.com/c/street-view-getting-started-with-julia) competition on Kaggle uses the [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) which consist of a series of characters cropped form google street view images. This dataset represents a very logical step for computer vision, trying to read signs and text in the real world. Although the Kaggle competition was set up to introduce the [Julia programming language](http://julialang.org/) it also serves as a great image classification dataset which deep learning is especially suited for. I chose to tackle this problem using python and convolution neural networks.

Convolution neural networks have been the state of the art in computer vision since 2012. There are slight changes to network architectures and data processing being made all the time. These changes have been steadily increasing the performance of image classification since CNNs have gotten popular. This is a VGG style convolution neural network with heavy data augmentation for the case sensitive character recognition Chars74k dataset. Currently gets 83.3% on holdout validation dataset of 6,220 images and gets [first place](https://www.kaggle.com/c/street-view-getting-started-with-julia/leaderboard) on the Kaggle leaderboards.

## Architecture

The input are 64 x 64 greyscale images
6 convolution layers with filter size 3x3 and ReLU activations. Max pooling layers after every other convolution layer. 2 hidden layers with dropout. Softmax output.


| __Layer Type__ | __Channels__ | __Parameters__ |
| :---: | :---: | :---: |
| Input | 1 | 64x64 |
| Convolution | 128 | 3x3 |
| Convolution | 128 | 3x3 |
| Max pool | - | 2x2 |
| Convolution | 256 | 3x3 |
| Convolution | 256 | 3x3 |
| Max pool | - | 2x2 |
| Convolution| 512 | 3x3 |
| Convolution | 512 | 3x3 |
| Max pool | - | 2x2 |
| Fully connected | 2048 | - |
| Dropout | 2048 | 0.5 |
| Fully connected | 2048 | - |
| Dropout | 2048 | 0.5 |
| Softmax | 62 | - |


## Training Algorithm

The nework was trained with stochastic gradient descent (SGD) and Nesterov momentum fixed at 0.9. Training was done in 300 iterations with an initial learning rate of 0.03, after 250 epochs the learning rate was dropped to 0.003 and then dropped again to 0.0003 after 275 epochs. This allowed the network to fine-tune itself with smaller updates once the classification accuracy got very high.

## Data augmentation

One of the drawbacks to how powerful deep learning can be is that the networks are very prone to overfitting. These data augmentation techniques are a great way to deal with that. Data augmentation allows the network to 'see' each image in different ways which in turn allows the network to have a much more flexible 'understanding' of each class of image. In this dataset, where the images are taken from the google streetview car and text on signs can come in any size and font it is especially important for the network to be flexible.

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. This ensures that the network will never see the same variation of each image twice allowing the network to better generalize.

* Random rotations between -10 and 10 degrees.
* Random translation between -10 and 10 pixels in any direction.
* Random zoom between factors of 1 and 1.3.
* Random shearing between -25 and 25 degrees.
* Bool choice to invert colors.
* Sobel edge detector applied to 1/4 of images.

On the left is the original image, on the right are possible variations that the network can receive as input.
![Imgur](http://i.imgur.com/vNkJrKi.png)![Imgur](http://i.imgur.com/0G8Khxv.gif)

### References

* Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", [link](http://arxiv.org/abs/1409.1556)
* Ren Wu, Shengen Yan, Hi Shan, Qingqing Dang, Gang Sun, "Deep Image: Scaling up Image Recognition", [link](http://arxiv.org/vc/arxiv/papers/1501/1501.02876v1.pdf)
* Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* Sander Dieleman, "Classifying plankton with deep neural networks", [link](http://benanne.github.io/2015/03/17/plankton.html)
