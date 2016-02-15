---
layout: post
title: Using deep learning to read street signs
published: True
---

Show me the [code](https://github.com/FlorianMuellerklein/Chars74k_CNN)!

The [First Steps with Julia](https://www.kaggle.com/c/street-view-getting-started-with-julia) competition on Kaggle uses a subset of the [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) which consist of a series of characters cropped from natural images. This dataset represents a very logical step for computer vision, trying to read text in the real world. Although the Kaggle competition was set up to introduce the [Julia programming language](http://julialang.org/) it also serves as a great image classification dataset which deep learning is well suited for. I chose to tackle this problem using python and convolution neural networks.

{: .center}
![chars74k](https://kaggle2.blob.core.windows.net/competitions/kaggle/3947/media/chars74k.jpg)

<p style="text-align:center; font-size:75%; font-style: italic;">Examples of images, taken from the Kaggle and the Chars74k website</p>

Convolution neural networks have been the top performers in computer vision since 2012. Changes to network architecture and data processing are being made all the time that are steadily increasing the performance of image classification. This network is inspired by the ImageNet winners of 2014. It is a [VGG](http://arxiv.org/abs/1409.1556) style convolution neural network with heavy data augmentation. They used 'networks of increasing depth using an architecture with very small (3x3) convolution filters'. On this dataset, pairing the VGG-style network with heavy data augmentation currently gets 83.3% on a holdout validation dataset of 6,220 images and [first place](https://www.kaggle.com/c/street-view-getting-started-with-julia/leaderboard) on the Kaggle leaderboards.

## Image pre-processing

Very little pre-processing was done to the images. The power and flexibility of deep learning comes from the algorithm doing it's own 'feature engineering' with raw data.

However, the images initially vary in size a lot. Some of the smaller images are 14 by 29 pixels and some of the larger ones can be as big as 178 by 197 pixels. I simply rescaled all of the images to 64 by 64 pixels with [Imagemagick](http://www.imagemagick.org/script/index.php). Although this approach will not preserve the scale of many of the images, it's a sort of a trade off between not losing information in the larger images and preserving the smaller images by scaling them up.

Additionally, the images were all converted to grayscale because the color information should have no impact on the networks ability to recognize letter shapes. So the final input to the network will be 64 by 64 pixel images with only one grey channel.

## Architecture

I am using 6 convolution layers with filter size 3x3 and ReLU activations. Max pooling layers after every other convolution layer. After the convolution layers I am using 2 hidden layers with dropout and a 62 way softmax output.


| __Layer Type__ | __Channels__ | __Size__ |
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

The network was trained with stochastic gradient descent (SGD) and Nesterov momentum fixed at 0.9. Training was done in 300 iterations with an initial learning rate of 0.03, after 250 epochs the learning rate was dropped to 0.003 and then dropped again to 0.0003 after 275 epochs. This allowed the network to fine-tune itself with smaller updates once the classification accuracy got very high.

{: .center}
![training_plot](http://i.imgur.com/nFy2C3P.png)

<p style="text-align:center; font-size:75%; font-style: italic;">Training plot showing categorical cross entropy and percent correct on validation set</p>


## Data augmentation

One of the drawbacks to how powerful deep learning can be is that the networks are very prone to overfitting. These data augmentation techniques are a great way to deal with that. Data augmentation allows the network to see each image in different ways which in turn allows the network to have a much more flexible understanding of each class of image. In this dataset, the characters are taken from natural images so the text on signs can come in any shape and it is especially important for the network to be flexible. In my opinion the data augmentation is more important than the network architecture.

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. This ensures that the network will never see the same variation of each image twice allowing the network to better generalize.

* Random rotations between -10 and 10 degrees.
* Random translation between -10 and 10 pixels in any direction.
* Random zoom between factors of 1 and 1.3.
* Random shearing between -25 and 25 degrees.
* Bool choice to invert colors.
* Sobel edge detector applied to 1/4 of images.

{: .center}
![Original](http://i.imgur.com/vNkJrKi.png =128x)![Augmented](http://i.imgur.com/0G8Khxv.gif =128x)

<p style="text-align:center; font-size:75%; font-style: italic;">On the left is the original image, on the right are possible variations that the network can receive as input.</p>


Here is the code for the data augmentation batch iterator. It mostly uses [scikit-image](http://scikit-image.org/) for all of the image processing. For a great example on how to implement a similar batch iterator into a python neural network see Daniel Nouri's tutorial [here](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

{: .code}
``` python
from skimage import transform, filters, exposure

PIXELS = 64
imageSize = PIXELS * PIXELS
num_features = imageSize

# much faster than the standard skimage.transform.warp method
def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params,
                                          output_shape=output_shape, mode=mode)

def batch_iterator(data, y, batchsize, model):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    rotate all images in a given batch between -10 and 10 degrees
    random translations between -10 and 10 pixels in all directions.
    random zooms between 1 and 1.3.
    random shearing between -25 and 25 degrees.
    randomly applies sobel edge detector to 1/4th of the images in each batch.
    randomly inverts 1/2 of the images in each batch.
    '''

    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS),
                               dtype = 'float32')

        # random rotations betweein -10 and 10 degrees
        dorotate = randint(-10,10)

        # random translations
        trans_1 = randint(-10,10)
        trans_2 = randint(-10,10)

        # random zooms
        zoom = uniform(1, 1.3)

        # shearing
        shear_deg = uniform(-25, 25)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (trans_1, trans_2))

        tform = tform_center + tform_aug + tform_uncenter

        # images in the batch do the augmentation
        for j in range(X_batch.shape[0]):

            X_batch_aug[j][0] = fast_warp(X_batch[j][0], tform,
                                          output_shape = (PIXELS, PIXELS))

        # use sobel edge detector filter on one quarter of the images
        indices_sobel = np.random.choice(X_batch_aug.shape[0],
                                         X_batch_aug.shape[0] / 4, replace = False)
        for k in indices_sobel:
            img = X_batch_aug[k][0]
            X_batch_aug[k][0] = filters.sobel(img)

        # invert half of the images
        indices_invert = np.random.choice(X_batch_aug.shape[0],
                                          X_batch_aug.shape[0] / 2, replace = False)
        for l in indices_invert:
            img = X_batch_aug[l][0]
            X_batch_aug[l][0] = np.absolute(img - np.amax(img))

        # fit model on each batch
        loss.append(model.train_on_batch(X_batch_aug, y_batch))

    return np.mean(loss)
```

### References

* Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", [link](http://arxiv.org/abs/1409.1556)
* Ren Wu, Shengen Yan, Hi Shan, Qingqing Dang, Gang Sun, "Deep Image: Scaling up Image Recognition", [link](http://arxiv.org/vc/arxiv/papers/1501/1501.02876v1.pdf)
* Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* Sander Dieleman, "Classifying plankton with deep neural networks", [link](http://benanne.github.io/2015/03/17/plankton.html)
