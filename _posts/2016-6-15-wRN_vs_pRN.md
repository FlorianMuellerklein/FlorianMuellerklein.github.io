---
layout: post
title: Reproducing CIFAR-10 Results from Deep and Wide Preactivation Residual Networks
published: False
---

Since their introduction as the winning solutions to ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation Deep Residual Networks have made a huge splash in the machine learning community. In addition to their amazing performance they also introduced a way to train extremely deep neural networks, up to 1000 or more layers. The main idea is that Residual Networks ["reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions."](https://arxiv.org/abs/1512.03385) The basic idea is that the layers are reformulated as residual blocks which are basically two branches that come together with an element-wise addition. One branch of the residual block is a stack of two convolution layers and the other is a simple identity function.

{: .center}
![Residual Block](https://qph.is.quoracdn.net/main-qimg-b1fcbef975924b2ec4ad3a851e9f3934?convert_to_webp=true)
<p style="text-align:center; font-size:75%; font-style: italic;">Diagram of the residual block taken from the the very first residual network paper.</p>

Just like normal convolution layers, these blocks can be layered to create networks of increasing depth. One of the reasons that these networks became popular was because we are now able to easily (with the right hardware) train networks of up to 1000 layers. However, there does seem to be diminishing returns when increasing the depth. Below is the basic structure of the CIFAR-10 residual network, with the depth being controlled by a multiplier *n* which dictates how many residual blocks to insert between each downsampling layer.

| Group | Size | Multiplier |
| ------|:------:|:----------:|
| Conv1 | [3x3, 16] | - |
| Conv2 | [3x3, 16]<br>[3x3, 16] | n |
| Conv3 | [3x3, 32]<br>[3x3, 32] | n |
| Conv4 | [3x3, 64]<br>[3x3, 64] | n |
| Avg-Pool | 8x8 | - |
| Softmax  | 10 | - |

<p style="text-align:center; font-size:75%; font-style: italic;">Basic structure of the CIFAR-10 Residual Network. An initial convolution layer is followed by residual blocks of two 3x3 convolutions which are parallel to identity mappings, the output of the identity and convolution stacks are added after each block. The depth is mostly altered by the multiplier n which defines how many residual blocks to use in each section.</p>

# Methods

This reproduction will focus on two recent improvements on the original residual network design, [Preactivation Residual Networks](https://arxiv.org/abs/1603.05027) and [Wide Residual Networks](https://arxiv.org/abs/1605.07146). The preactivation architecture switches up the order of the convolution, batch normalization and nonlinearities within each residual block. The wide architecture simply increases the number of convolution kernels within each preactivation residual block.

## Preactivation Residual Blocks

[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) introduces the preactivation architecture which changes the order of the batch normalizations and nonlinearities creating the preactivation residual network. The original residual network contains residual blocks of convolution -> batch normalization -> ReLU -> convolution -> batch normalization. Those blocks are then added to the identity function that takes the input to the residual block. After the addition another ReLU is applied which then defines the input to the next residual block. The preactivation architecture changes the order to batch normalization -> ReLU -> convolution -> batch normalization -> ReLU -> convolution. Again, the output of that residual block is added to the same type of identity function to create the input to the next residual block.

{: .center}
![PreResNet](https://qiita-image-store.s3.amazonaws.com/0/100523/a156a5c2-026b-de55-a6fb-e4fa1772b42c.png)
<p style="text-align:center; font-size:75%; font-style: italic;">The alterations from the original ResNet to the Preactivation ResNet.</p>

## Wide Residual Networks

Those preactivation residual etworks are  very deep but also very thin, so another very recent paper is [Wide Residual Networks](https://arxiv.org/abs/1605.07146) which makes the network much more shallow but also much wider. The wide-ResNet simply adds another multiplier *k* that increases the number of filters used in each residual block. The idea of adding dropout in between the two convolution layers is also introduced with the wider residual blocks. The basic structure can be seen below.

| Group | Size | Multiplier |
| ------|:------:|:----------:|
| Conv1 | [3x3, 16] | - |
| Conv2 | [3x3, 16 x k]<br>[3x3, 16 x k] | n |
| Conv3 | [3x3, 32 x k]<br>[3x3, 32 x k] | n |
| Conv4 | [3x3, 64 x k]<br>[3x3, 64 x k] | n |
| Avg-Pool | 8x8 | - |
| Softmax  | 10 | - |

{: .center}
![WideResNet](http://i.imgur.com/3b0fw7b.png)
<p style="text-align:center; font-size:75%; font-style: italic;">An example of the wide ResNet, it's basically a Preactivation ResNet with increased filters count in the residual blocks and optional dropout.</p>

## Training and Testing

Both the original residual network and follow up preactivation residual network use identical preprocessing, training and regularization parameters. However, the wide residual paper uses different preprocessing, training, and regularization while still comparing results to the previous preactivation residual network. It is possible that different network architectures would require different parameters and input data but there should also be comparisons done where everything stays exactly the same except for the networks. This reproduction will be done using the preprocessing, training and regularization parameters from the original and preactivation residual network papers.

**Preprocessing:** The only preprocessing done to the CIFAR-10 images are per-pixel mean subtraction.

**Data Augmentation:** As the data were fed into the network it was zero-padded with 4 pixels on every side and a random crop was taken from the expanded image of the original size of 32x32. This effectively results in random translations. Additionally, a horizontal flip was applied with probability 0.5.

**Training and regularization:** The networks were trained with 200 epochs (full passes through training dataset) with stochastic gradient descent and nesterov momentum of 0.9. The initial learning rate was set to 0.01 to warm up the network and was increased to 0.1 at epoch 10. The learning rate was adjusted by the following schedule {0:0.01, 10: 0.1, 80: 0.01, 120: 0.001}. L2 regularization of 0.0001 was used.

# Results

Results are presented as classification error percent.

| __ResNet Type__ | __Original Paper__ | __Test Results__ |
| :---------:|:---------:|:---------: |
| ResNet-110 | 6.37 | 6.38 |
| ResNet-164 | 5.46 | 5.66 |
| WResNet-n2-k4<br>with dropout 0.3 | 5.55 | 5.41 |
| WResNet-n3-k8<br>no dropout | 4.38 | 5.02 |
| WResNet-n3-k8<br>with dropout 0.3 | - |  |

<p style="text-align:center; font-size:75%; font-style: italic;">All results are presented from the first and only training run. I did not run each network multiple times and choose the best score.</p>

I was able to reproduce the results of the two papers within a reasonable range. However since I used different preprocessing, training and regularization for the wide residual networks I did not expect them to match. The trends held, wide residual networks still outperformed the normal preactivation networks. Interestingly dropout seemed to have more benefit here than in the original wide residual network paper. Perhaps due to the less aggressive L2 (0.0001 vs 0.0005) used here the regularization from dropout could have a more significant benefit.

### References

* [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Networks for Image Recognition", [link](https://arxiv.org/abs/1512.03385)
* [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", [link](https://arxiv.org/abs/1603.05027)
* [3] Sergey Zagoruyko, Nikos Komodakis, "Wide Residual Neural Networks", [link](https://arxiv.org/abs/1605.07146)
