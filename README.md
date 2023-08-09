# Simple-Attention
This is the code base for **`"CS-Net: Conv-Simpleformer Network for agricultural image segmentation"`**

## Abstract
The agricultural industry faces numerous challenges, including incomplete sensor networks, long image acquisition cycles, and poor image quality, resulting in limited prediction accuracy and inadequate generality of conventional semantic segmentation models. 

To address this, we propose Conv-Simpleformer Network (CS-Net), which fuses CNN and Transformer. **`A lightweight Simple-Attention Block (SIAB) is designed to solve the problem of quadratic correlation between the computational complexity of Self-Attention and image size`**. **`SIAB uses channel-by-channel and spatial convolution computation, and its computational complexity is linearly related to image size`**. 

Additionally, this study cascades an FFN after SIAB to form a Simpleformer block, enhancing its spatial modeling capability. 

Finally, we performed comparison and ablation experiments on agricultural image datasets. The experimental results showcase CS-Net's superiority over other control models across various agricultural image segmentation tasks. The cascaded CNN and Simpleformer blocks effectively harness SIAB's potential to extract deep features and uncover hidden connections between targets.

## Design of the Proposed *Simple-Attention Block(SIAB)*
Although self-attention block(SEAB) effectively encodes spatial information and extracts global information, the similarity score matrix A has a shape of *NxN*, which causes the **computational complexity of SEAB to grow quadratically with the increase in input image size**.

**Element-by-element computation in SEAB is redundant**, so we design an efficient and lightweight attention block called Simple-Attention Block (SIAB). **SIAB converts the `element-by-element calculation` operation of SEAB into `channel-by-channel and spatial convolution calculations`**
![computational complexity](figures/computational complexity.png)

## Design of the Proposed *Conv-Simpleformer Network(CS-Net)*
Referring to the design idea of the classical Transformer module, to enhance the modeling ability of the location information of the target to be detected and to enrich the feature representation ability, the FFN is cascaded after the SIAB to form a Simpleformer block.
The overall architecture of CS-Net, is in the shape of a letter "V" and consists of two parts: an encoder and a decoder, which generate segmentation results end-to-end.
CS-Net extracts the global representation and local features of the input image and retains the detailed features of the image through the combination of encoder and decoder, thus improving the accuracy and efficiency of image segmentation.

## Results

## Requirements
- The code has been written in Python (3.9.16) and requires pyTorch (version 2.0.1)
- Install the dependencies using pip install -r requirements.txt

## Preparing your data
You have to split your data into three folders: train/val/test. Each folder will contain two sub-folders: Img and GT, which contain the png files for the images and their corresponding ground truths. The naming of these images is important, as the code to save the results temporarily to compute the 3D DSC, for example, is sensitive to their names.
