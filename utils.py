import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:
        w_class = median_freq / freq_class,
    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq


def add_mask_to_source_multi_classes(source_np, mask_np, num_classes):
    colors = [[0, 0, 0], [255, 0, 255], [255, 255, 0], [0, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 0]]
    foreground_mask_bool = mask_np.astype('bool')
    foreground_mask = mask_np * foreground_mask_bool
    foreground = np.zeros(source_np.shape, dtype='uint8')
    background = source_np.copy()

    for i in range(1, num_classes + 1):
        fg_tmp = np.where(foreground_mask == i, 1, 0)
        fg_tmp_mask_bool = fg_tmp.astype('bool')

        fg_color_tmp = np.zeros(source_np.shape, dtype='uint8')
        fg_color_tmp[:, :] = colors[i]
        for c in range(3):
            fg_color_tmp[:, :, c] *= fg_tmp_mask_bool
        foreground += fg_color_tmp
    foreground = cv2.addWeighted(source_np, 0.0, foreground, 1, 0)

    for i in range(3):
        foreground[:, :, i] *= foreground_mask_bool
        background[:, :, i] *= ~foreground_mask_bool

    show = foreground
    return show


def add_mask_to_source(source_np, mask_np, color):
    mask_bool = (np.ones(mask_np.shape, dtype='uint8') & mask_np).astype('bool')

    foreground = np.zeros(source_np.shape, dtype='uint8')
    for i in range(3):
        foreground[:, :, i] = color[i]
    foreground = cv2.addWeighted(source_np, 0.5, foreground, 0.5, 0)

    background = source_np.copy()
    for i in range(3):
        foreground[:, :, i] *= mask_bool
        background[:, :, i] *= (~mask_bool)

    return background + foreground


if __name__ == "__main__":
    pass
